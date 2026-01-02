from functools import partial
from types import MethodType
from typing import Optional

import torch
from torch import Tensor

from ..utils import nag, cat_context, check_nag_activation, NAGSwitch


class NAGNextDiT:
    """
    NAG wrapper for NextDiT (Lumina) model.

    This class provides methods that can be bound to a NextDiT model instance
    to enable Normalized Attention Guidance (NAG) during inference.

    Lumina/NextDiT uses a unified transformer architecture where context and
    image tokens are processed together. NAG is applied by:
    1. Duplicating the batch with negative context
    2. Running the model on both positive and negative conditions
    3. Applying NAG guidance to extrapolate between the outputs
    """

    @staticmethod
    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        num_tokens: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
        transformer_options: dict = {},
        # NAG parameters
        nag_negative_context: Optional[Tensor] = None,
        nag_negative_y: Optional[Tensor] = None,
        nag_sigma_end: float = 0.,
        nag_scale: float = 5.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass with NAG support for NextDiT/Lumina models.

        This method runs the model with both positive and negative conditions,
        then applies NAG guidance to the outputs.
        """
        apply_nag = check_nag_activation(transformer_options, nag_sigma_end)

        if apply_nag and nag_negative_context is not None:
            origin_batch_size = x.shape[0]
            nag_batch_size = nag_negative_context.shape[0]

            # Concatenate negative context with positive context
            context_combined = cat_context(context, nag_negative_context, trim_context=True)

            # Handle pooled output (y parameter) if provided
            y = kwargs.get('y', None)
            if y is not None and nag_negative_y is not None:
                y_combined = torch.cat((y, nag_negative_y.to(y)), dim=0)
                kwargs = dict(kwargs)
                kwargs['y'] = y_combined

            # Duplicate the latent input for negative conditioning
            x_combined = torch.cat([x, x[-nag_batch_size:]], dim=0)

            # Duplicate timesteps
            timesteps_combined = torch.cat([timesteps, timesteps[-nag_batch_size:]], dim=0)

            # Duplicate attention mask if present
            if attention_mask is not None:
                attention_mask_combined = torch.cat(
                    [attention_mask, attention_mask[-nag_batch_size:]],
                    dim=0
                )
            else:
                attention_mask_combined = None

            # Handle num_tokens - it might need adjustment for combined context
            # For Lumina, num_tokens specifies the context token count
            if num_tokens is not None:
                # When contexts are concatenated, we need the combined length
                num_tokens_combined = context_combined.shape[1]
            else:
                num_tokens_combined = None

            # Call the original forward method with combined inputs
            try:
                output_combined = self._nag_original_forward(
                    x_combined,
                    timesteps_combined,
                    context_combined,
                    num_tokens_combined,
                    attention_mask=attention_mask_combined,
                    transformer_options=transformer_options,
                    **kwargs,
                )
            except Exception:
                # If combined forward fails, fall back to separate calls
                # This handles models that may not support batched negative conditioning
                output_positive = self._nag_original_forward(
                    x, timesteps, context, num_tokens,
                    attention_mask=attention_mask,
                    transformer_options=transformer_options,
                    **dict(kwargs, y=y) if y is not None else kwargs,
                )
                output_negative = self._nag_original_forward(
                    x[-nag_batch_size:],
                    timesteps[-nag_batch_size:],
                    nag_negative_context,
                    nag_negative_context.shape[1] if num_tokens is not None else None,
                    attention_mask=attention_mask[-nag_batch_size:] if attention_mask is not None else None,
                    transformer_options=transformer_options,
                    **dict(kwargs, y=nag_negative_y.to(y) if y is not None and nag_negative_y is not None else None) if nag_negative_y is not None else kwargs,
                )
                output_combined = torch.cat([output_positive, output_negative], dim=0)

            # Split outputs
            output_positive = output_combined[:origin_batch_size]
            output_negative = output_combined[origin_batch_size:]

            # Apply NAG guidance
            # Get the positive outputs that correspond to the negative batch
            output_pos_for_nag = output_positive[-nag_batch_size:]

            # Apply NAG
            output_guided = nag(
                output_pos_for_nag,
                output_negative,
                nag_scale, nag_tau, nag_alpha
            )

            # Replace the guided portion in the positive output
            output = output_positive.clone()
            output[-nag_batch_size:] = output_guided

            return output
        else:
            # No NAG - call original forward
            return self._nag_original_forward(
                x, timesteps, context, num_tokens,
                attention_mask=attention_mask,
                transformer_options=transformer_options,
                **kwargs,
            )


class NAGNextDiTSwitch(NAGSwitch):
    """
    Switch class to enable/disable NAG for NextDiT models.

    This class patches the model's forward method to enable NAG guidance
    and restores it when done.
    """

    def set_nag(self):
        """Enable NAG by patching the model's forward method."""
        # Store the original forward method
        self.model._nag_original_forward = self.origin_forward

        # Patch the forward method with NAG-enabled version
        self.model.forward = MethodType(
            partial(
                NAGNextDiT.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_negative_y=self.nag_negative_cond[0][1].get("pooled_output", None),
                nag_sigma_end=self.nag_sigma_end,
                nag_scale=self.nag_scale,
                nag_tau=self.nag_tau,
                nag_alpha=self.nag_alpha,
            ),
            self.model,
        )

    def set_origin(self):
        """Restore the original forward method."""
        super().set_origin()
        # Clean up the temporary method reference
        if hasattr(self.model, '_nag_original_forward'):
            del self.model._nag_original_forward
