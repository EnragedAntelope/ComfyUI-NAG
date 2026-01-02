import torch
from torch import Tensor

from ..utils import nag


def nag_joint_attention(
    attn_positive: Tensor,
    attn_negative: Tensor,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
    num_tokens: int,
) -> Tensor:
    """
    Apply NAG guidance to joint attention output.

    Args:
        attn_positive: Attention output for positive conditioning [batch, seq_len, hidden]
        attn_negative: Attention output for negative conditioning [batch, seq_len, hidden]
        nag_scale: NAG scale factor
        nag_tau: NAG tau parameter
        nag_alpha: NAG alpha blending parameter
        num_tokens: Number of caption/context tokens (to separate from image tokens)

    Returns:
        Guided attention output
    """
    # Only apply NAG to the image tokens, not the context tokens
    img_attn_positive = attn_positive[:, num_tokens:]
    img_attn_negative = attn_negative[:, num_tokens:]

    # Apply NAG guidance
    img_attn_guidance = nag(img_attn_positive, img_attn_negative, nag_scale, nag_tau, nag_alpha)

    # Combine context tokens (unchanged) with guided image tokens
    result = torch.cat([attn_positive[:, :num_tokens], img_attn_guidance], dim=1)

    return result
