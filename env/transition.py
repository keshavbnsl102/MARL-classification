import torch


def translate(pos, a_t_plus_1, f, img_size):
    """
    move the frame if possible
    else keep the original pos
    """
    mask = torch.ones_like(pos[:, :, 0]).bool()
    for d in range(pos.size(-1)):
        cur_mask = pos[:, :, d] + a_t_plus_1[:, :, d] >= 0
        cur_mask *= pos[:, :, d] + a_t_plus_1[:, :, d] + f < img_size[d]
        mask = mask & cur_mask

    mask = mask.unsqueeze(2).float()
    return mask * (pos + a_t_plus_1) + (1 - mask) * pos
