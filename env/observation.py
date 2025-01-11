import torch


def obs(x, pos, f):
    batch_size, channels, h, w = x.size()
    n_agents, _, _ = pos.size()
    init_pos = pos
    next_pos = pos + f

    masks = []
    for i, dim in enumerate([h, w]):
        val_range = torch.arange(dim).to(pos.device)
        mask = (init_pos[:, :, i, None] <= val_range.view(1, 1, dim)) & (
            val_range.view(1, 1, dim) < next_pos[:, :, i, None]
        )

        mask = mask.unsqueeze(-(i + 1))
        masks.append(mask)

    mask = (masks[0] & masks[1]).unsqueeze(2)
    return x.unsqueeze(0).masked_select(mask).view(n_agents, batch_size, channels, f, f)

