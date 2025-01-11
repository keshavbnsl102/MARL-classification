import torch


def episode(agents, img_batch, eps, max_it):
    img_sizes = list(img_batch.size()[2:])
    agents.reset_params(img_batch.size(0), img_sizes)
    for t in range(max_it):
        agents.step(img_batch, eps)

    return agents.predict()


def pos_episode(agents, img_batch, eps, max_it, device, n_classes, verbose=False):
    agents.reset_params(img_batch.size(0), img_batch.size()[2:])
    img_batch = img_batch.to(device)

    cur_pos = torch.zeros(max_it, *agents.pos.size()).long().to(device)
    cur_class = torch.zeros(max_it, img_batch.size(0), n_classes).to(device)
    cur_prob = torch.zeros(max_it, img_batch.size(0)).to(device)

    for t in range(max_it):
        agents.step(img_batch, eps)
        # print(agents.m)
        cur_pos[t, :, :] = agents.pos
        cur_class[t, :, :], cur_prob[t, :] = agents.predict()

    return cur_class, cur_prob, cur_pos


def retry_episode(agents, img_batch, eps, max_it, max_retry, device, n_classes):
    img_batch = img_batch.to(device)
    cur_class = torch.zeros(max_retry, max_it, img_batch.size(0), n_classes).to(device)
    cur_prob = torch.zeros(max_retry, max_it, img_batch.size(0)).to(device)

    # print("works")
    for r in range(max_retry):
        cur_class[r, :, :, :], cur_prob[r, :, :], _ = pos_episode(
            agents, img_batch, eps, max_it, device, n_classes
        )

    return cur_class, cur_prob
