from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import torch

from env.episode import pos_episode


def prec_rec(conf_meter):
    conf_mat = conf_meter.value()

    precs_sum = [conf_mat[:, i].sum() for i in range(conf_mat.shape[1])]
    precs = np.array(
        [
            conf_mat[i, i] / precs_sum[i] if precs_sum[i] != 0.0 else 0.0
            for i in range(conf_mat.shape[1])
        ]
    )

    recs_sum = [conf_mat[i, :].sum() for i in range(conf_mat.shape[1])]
    recs = np.array(
        [
            (conf_mat[i, i] / recs_sum[i] if recs_sum[i] != 0.0 else 0.0)
            for i in range(conf_mat.shape[0])
        ]
    )

    return precs, recs



def visualize_steps(
    agents, img, img_ori,
    max_it, f, output_dir,
    nb_class, device_str,
    class_map
):
   

    idx_to_class = {class_map[k]: k for k in class_map}

    color_map = None

    preds, _, pos = pos_episode(
        agents, img.unsqueeze(0), 0.,
        max_it, device_str, nb_class
    )
    preds, pos = preds.cpu(), pos.cpu()
    img_ori = img_ori.permute(1, 2, 0).cpu()

    h, w, c = img_ori.size()

    if c == 1:
        img_ori = img_ori.repeat(1, 1, 3)

    img_idx = 0

    fig = plt.figure()
    plt.imshow(img_ori, cmap=color_map)
    plt.title("Original")
    plt.savefig(join(output_dir, f"pred_original.png"))
    plt.close(fig)


    curr_img = torch.zeros(h, w, 4)
    for t in range(max_it):

        for i in range(len(agents)):
            # Color
            curr_img[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
            pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f, :3] = \
                img_ori[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
                pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f, :]
            # Alpha
            curr_img[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
            pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f, 3] = 1

        fig = plt.figure()
        plt.imshow(curr_img, cmap=color_map)
        prediction = preds[t][img_idx].argmax(dim=-1).item()
        pred_proba = preds[t][img_idx][prediction].item()
        plt.title(
            f"Step = {t}, step_pred_class = "
            f"{idx_to_class[prediction]} ({pred_proba * 100.:.1f}%)"
        )

        plt.savefig(join(output_dir, f"pred_step_{t}.png"))
        plt.close(fig)

def save_conf_matrix(conf_meter, epoch, output_dir, stage) -> None:
    plt.matshow(conf_meter.value().tolist())
    plt.title(f"confusion matrix epoch {epoch} - {stage}")
    plt.colorbar()
    plt.ylabel("True Label")
    plt.xlabel("Predicated Label")
    plt.savefig(join(output_dir, f"confusion_matrix_epoch_{epoch}_{stage}.png"))
    plt.close()
