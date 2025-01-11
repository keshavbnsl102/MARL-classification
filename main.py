import argparse
import json
from random import randint

import torch
from torch.utils.data import DataLoader, Subset
from torchnet.meter import ConfusionMeter
from torchvision import transforms
from tqdm import tqdm

from data.dataset import MNIST, Normalize
from env.agent import MultiAgent
from env.episode import retry_episode, episode
from env.observation import obs
from env.transition import translate
from networks.classify import Classify, init_weights
from networks.feature_extract import ObsFeatExtract, PoseFeatExtract
from networks.memory import LSTM
from networks.message import MessageGenNet
from networks.policy import Policy
from utils import prec_rec, save_conf_matrix, visualize_steps

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("-a", type=int, default=3, help="#agents")
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("-f", type=int, default=4)
parser.add_argument(
    "-nb", type=int, default=64, help="num of hidden units in belief LSTM"
)
parser.add_argument(
    "-na", type=int, default=64, help="num of hidden units in policy LSTM"
)
parser.add_argument("-nm", type=int, default=16, help="message size")
parser.add_argument("-nd", type=int, default=8, help="num of hidden units in state")
parser.add_argument(
    "-nlb", type=int, default=96, help="hidden size for lin proj belief LSTM"
)
parser.add_argument(
    "-nla", type=int, default=96, help="hidden size for lin proj policy LSTM"
)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--nr", type=int, default=30)
parser.add_argument("--eps", type=float, default=1)
parser.add_argument("--eps-decay", type=float, default=0.99995)
parser.add_argument("--op", type=str, default="outputs")



# img-size = 28
# nb-class = 10
# d = 2
# ft-ext mnist


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_dict = {}

def train():

    transform = transforms.Compose([transforms.ToTensor(), Normalize()])

    dataset = MNIST(transform)

    classifier_net = Classify(args.nb, 10, args.nlb).to(device)
    classifier_net.apply(init_weights)

    observer_net = ObsFeatExtract(args.f).to(device)
    observer_net.apply(init_weights)

    spatial_net = PoseFeatExtract(2, args.nd).to(device)
    spatial_net.apply(init_weights)

    msg_net = MessageGenNet(args.nb, args.nm, args.nlb).to(device)
    msg_net.apply(init_weights)

    classifier_lstm = LSTM(observer_net.output_size() + args.nd + args.nm, args.nb).to(
        device
    )
    policy_lstm = LSTM(observer_net.output_size() + args.nd + args.nm, args.na).to(
        device
    )

    policy = Policy(4, args.na, args.nla).to(device)
    policy.apply(init_weights)

    networks = [
        observer_net,
        spatial_net,
        msg_net,
        classifier_lstm,
        policy_lstm,
        policy,
        classifier_net,
    ]

    params = [p for net in networks for p in net.parameters()]

    marl_agents = MultiAgent(
        args.a,
        args.nb,
        args.na,
        args.f,
        args.nm,
        obs,
        translate,
        classifier_net,
        observer_net,
        spatial_net,
        classifier_lstm,
        policy_lstm,
        msg_net,
        policy,
    )

    indices = torch.randperm(len(dataset))
    train_indices = indices[: int(0.85 * indices.size(0))]
    test_indices = indices[int(0.85 * indices.size(0)) :]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=False,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=False,
        pin_memory=True,
    )

    optim = torch.optim.Adam(params, lr=args.lr)

    cur_time = 0

    eps = args.eps

    for epoch in range(args.epochs):

        # if epoch == 30:
        #     params = [p for net in [policy_lstm] for p in net.parameters()]
        #     optim = torch.optim.Adam(params, lr=args.lr)


        # if epoch >= 30:
        #     for net in [policy_lstm]:
        #         net.train()

        for net in networks:
            net.train()

        total_loss = 0
        index = 0

        conf_meter = ConfusionMeter(10)
        tqdm_bar = tqdm(train_dataloader)
        for x, y in tqdm_bar:
            x, y = x.to(device), y.to(device)

            r_pred, r_prob = retry_episode(
                marl_agents, x, eps, args.steps, args.nr, device, 10
            )

            # last step
            pred = r_pred[:, -1, :, :]
            y_one_hot = torch.eye(10, device=device)[y.unsqueeze(0)]

            conf_meter.add(pred.detach().mean(dim=0), y)
            r = -torch.pow(y_one_hot - pred, 2).mean(dim=-1)
            losses = r_prob.sum(dim=1).exp() * r.detach() + r

            loss = -losses.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

            precs, recs = prec_rec(conf_meter)
            if cur_time % 100 == 0:
                # print(
                #     "loss:", loss.item(),
                # "train_prec:", precs.mean().item(),
                # "train_rec:", recs.mean().item(),
                # "epsilon", eps
                # )
                pass

            tqdm_bar.set_description(
                f"Epoch {epoch} - Train, "
                f"loss = {total_loss / (index + 1):.4f}, "
                f"eps = {eps:.4f}, "
                f"train_prec = {precs.mean():.3f}, "
                f"train_rec = {recs.mean():.3f}"
            )

            eps *= args.eps_decay
            eps = max(eps, 0)

            index += 1
            cur_time += 1

        total_loss /= len(train_dataloader)
        save_conf_matrix(conf_meter, epoch, args.op, "train")


        for net in networks:
            net.eval()


        num_correct = 0
        total_num = 0
        conf_meter.reset()
        with torch.no_grad():
            tqdm_bar = tqdm(test_dataloader)
            for x, y in tqdm_bar:
                x, y = x.to(device), y.to(device)

                preds, _ = episode(marl_agents, x, 0, args.steps)
                _, predicted = preds.max(1)
                num_correct += predicted.eq(y).sum().item()
                total_num += y.shape[0]
                conf_meter.add(preds.detach(), y)
                precs, recs = prec_rec(conf_meter)
                tqdm_bar.set_description(
                     f"Epoch {epoch} - Eval, "
                    f"eval_prec = {precs.mean():.4f}, "
                    f"eval_rec = {recs.mean():.4f}"
                )

        
        data_dict[epoch] = 1 -  num_correct/total_num
        # print("test_acc:",1 -  num_correct/total_num)
        precs, recs = prec_rec(conf_meter)
        save_conf_matrix(conf_meter, epoch, args.op, "test")

    with open("outputs/{}_{}_{}.json".format(args.f, args.steps, args.a), \
        "w") as f:
        json.dump(data_dict, f)




    dataset_copy = MNIST(
        transforms.Compose([
            transforms.ToTensor()
        ])
    )
    test_dataloader_clean = Subset(dataset_copy, test_indices)
    test_dataloader = Subset(dataset, test_indices)
    test_indices = randint(0, len(test_dataloader_clean))
    visualize_steps(marl_agents, test_dataloader[test_indices][0],
        test_dataloader_clean[test_indices][0],
        args.steps, args.f, args.op,
        10, device, dataset.class_to_idx
    )





if __name__ == "__main__":
    train()
