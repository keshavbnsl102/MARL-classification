import torch.nn.functional as F
import torch


from networks.classify import Classify
from networks.feature_extract import ObsFeatExtract


class MultiAgent:
    def __init__(
        self,
        n_agents,
        n_hidden_b,
        n_hidden_a,
        f,
        n_m,
        obs,
        transition,
        classifier_net,
        observer_net,
        spatial_net,
        classifier_lstm,
        policy_lstm,
        msg_net,
        policy,
    ) -> None:

        # agent
        self.n_agents = n_agents
        self.n_hidden_b = n_hidden_b
        self.n_hidden_a = n_hidden_a
        self.f = f
        self.n_m = n_m
        self.actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.n_actions = len(self.actions)
        self.batch_size = None
        self.msg = None
        self.action_distrib = None

        # env
        self.obs = obs
        self.transition = transition

        # networks
        self.classifier_net = classifier_net
        self.observer_net = observer_net
        self.spatial_net = spatial_net
        self.classifier_lstm = classifier_lstm
        self.policy_lstm = policy_lstm
        self.msg_net = msg_net
        self.policy = policy

        # state
        self.t = 0
        self.pos = None

        # classifier LSTM
        self.h = None
        self.c = None

        # Policy LSTM
        self.h_hat = None
        self.c_hat = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self) -> int:
        return self.n_agents

    def predict(self):
        return (
            F.softmax(self.classifier_net(self.c[-1]).mean(dim=0), dim=-1),
            self.action_distrib[-1].log().sum(dim=0),
        )

    def reset_params(self, batch_size, img_size):
        self.batch_size = batch_size
        self.t = 0
        self.h = [torch.zeros(len(self), batch_size, self.n_hidden_b).to(self.device)]
        self.c = [torch.zeros(len(self), batch_size, self.n_hidden_b).to(self.device)]
        self.h_hat = [
            torch.zeros(len(self), batch_size, self.n_hidden_a).to(self.device)
        ]
        self.c_hat = [
            torch.zeros(len(self), batch_size, self.n_hidden_a).to(self.device)
        ]
        self.msg = [torch.zeros(len(self), batch_size, self.n_m).to(self.device)]
        self.action_distrib = [
            torch.full((len(self), batch_size), 1 / len(self)).to(self.device)
        ]
        self.pos = torch.stack(
            [
                torch.randint(d - self.f, (len(self), batch_size)).to(self.device)
                for d in img_size
            ],
            dim=-1,
        )

    def step(self, img, eps):
        """
        run one time step for the agent
        """
        img_sizes = list(img.size()[2:])
        o_t = self.obs(img, self.pos, self.f)
        b_t = self.observer_net(o_t.flatten(end_dim=1)).view(
            len(self), self.batch_size, -1
        )
        mean_decoded_msg_t = self.msg[self.t].mean(dim=0)
        decoded_msg_t = (mean_decoded_msg_t * len(self) - self.msg[self.t]) / (
            len(self) - 1
        )
        pos = self.pos.float() / torch.tensor([[img_sizes]]).to(self.device)

        lamda_t = self.spatial_net(pos)

        # information input
        u_t = torch.cat((b_t, decoded_msg_t, lamda_t), dim=2)

        # classifier LSTM
        h_t_plus_1, c_t_plus_1 = self.classifier_lstm(
            self.h[self.t], self.c[self.t], u_t
        )
        self.h.append(h_t_plus_1)
        self.c.append(c_t_plus_1)

        self.msg.append(self.msg_net(self.h[self.t + 1]))

        # policy LSTM
        h_hat_t_plus_1, c_hat_t_plus_1 = self.policy_lstm(
            self.h_hat[self.t], self.c_hat[self.t], u_t
        )
        self.h_hat.append(h_hat_t_plus_1)
        self.c_hat.append(c_hat_t_plus_1)

        action_distrib = self.policy(self.h_hat[self.t + 1])
        # print("elf.h_hat[self.t + 1]", self.h_hat[self.t + 1].shape)
        # print("action_distrib", action_distrib.shape)

        actions = torch.tensor(self.actions).to(self.device)
        _, greed_action = action_distrib.max(dim=-1)
        random_action = torch.randint(
            actions.size(0), size=(len(self), self.batch_size)
        ).to(self.device)

        flip = (
            torch.rand((len(self), self.batch_size), device=self.device) > eps
        ).int()

        # print("flip", flip.shape)
        # print("random_action", random_action.shape)
        # print("greed_action", greed_action.shape)

        choosen_action = flip * greed_action + (1 - flip) * random_action

        a_t_plus_1 = actions[choosen_action.view(-1)].view(
            len(self), self.batch_size, actions.size()[-1]
        )

        self.action_distrib.append(
            action_distrib.gather(-1, choosen_action.unsqueeze(-1)).squeeze(-1)
        )

        self.pos = self.transition(
            self.pos.float(), a_t_plus_1, self.f, img_sizes
        ).long()

        self.t += 1

