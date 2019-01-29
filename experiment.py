import itertools
from models.q_network import NN
import torch
from torch.optim import Adam
import numpy as np
from utils.ReplayMemory import Memory
from utils.utils import smooth_update

#TODO:
""""
- Visualisation des distributions en temps réel

"""





class Experiment():
    def __init__(self, env, args, device):
        self.env = env
        self.args = args
        self.device = device

    def _build_inputs(self, actions_range, state):
        inputs = []
        actions_range = np.expand_dims(np.array([0, 1, 2]), 1)
        for act_i in actions_range:
            inputs.append(torch.Tensor(np.concatenate([state, act_i])))
        inputs = torch.stack(inputs, 1)
        return inputs

    def run(self):
        test = False
        mem = Memory(size = 1000)
        Q_network = NN().to(self.device)
        target_network = NN()
        target_network.load_state_dict(Q_network.state_dict())
        optimizer = Adam(list(Q_network.parameters()))
        #actions_range = self.env.action_space.spaces.items()
        actions_range = np.expand_dims(np.array([0, 1, 2]), 1)

        for episode in range(self.args["number_episodes"]):
            epsilon = self.args["epsilon"]
            state = self.env.reset()
            reward_tot = 0
            display = self.args["render"]
            best_action = None
            for t in itertools.count():
                if display:
                    self.env.render()

                if t == 0:
                    inputs = self._build_inputs(actions_range, state)
                    Q = Q_network(inputs)
                    best_action = torch.argmax(Q).data.numpy()



                if not test:
                    #epsilon = epsilon * 0.95
                    p = np.random.binomial(1, epsilon)
                    if p:
                        best_action = actions_range[np.random.randint(0,len(actions_range)-1)][0]

                action = best_action
                next_state, reward, done, _ = self.env.step(action)

                next_inputs = self._build_inputs(actions_range, next_state)
                next_Q = Q_network(next_inputs)
                next_best_action = torch.argmax(next_Q).data.numpy()
                best_action = next_best_action

                transition = [state, action, reward, next_state, next_best_action, done]
                mem.addMemory(*transition)

                if t % self.args["update"]:
                    optimizer.zero_grad()

                    mini_batch = mem.getMiniBatch(self.args["mini_batch_size"])

                    inputs = torch.cat([mini_batch["next_observations"], mini_batch["next_actions"]], 1)

                    target = mini_batch["rewards"] + self.args["gamma"] * target_network.forward(inputs)
                    prediction = Q_network(torch.cat([mini_batch["observations"], mini_batch["actions"]], 1))

                    loss = torch.mean((target.detach()-prediction)**2)
                    loss.backward()
                    optimizer.step()
                    smooth_update(Q_network, target_network, self.args["tau"])

                reward_tot += self.args["gamma"] * reward
                state = next_state
                if t > 1000:
                    break
            print("Episode n°{}, Reward:{}".format(episode, reward_tot))

