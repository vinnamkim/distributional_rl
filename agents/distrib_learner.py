##### Code taken from udacity/reinforcement learning

import numpy as np
import random
from collections import namedtuple, deque

from models.q_network import QNetwork
from utils.ReplayMemory import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Distrib_learner():

    def __init__(self, state_size, action_size, N, Vmin, Vmax,  hiddens, args, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.hiddens = hiddens
        self.BUFFER_SIZE = args["BUFFER_SIZE"]
        self.BATCH_SIZE = args["BATCH_SIZE"]
        self.GAMMA = args["GAMMA"]
        self.UPDATE_EVERY = args["UPDATE_EVERY"]
        self.LR = args["LR"]
        self.TAU = args["TAU"]
        self.N = N
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = (Vmax-Vmin)/(N-1)

        self.qnetwork_local = QNetwork(state_size, action_size * self.N, hiddens, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size * self.N, hiddens, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        z_dist = torch.Tensor(np.array([[self.Vmin + i*self.delta_z for i in range(self.N)]]*self.BATCH_SIZE)).to(device)
        z_dist = torch.unsqueeze(z_dist, 1)

        Q_dist_prediction = self.qnetwork_local(states).detach()

        Q_dist_target = self.qnetwork_target(next_states).detach()
        Q_dist_target = Q_dist_target.view(-1, self.N, self.action_size)

        Q_target = torch.matmul(z_dist, Q_dist_target).squeeze(1)
        a_star = torch.argmax(Q_target, dim=1)

        m = np.zeros(self.N)
        for j in range(self.N):
            T_zj = torch.clamp(rewards + self.GAMMA * Q_dist_target[j], min = self.Vmin, max = self.Vmax)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

