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
        self.range_batch = torch.arange(self.BATCH_SIZE).long().to(device)
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
        z_dist = torch.from_numpy(np.array([[self.Vmin + i*self.delta_z for i in range(self.N)]]*self.BATCH_SIZE)).to(device)
        z_dist = torch.unsqueeze(z_dist, 1).float()

        Q_dist_prediction = self.qnetwork_local(states)

        Q_dist_target = self.qnetwork_target(next_states).detach()
        Q_dist_target = Q_dist_target.view(-1, self.N, self.action_size)

        Q_target = torch.matmul(z_dist, Q_dist_target).squeeze(1)
        a_star = torch.argmax(Q_target, dim=1)
        Q_dist_star = Q_dist_target[self.range_batch,:,a_star]

        m = torch.ones(self.BATCH_SIZE,self.N).to(device)
        for j in range(self.N):
            T_zj = torch.clamp(rewards + self.GAMMA * (self.Vmin + j*self.delta_z), min = self.Vmin, max = self.Vmax)
            bj = (T_zj - self.Vmin)/self.delta_z
            l = bj.floor().long()
            u = bj.ceil().long()
            mask_l = torch.zeros(m.size())
            mask_l.scatter_(1, l, 1)
            mask_Q = torch.zeros(m.size())
            mask_u = torch.zeros(m.size())
            mask_u.scatter_(1, u, 1)
            mask_Q.scatter_(1, l, Q_dist_star[:,j].unsqueeze(1))
            m = mask_l*m + mask_Q*(u.float()-bj.float())
            m = mask_u*m + mask_Q*(u.float()-bj.float())
            print(m)
            #m[indice_list_u] = m[indice_list_u] + Q_dist_star[j]*(l-bj)
            #m[indice_list_l] = m[indice_list_l] + Q_dist_star[j]*(l-bj)
        log_Q_dist_prediction = torch.log(Q_dist_prediction)
        loss = - torch.sum(torch.sum(torch.mul(log_Q_dist_prediction, m),-1),-1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

