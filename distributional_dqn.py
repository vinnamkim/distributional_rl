
# coding: utf-8

# In[28]:


import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agents.q_learner import Q_learner
from agents.distrib_learner import Distrib_learner
import matplotlib.pyplot as plt

plt.rcdefaults()

args = dict()
args["BUFFER_SIZE"] = int(500)  # replay buffer size
args["BATCH_SIZE"] = 32  # minibatch size
args["GAMMA"] = 0.95  # discount factor
args["TAU"] = 1e-3  # for soft update of target parameters
args["LR"] = 1e-3  # learning rate
args["UPDATE_EVERY"] = 4  # how often to update the network

N = 51
Vmin = 0
Vmax = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 0
env = gym.make('CartPole-v1')
env.seed(seed)
agent = Distrib_learner(N=N, Vmin=Vmin, Vmax=Vmax, state_size=env.observation_space.shape[0], action_size= env.action_space.n, seed=seed, hiddens = [100, 100], args = args)

def run_test():
    score_test = 0
    state = env.reset()
    for t in range(max_t):
        action = agent.act(state, 0)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        score_test += reward
        if done:
            break
    return score_test

def distributional_dqn(n_episodes=30000, max_t=1000, test_interval = 100, eps_start=0.5, eps_end=0.05, eps_decay=0.999):
    test = False
    scores = []                        # list containing scores from each episode
    scores_tests = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):

        if i_episode % test_interval == 0:
            test = True
        else:
            test = False

        state = env.reset()
        score = 0

        for t in range(max_t):
            if test == True:
                action = agent.act(state, 0)
            else:
                action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break

        if test:
            score_test = run_test()
            scores_tests.append(score_test)

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if test:
            print(scores_tests)
        if i_episode % 30000 == 0 or i_episode == 1:
            state = torch.from_numpy(np.array([0,0,0,0])).float().unsqueeze(0).to(device)
            q_distrib = agent.qnetwork_local.forward(state).detach()
            q_distrib = q_distrib.reshape(-1,  env.action_space.n, N)[0]

            delta_z = (Vmax - Vmin) / (N - 1)
            y = np.arange(Vmin, Vmax + delta_z, delta_z)
            z = []
            for i in range(env.action_space.n):
                z.append(q_distrib[i,:].cpu().data.numpy())
            plt.bar(y, z[0], width = 2)
            plt.bar(y, z[1], width = 2)
            plt.title("distribution at step:{}".format(i_episode))
            plt.show()

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoints/checkpoint.pth')
            break
    return scores

scores = distributional_dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

