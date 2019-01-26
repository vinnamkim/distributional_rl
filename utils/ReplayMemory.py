from numpy.random import random
import numpy as np
import torch

class Memory:
    """
    This class provides an abstraction to store the [s, a, r, s', a'] elements of each iteration.
    Instead of using tuples (as other implementations do), the information is stored in lists
    that get returned as another list of dictionaries with each key corresponding to either
    "state", "action", "reward", "nextState" or "isFinal".
    """

    def __init__(self, size):
        self.size = size
        self.currentPosition = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.newBestAction = []
        self.finals = []

    def transform_to_tensor(mini_batch):
        mini_batch_size = len(mini_batch)
        next_observations_ar, observations_ar, next_actions_ar, actions_ar = np.zeros(
            (mini_batch_size, policy_net.observation_size)), np.zeros(
            (mini_batch_size, policy_net.observation_size)), np.zeros(
            (mini_batch_size, policy_net.n_actions)), np.zeros((mini_batch_size, policy_net.n_actions))

        rewards_ar = np.zeros((mini_batch_size, 1))

        for i, b in enumerate(mini_batch):
            next_observations_ar[i] = b["newState"]
            next_actions_ar[i] = b["newBestAction"]
            actions_ar[i] = b["action"]
            observations_ar[i] = b["state"]
            rewards_ar[i] = b["reward"]

        rewards = torch.Tensor(rewards_ar)
        observations = torch.Tensor(observations_ar)
        actions = torch.Tensor(actions_ar)
        next_observations = torch.Tensor(next_observations_ar)
        next_actions = torch.Tensor(next_actions_ar)
        return {"rewards": rewards,
                "observations": observations,
                "actions": actions,
                "next_observations" : next_observations,
                "next_actions": next_actions }

    def getMiniBatch(self, size):

        indices = random.sample(list(np.arange(len(self.states))), min(size, len(self.states)))
        miniBatch = []
        for index in indices:
            miniBatch.append({'state': self.states[index], 'action': self.actions[index], 'reward': self.rewards[index],
                              'newState': self.newStates[index], 'isFinal': self.finals[index]})

        return self.transform_to_tensor(miniBatch)

    def getCurrentSize(self):
        return len(self.states)

    def getMemory(self, index):
        return {'state': self.states[index], 'action': self.actions[index], 'reward': self.rewards[index],
                'newState': self.newStates[index],'isFinal': self.finals[index]}

    def addMemory(self, state, action, reward, newState, isFinal):
        if (self.currentPosition >= self.size - 1):
            self.currentPosition = 0
        if (len(self.states) > self.size):
            self.states[self.currentPosition] = state
            self.actions[self.currentPosition] = action
            self.rewards[self.currentPosition] = reward
            self.newStates[self.currentPosition] = newState
            self.finals[self.currentPosition] = isFinal
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.finals.append(isFinal)


        self.currentPosition += 1

    def cleanMemory(self):
        self.currentPosition = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.finals = []

