from experiment import Experiment
import gym
import torch

#define args
args = dict()
args["gamma"] = 0.98
args["number_episodes"] = 10000
args["render"] = True
args["epsilon"] = 0.05
args["update"] = 100
args["mini_batch_size"] = 100
args["tau"] = 0.01


env = gym.envs.make("MountainCar-v0")
use_cuda = True
if __name__ == '__main__':

    if use_cuda == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    #env._max_episode_steps = 10
    experiment = Experiment(env, args)
    experiment.run()
    env.close()