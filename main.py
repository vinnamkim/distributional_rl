from experiment import Experiment
import gym

#define args
args = dict()
args["gamma"] = 0.98
args["number_episodes"] = 10000
args["render"] = True

env = gym.envs.make("MountainCar-v0")

if __name__ == '__main__':
    env._max_episode_steps = 10
    experiment = Experiment(env, args)
    experiment.run()
    env.close()