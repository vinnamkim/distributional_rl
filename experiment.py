import itertools
import gym


#TODO:
"""
- Replay Buffer
- Fonction epsilon-greedy
- Définir le réseau de neurones
- Categorical Algorithm: calcul de la loss
- Visualisation des distributions en temps réel

"""
class Experiment:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def run(self):
        for episode in range(self.args["number_episodes"]):
            state = self.env.reset()
            reward_tot = 0
            display = self.args["render"]

            for t in itertools.count():
                if display:
                    self.env.render()




                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)


                reward_tot += self.args["gamma"] * reward
                
            print("Episode n°{}, Reward:{}".format(episode, reward_tot))
        #self.env.monitor.close()