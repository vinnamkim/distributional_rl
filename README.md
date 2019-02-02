#Distributional Reinforcement Learning
This repository is by Pierre-Alexandre K. and Paul-Ambroise D. and contains the PyTorch source code to reproduce the 
results of Bellemare and al. ["A Distributional Perspective on Reinforcement Learning"](https://arxiv.org/abs/1707.06887).

##Requirements
- Python 3.6
- Torch
- OpenAI gym

#Results
We used the categorical algorithm to solve [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/).

The following results were not optimized over different hyperparameters, so there is room for improvement.

![](/results/figs/test_score.png)

The evolution of the distribution for the [0, 0, 0, 0] state is the following:
![](/results/figs/gifs/seed-1.gif)
