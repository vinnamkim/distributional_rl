import torch
import matplotlib.pyplot as plt
import numpy as np

test = torch.load('results.stats')
test["ON"] = np.array(test["ON"])
test["OFF"] = np.array(test["OFF"])

plt.plot(test["ON"].mean(0), label="ON")
plt.plot(test["OFF"].mean(0), label="OFF")
plt.legend()
plt.show()