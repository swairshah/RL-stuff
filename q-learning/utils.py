import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import time

def plot_learning_curve(x, scores, eps_history, filename="learning_curve.png"):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, label="1")
    ax2 = fig.add_subplot(122, label="2")

    ax1.plot(x, eps_history)
    ax1.set_xlabel("steps")
    ax1.set_ylabel("eps")
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-200):(t+1)])

    ax2.scatter(x, running_avg)
    ax2.set_ylabel('score')
    ax2.yaxis.set_label_position('right')

    plt.savefig(filename)
