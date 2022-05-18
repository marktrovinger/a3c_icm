import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(x, scores, filename):
    running_average = np.zeros(len(scores))

    for i in range(len(running_average)):
        running_average[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_average)
    plt.title('Running Average of Previous 100 Games')
    plt.savefig(filename)