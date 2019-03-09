import numpy as np
import matplotlib.pyplot as plt
import time
plothists = False
plotmeanpx = True

def main():
    hists = np.load('./samplevidhistograms.npy')
    meanpx = np.load('./samplevidmeanpx.npy')
    if plotmeanpx:
        fig, axs = plt.subplots(3, 3, sharex=True, tight_layout=True)
        for i in range(0, 3):
            for j in range(0, 3):
                axs[j, i].plot(meanpx[:, 3 * i + j])
        plt.show()
    if plothists:
        plt.ion()
        fig2, axs2 = plt.subplots(3, 3, sharex=True, tight_layout=True)
        for k in range(0, np.shape(hists)[2]):
            for i in range(0, 3):
                for j in range(0, 3):
                    axs2[j, i].cla()
                    axs2[j, i].plot(hists[:, 3 * i + j, k])
            plt.draw()
            plt.pause(0.01)


if __name__ == "__main__":
    main()
