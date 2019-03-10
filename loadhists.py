import numpy as np
import matplotlib.pyplot as plt

#select what gets plotted
bplothists = False
bplotmeanpx = True

def main():
    if bplothists:
        plothists()
    if bplotmeanpx:
        plotmeanpx()
    
def plotmeanpx():
    try:
        meanpx = np.load('./samplevidmeanpx.npy')       #load average pixel for each ROI
    except FileNotFoundError:
        print("Error: The file doesn't exist in this folder. Did you run vidloader.py already?")
        return
    except:
        print("Error: Something went wrong loading the meanpx file.")
        return
    #make a new figure with 3x3 plots (3 colourmaps with 3 parameters)
    fig, axs = plt.subplots(3, 3, sharex=True, tight_layout=True)  
    for i in range(0, 3):       #nested for loops to point to 2D indices
        for j in range(0, 3):
            axs[j, i].plot(meanpx[3 * i + j, :])    #use i and j to count from 0-8
    plt.show()                                      #figure doesn't open without this

def plothists():
    try:
        hists = load('./samplevidhists.npy')
    except FileNotFoundError:
        print("Error: The file doesn't exist in this folder. Did you run vidloader.py already?")
        return
    except:
        print("Error: Something went wrong loading the meanpx file.")
        return   
    plt.ion()               #interactive mode on, allows to redraw on same plots repeatedly
    fig2, axs2 = plt.subplots(3, 3, sharex=True, tight_layout=True)     #3 colourmaps with 3 parameters
    for k in range(0, np.shape(hists)[2]):  #nested for loops, draw all histograms for each frame
        for i in range(0, 3):                           
            for j in range(0, 3):
                axs2[j, i].cla()            #clear the axis so plots don't accumulate on top of one another
                axs2[j, i].plot(hists[:, 3 * i + j, k])     #plot all the hists for one frame
        plt.draw()                                          #draw after all plots have been set
        plt.pause(0.01)                                     #supposed to be a delay in seconds, takes far longer


if __name__ == "__main__":
    main()
