import cv2
import numpy as np
import matplotlib.pyplot as plt

refPt = []              #will be used for coordinates when the user selects an ROI
movPt = []              #will be used to draw rectangle 'live' while the user is selecting an ROI
cropping = False        #used to decide whether to fill movPt with coordinates or not
windowname = 'Select an ROI and press c to continue'    #unnecessary with a GUI
filepath = './samplevid.mp4'                            #the video the user wants to analyse
filenamestart = filepath.rfind('/')                     #find the last / in the filepath
filenameend = filepath.rfind('.')                       #find the last . in the filepath
if filenamestart >= 0 and filenameend >= 0:
    filename = filepath[filenamestart+1:filenameend]    #lose the path and extension, we can use this name later

#This function will be associated with the ROI selection window so it can accept mouse inputs
def click_and_crop(event, x, y, flags, param):
    global refPt, movPt, cropping, windowname   #global variables, cropping isn't necessary?

    if event == cv2.EVENT_LBUTTONDOWN:              #left-click
        refPt = [(x, y)]                            #record start coordinates
        cropping = True                             
    if cropping and event == cv2.EVENT_MOUSEMOVE:   #if left mouse button is down and the mouse moves
        movPt = [(x, y)]                            #keep track of coordinates in separate list

    elif event == cv2.EVENT_LBUTTONUP:              #if user releases left mouse button
        refPt.append((x, y))                        #record end coordinates in addition to start coords
        cropping = False
        movPt = []                                  #empty the motion list

def main():
    global refPt, movPt, windowname, filepath       #global variables
    frameNum = 0                                    #for counting frames in video
    vid = cv2.VideoCapture(filepath)                #open the video

    cv2.namedWindow(windowname)                     #open an OpenCV window with the selected name
    cv2.setMouseCallback(windowname, click_and_crop)    #let the window record mouse events with the right function

    if vid.isOpened():                              #make sure the video is open before we try anything
        ret, frame = vid.read()                     #read the first frame to determine sizing and let the user select an ROI
        while True:                                 #forever
            #draw a rectangle if the user draws an ROI
            framecopy = frame.copy()    #Create a copy of the frame so we can clear the rectangle while animating it
            if np.size(refPt) == 4:     #if the refPt has start and end coordinates, draw a rectangle with those coords
                cv2.rectangle(framecopy, refPt[0], refPt[1], (0, 255, 0), 1)
            elif np.size(movPt) == 2:   #if the user is selecting an ROI, draw a rectagle with start and cursor coords
                cv2.rectangle(framecopy, refPt[0], movPt[0], (0, 255, 0), 1)
            cv2.imshow(windowname, framecopy)   #draw the frame copy and rectangle on the screen
            keypress = cv2.waitKey(1)           #the frame wouldn't appear without waitKey, and it doubles as keyboard input
            if keypress == ord('c'):            #if the user pressed 'c'
                cv2.destroyWindow(windowname)   #destroy the window
                break                           #exit this while loop
            elif keypress == ord('q'):          #if the user pressed 'q'
                vid.release()                   #close everything safely
                cv2.destroyAllWindows()
                return                          #exit
        if np.size(refPt) < 4:                  #if ROI has not been selected
            return
        else:                                   #if an ROI has been selected
            if abs(refPt[0][1] - refPt[1][1]) > 0:          #if the start and end y coordinates are separate
                yrefmin = min(refPt[0][1], refPt[1][1])     #need to find upper and lower y bounds for ROI
                yrefmax = max(refPt[0][1], refPt[1][1])     #otherwise the selection won't work
            else:                                           #if start and end coords are the same
                yrefmin = refPt[0][1]                       #select the y coord and the next one down
                yrefmax = yrefmin + 1
            if abs(refPt[0][0] - refPt[1][0]) > 0:          #if the x coords are separate
                xrefmin = min(refPt[0][0], refPt[1][0])     #find lower and upper x bounds
                xrefmax = max(refPt[0][0], refPt[1][0])     
            else:                                           #if the start and end x coords are the same
                xrefmin = refPt[0][0]                       #select the x coord and the next one along
                xrefmax = xrefmin + 1
            framenum = frameNum + 1                         #keep counting frames
            ROI = frame[yrefmin:yrefmax, xrefmin:xrefmax]   #cut frame down to selection
            cv2.imshow('ROI', ROI)                          #display the first cut frame
            cv2.waitKey(1)                                  #delay for 1 ms
            while vid.isOpened:                             
                ret, frame = vid.read()                     #attempt to read frame
                if ret:                                     #if ret is True (successful frame read)        
                    ROI = frame[yrefmin:yrefmax, xrefmin:xrefmax]   #cut the frame
                    cv2.imshow('ROI', ROI)                  #show the selection
                    keypress = cv2.waitKey(1)               #wait for 1 ms, record a keypress
                    if keypress == ord('q'):                #if the user presses 'q'
                        break                               #end this while loop
                    frameNum = frameNum + 1                 #keep count of frames
                else:                                       #if frame read was unsuccessful
                    break                                   #exit this loop
        cv2.destroyWindow('ROI')                            #destroy window that was displaying ROI
        print(frameNum)                                     #print frameNum (debug)
        vid = cv2.VideoCapture(filepath)                    #open the file again for obtaining histograms
        hists = np.zeros((256, 9, frameNum))                #set up the size of array we need for histograms
        meanpx = np.zeros((9, frameNum))                    #set up the size of array we need for mean pixel value
        if vid.isOpened:                        
            for i in range(0, frameNum):                    #now we know how many frames we are seeking
                ret, frame = vid.read()                     #read the frame
                ROI = frame[yrefmin:yrefmax, xrefmin:xrefmax]                   #cut the frame down to selection
                CIELAB = cv2.cvtColor(ROI, cv2.COLOR_BGR2Lab)                   #convert the colour map from BGR to CIELAB
                HSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)                      #convert the colour map frm BGR to HSV
                rhist = cv2.calcHist(ROI, [2], None, [256], [0, 256])           #histogram for BGR red
                ghist = cv2.calcHist(ROI, [1], None, [256], [0, 256])           #histogram for BGR green
                bhist = cv2.calcHist(ROI, [0], None, [256], [0, 256])           #histogram for BGR blue
                cielhist = cv2.calcHist(CIELAB, [0], None, [256], [0, 256])     #hist for CIELAB lightness
                cieahist = cv2.calcHist(CIELAB, [1], None, [256], [0, 256])     #hist for CIELAB green-red
                ciebhist = cv2.calcHist(CIELAB, [2], None, [256], [0, 256])     #hist for CIELAB blue-yellow
                hhist = cv2.calcHist(HSV, [0], None, [256], [0, 256])           #hist for HSV hue
                shist = cv2.calcHist(HSV, [1], None, [256], [0, 256])           #hist for HSV saturation
                vhist = cv2.calcHist(HSV, [2], None, [256], [0, 256])           #hist for HSV value
                hists[:, 0, i] = rhist.ravel()      #Now we want to store all hists in the same array to save in a file
                hists[:, 1, i] = ghist.ravel()      #ravel is necessary so each hist has elements ((256,)) instead of
                hists[:, 2, i] = bhist.ravel()      #((256,1)), which wouldn't work
                hists[:, 3, i] = cielhist.ravel()
                hists[:, 4, i] = cieahist.ravel()
                hists[:, 5, i] = ciebhist.ravel()
                hists[:, 6, i] = hhist.ravel()
                hists[:, 7, i] = shist.ravel()
                hists[:, 8, i] = vhist.ravel()
                meanpx[0, i] = np.mean(ROI[:, :, 2])    #save all mean pixel values in a separate array for saving to a file
                meanpx[1, i] = np.mean(ROI[:, :, 1])
                meanpx[2, i] = np.mean(ROI[:, :, 0])
                meanpx[3, i] = np.mean(CIELAB[:, :, 0])
                meanpx[4, i] = np.mean(CIELAB[:, :, 1])
                meanpx[5, i] = np.mean(CIELAB[:, :, 2])
                meanpx[6, i] = np.mean(HSV[:, :, 0])
                meanpx[7, i] = np.mean(HSV[:, :, 1])
                meanpx[8, i] = np.mean(HSV[:, :, 2])
                #display the ROI in all the values from the different colour maps
                cv2.imshow("Different colour maps", np.concatenate((ROI[:, :, 2], ROI[:, :, 1], ROI[:, :, 0],
                                                                    CIELAB[:, :, 0], CIELAB[:, :, 1], CIELAB[:, :, 2],
                                                                    HSV[:, :, 0], HSV[:, :, 1], HSV[:, :, 2])))
                keypress = cv2.waitKey(1)                   #wait 1 ms, accept keyboard input
                if keypress == ord('q'):                    #if the key pressed is 'q'
                    break                                   #exit the loop
            np.save('./' + filename + 'histograms.npy', hists, True)        #save all the histograms to a .npy file
            np.save('./' + filename + 'meanpx.npy', meanpx, True)           #save all mean pixel values to a .npy file

    vid.release()                                           #close video
    cv2.destroyAllWindows()                                 #close any OpenCV windows that might still exist

if __name__ == "__main__":
    main()
