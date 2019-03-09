import cv2
import numpy as np
import matplotlib.pyplot as plt

refPt = []
movPt = []
cropping = False
windowname = 'Select an ROI and press c to continue'
filepath = './samplevid.mp4'
filenamestart = filepath.rfind('/')
filenameend = filepath.rfind('.')
filename = filepath[filenamestart+1:filenameend]


def click_and_crop(event, x, y, flags, param):
    global refPt, movPt, cropping, windowname

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    if cropping and event == cv2.EVENT_MOUSEMOVE:
        movPt = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        movPt = []

def main():
    global refPt, movPt, windowname, filepath
    frameNum = 0
    vid = cv2.VideoCapture(filepath)

    cv2.namedWindow(windowname)
    cv2.setMouseCallback(windowname, click_and_crop)

    if vid.isOpened():
        ret, frame = vid.read()
        while True:
            framecopy = frame.copy()
            if np.size(refPt) == 4:
                cv2.rectangle(framecopy, refPt[0], refPt[1], (0, 255, 0), 1)
            elif np.size(movPt) == 2:
                cv2.rectangle(framecopy, refPt[0], movPt[0], (0, 255, 0), 1)
            cv2.imshow(windowname, framecopy)
            keypress = cv2.waitKey(1)
            if keypress == ord('c'):
                cv2.destroyWindow(windowname)
                break
            elif keypress == ord('q'):
                return
        if np.size(refPt) < 4:
            return
        else:
            if abs(refPt[0][1] - refPt[1][1]) > 0:
                yrefmin = min(refPt[0][1], refPt[1][1])
                yrefmax = max(refPt[0][1], refPt[1][1])
            else:
                yrefmin = refPt[0][1]
                yrefmax = yrefmin + 1
            if abs(refPt[0][0] - refPt[1][0]) > 0:
                xrefmin = min(refPt[0][0], refPt[1][0])
                xrefmax = max(refPt[0][0], refPt[1][0])
            else:
                xrefmin = refPt[0][0]
                xrefmax = xrefmin + 1
            framenum = frameNum + 1
            ROI = frame[yrefmin:yrefmax, xrefmin:xrefmax]
            cv2.imshow('ROI', ROI)
            cv2.waitKey(1)
            while vid.isOpened:
                ret, frame = vid.read()
                if ret:
                    ROI = frame[yrefmin:yrefmax, xrefmin:xrefmax]
                    cv2.imshow('ROI', ROI)
                    keypress = cv2.waitKey(1)
                    if keypress == ord('q'):
                        break
                    frameNum = frameNum + 1
                else:
                    break
        cv2.destroyWindow('ROI')
        print(frameNum)
        vid = cv2.VideoCapture(filepath)
        hists = np.zeros((256, 9, frameNum))
        meanpx = np.zeros((9, frameNum))
        if vid.isOpened:
            for i in range(0, frameNum):
                ret, frame = vid.read()
                ROI = frame[yrefmin:yrefmax, xrefmin:xrefmax]
                CIELAB = cv2.cvtColor(ROI, cv2.COLOR_BGR2Lab)
                HSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
                rhist = cv2.calcHist(ROI, [2], None, [256], [0, 256])
                ghist = cv2.calcHist(ROI, [1], None, [256], [0, 256])
                bhist = cv2.calcHist(ROI, [0], None, [256], [0, 256])
                cielhist = cv2.calcHist(CIELAB, [0], None, [256], [0, 256])
                cieahist = cv2.calcHist(CIELAB, [1], None, [256], [0, 256])
                ciebhist = cv2.calcHist(CIELAB, [2], None, [256], [0, 256])
                hhist = cv2.calcHist(HSV, [0], None, [256], [0, 256])
                shist = cv2.calcHist(HSV, [1], None, [256], [0, 256])
                vhist = cv2.calcHist(HSV, [2], None, [256], [0, 256])
                hists[:, 0, i] = rhist.ravel()
                hists[:, 1, i] = ghist.ravel()
                hists[:, 2, i] = bhist.ravel()
                hists[:, 3, i] = cielhist.ravel()
                hists[:, 4, i] = cieahist.ravel()
                hists[:, 5, i] = ciebhist.ravel()
                hists[:, 6, i] = hhist.ravel()
                hists[:, 7, i] = shist.ravel()
                hists[:, 8, i] = vhist.ravel()
                meanpx[0, i] = np.mean(ROI[:, :, 2])
                meanpx[1, i] = np.mean(ROI[:, :, 1])
                meanpx[2, i] = np.mean(ROI[:, :, 0])
                meanpx[3, i] = np.mean(CIELAB[:, :, 0])
                meanpx[4, i] = np.mean(CIELAB[:, :, 1])
                meanpx[5, i] = np.mean(CIELAB[:, :, 2])
                meanpx[6, i] = np.mean(HSV[:, :, 0])
                meanpx[7, i] = np.mean(HSV[:, :, 1])
                meanpx[8, i] = np.mean(HSV[:, :, 2])

                cv2.imshow("Different colour maps", np.concatenate((ROI[:, :, 2], ROI[:, :, 1], ROI[:, :, 0],
                                                                    CIELAB[:, :, 0], CIELAB[:, :, 1], CIELAB[:, :, 2],
                                                                    HSV[:, :, 0], HSV[:, :, 1], HSV[:, :, 2])))
                keypress = cv2.waitKey(10)
                if keypress == ord('q'):
                    break
            np.save('./' + filename + 'histograms.npy', hists, True)
            np.save('./' + filename + 'meanpx.npy', meanpx, True)

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
