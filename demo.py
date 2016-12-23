
import numpy as np
import cv2
reft = []

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        #cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        #cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", frame)

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

step = 1
path = "smb_light/"
str_step = "%05d" % step

frame = cv2.imread(path + str_step+ ".jpg")
cv2.imshow("image", frame)
key = cv2.waitKey(0) & 0xFF
# setup initial location of window
# r,h,c,w - region of image
#           simply hardcoded the values
#c,w,r,h = 240,20, 310, 5 #Dark
c,w,r,h = refPt[0][0], refPt[1][0] - refPt[0][0], refPt[0][1], refPt[1][1] -refPt[0][1]
track_window = (c,r, w, h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]

hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))


roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    step += 1
    str_step = "%05d" % step
    frame = cv2.imread(path + str_step+ ".jpg")

    ret = True

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        result = cv2.polylines(frame, [pts], True, 255, 2)

        cv2.imshow('BackProjection',dst)
        cv2.imshow("image",result)


        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break


    else:
        break

cv2.destroyAllWindows()
