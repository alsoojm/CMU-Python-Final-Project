#    15-112: Principles of Programming and Computer Science
#    Project: Camera Object/Tracking, Drawing and Image Manipulation
#    Name      : Mohammad Al-Sooj
#    AndrewID  : msooj

#    File Created: 5-11-2017
#    Modification History:
#    Start              End
#    5/11   9:30 AM     5/11    1:00 PM
#    7/11   5:20 PM     7/11    10:42 PM
#    10/11  4:10 PM     10/11   9:32 PM
#    11/11  4:05 PM     11/11   11:25 PM

# This project utilizies openCV. Specifically cv2 and numpy libraries to
# access the camera and create objects from these libraries.
import cv2
import numpy as np

# -------------------------------------------------------------------------------------
#                                   Color Tracker Class
# -------------------------------------------------------------------------------------
class colorTracker():
    # Initializers
    def __init__(self):
        # Capture video feed from the primary camera
        # CHANGE THIS VALUE TO SELECT WHICH CAMERA TO CAPTURE FROM
        self.cap = cv2.VideoCapture(0)

        # Define the range to threshold the selected color
        # Green color for now. Later on the user will be able to select thier
        # desired color by clicking on the screen of the color they want to use
        # to track. Which modifies these values.
        # Here we set the lower and upper bounds of the color we want to track.
        # THE COLOR SET TO TRACK HERE IS GREEN.
        # NOTE: in OpenCV, The color are not ordered in RGB, but BGR. Keep that in mind!
        self.lowerColor = np.array([33,80,40])
        self.upperColor = np.array([102,255,255])

        # BLUE
        #self.lowerColor = np.array([110,50,50])
        #self.upperColor = np.array([130,255,255])

        # Here we create open and close kernels to remove the noise from the image.
        # This will be used to mask the object and remove and remaining noise left.
        self.kernelOpen = np.ones((5,5))
        self.kernelClose = np.ones((20,20))

        # This is the font used to display the coordinates and any text to be
        # desplayed on the screen
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Canvas for drawing. Merge into main window screen later
        self.canvas = np.zeros((480,640,3), np.uint8)

    # --------------------------------------------------------------------------------------
    #                                   run(self)
    # --------------------------------------------------------------------------------------
    # Function Description/Algorithm:                        
    #   This is the run function of this class. After reading the documentation of 
    # openCV and knowing it's capabalities. I use color spacing method to figure out
    # the distingushing color and isolate it. This is done by first smoothing the image,
    # converting it to HSV (Hue, Saturation, Value). The HSV makes it easier for me to 
    # differentiate colors and be able to distingusih them later on when masking.
    #
    # Then create a masked image our of the HSV image using thresholds of colors that
    # correspond to the desired color I want (selected in the __init__ function).
    # This threshold masks out anything that is not in the range of the lower and upper
    # bounds of the desired color.
    #
    # By using morphological transformation, it is done by clearing noise using opening and closing methods.
    # Opening clears the noise OUTSIDE the selected color, while closing removes the noise INSIDE the color.
    # Opening is done by erosion then dilation, while closing is done by frist dilation then erosion. 
    # Erosion and dilation is basically either keeping the inside as white as possible, or the outside as black
    # as possible respectively.
    #
    # By getting the contours from the result of the masking, we can finally get the x,y coordinates
    # which we use to draw circles on those pixles. (Circles currently don't seem to be the most optimum
    # option. Due the the lower frame rate, we cannot get consistent lines if the motion tracked is fast
    # this will be fixed and further improved.

    def run(self):
        # Initialze variables
        continueCap = True              # Boolean to keep capturing if this is true
        paintNow = False                # This is used to trigger the painting command
        paintStr = "OFF"                # This will be used to show if there drawing is active or not

        while continueCap:
            # Get the capture feed into variable frame
            _, frame = self.cap.read()

            # Flip for mirror effect
            frame = cv2.flip(frame,1)

            # Smooth by applying gaussian blur for less noise and easier detection
            frame = cv2.GaussianBlur(frame,(5,5),0)

            # convert to HSV (Huw, Saturation, Value) instead of BGR to better detemine the color
            # to track
            HSV_Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # GRAY_Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # cv2.imshow("Grayscale Converted", GRAY_Frame)

            # Mask the frame using the threshold of lower and upper bounds
            maskedFrame = cv2.inRange(HSV_Frame, self.lowerColor, self.upperColor)

            # Using morphological transformation
            maskOpen = cv2.morphologyEx(maskedFrame, cv2.MORPH_OPEN, self.kernelOpen)
            maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, self.kernelClose)

            # We set the closing mask to the final to make changes
            maskFinal = maskClose
            
            # result = cv2.bitwise_and(frame, frame, mask = maskedFrame)

            # Get the contours around the final mask
            _,conts,_ = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Draw the contour on the main display to show which objects exactly 
            # are being tracked
            cv2.drawContours(frame,conts, -1, (255,0,0), 3)

            # Go through the contours
            for i in range(len(conts)):
                # get the bouding rectangle of that contour to get the x and y axis
                x,y,w,h=cv2.boundingRect(conts[i])
                # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255), 2)
                # Display the coordinates on the close mask screen
                cv2.putText(maskClose, "(" + str(x) + ", " + str(y) + ")", (50, 50), self.font, 0.90, (255,255,255), 1)
                # Draw when printNow is true (when the space key is pressed)
                if paintNow:
                    cv2.circle(self.canvas,(x,y), 10, (0,255,0), -1)

            # Display of it is ready to draw or not
            cv2.putText(maskClose, paintStr, (50, 100), self.font, 0.90, (255,255,255), 1)
                
            # Show the windows
            cv2.imshow("Original Gaussian Blurred", frame)
            cv2.imshow("HSV Converted", HSV_Frame)
            cv2.imshow("Masked Frame", maskedFrame)
            # cv2.imshow("Result", result)
            cv2.imshow("Mask Close", maskClose)
            cv2.imshow("Canvas", self.canvas)

            # If the space key is pressed, invered the painting functionality
            if cv2.waitKey(1) == 32:
                paintNow = ~paintNow
                if paintNow:
                    paintStr = "ON"
                else:
                    paintStr = "OFF"

            # Exit when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                continueCap = False

        # reslease capture and kill all windows
        self.cap.release()
        cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------------
#                                       Main
# ------------------------------------------------------------------------------------------
# Create an object for the colorTracker class
colorTrackerObject = colorTracker()
# Call the run functionality
colorTrackerObject.run()


# Below is a commented section where I plan to make into different functions/classes
# for which it will contribute to the complete project later on. For now, I focused on the
# getting the checkpoint completed.

'''
import numpy as np
import cv2
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
        # write the flipped frame
#        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# Release everything if job is finished
cap.release()
#out.release()
cv2.destroyAllWindows()
'''


'''
import cv2
import numpy as np

# create video capture
cap = cv2.VideoCapture(0)

while(1):

    # read the frames
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # smooth it
    frame = cv2.blur(frame,(3,3))

    # convert to hsv and find range of colors
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv,np.array((0, 80, 80)), np.array((20, 255, 255)))
    #thresh2 = thresh.copy()
    gray =  cv2.GaussianBlur(gray,(5,5),0);
    # find contours in the threshold image
    #thresh,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,2,20,
                            param1=100,param2=100,minRadius=60,maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            print x,y,r
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            #cv2.rectangle(thresh2, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    	    

    # Show it, if key pressed is 'Esc', exit the loop
    cv2.imshow('frame',frame)
    cv2.imshow('thresh',gray)
    if cv2.waitKey(33)== 27:
        break

# Clean up everything before leaving
cv2.destroyAllWindows()
cap.release()
'''


'''
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('sachin.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''

import cv2
import numpy as np
import copy
import math
from appscript import app

# Environment:
# OS    : Mac OS EL Capitan
# python: 3.5
# opencv: 2.4.13

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)


        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal,cnt = calculateFingers(res,drawing)
            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print cnt
                    app('System Events').keystroke(' ')  # simulate pressing blank space

        cv2.imshow('output', drawing)

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.BackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print '!!!Background Captured!!!'
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print '!!!Reset BackGround!!!'
    elif k == ord('n'):
        triggerSwitch = True
        print '!!!Trigger On!!!'
'''

'''
import cv2
import numpy as np
import time

#Open Camera object
cap = cv2.VideoCapture(0)

#Decrease frame size
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 600)

def nothing(x):
    pass

# Function to find angle between two vectors
def Angle(v1,v2):
 dot = np.dot(v1,v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle

# Function to find distance between two points in a list of lists
def FindDistance(A,B): 
 return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2)) 
 

# Creating a window for HSV track bars
cv2.namedWindow('HSV_TrackBar')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)

while(1):

    #Measure execution time 
    start_time = time.time()
    
    #Capture frames from the camera
    ret, frame = cap.read()
    
    #Blur the image
    blur = cv2.blur(frame,(3,3))
 	
 	#Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    
    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
    
    #Kernel matrices for morphological transformation    
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation2,5)
    ret,thresh = cv2.threshold(median,127,255,0)
    
    #Find contours of the filtered frame
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    
    #Draw Contours
    #cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
    #cv2.imshow('Dilation',median)
    
	#Find Max contour area (Assume that hand is in the frame)
    max_area=100
    ci=0	
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i  
            
	#Largest area contour 			  
    cnts = contours[ci]

    #Find convex hull
    hull = cv2.convexHull(cnts)
    
    #Find convex defects
    hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)
    
    #Get defect points and draw them in the original image
    FarDefect = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        FarDefect.append(far)
        cv2.line(frame,start,end,[0,255,0],1)
        cv2.circle(frame,far,10,[100,255,255],3)
    
	#Find moments of the largest contour
    moments = cv2.moments(cnts)
    
    #Central mass of first order moments
    if moments['m00']!=0:
        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
        cy = int(moments['m01']/moments['m00']) # cy = M01/M00
    centerMass=(cx,cy)    
    
    #Draw center mass
    cv2.circle(frame,centerMass,7,[100,0,255],2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Center',tuple(centerMass),font,2,(255,255,255),2)     
    
    #Distance from each finger defect(finger webbing) to the center mass
    distanceBetweenDefectsToCenter = []
    for i in range(0,len(FarDefect)):
        x =  np.array(FarDefect[i])
        centerMass = np.array(centerMass)
        distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
        distanceBetweenDefectsToCenter.append(distance)
    
    #Get an average of three shortest distances from finger webbing to center mass
    sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
    AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
 
    #Get fingertip points from contour hull
    #If points are in proximity of 80 pixels, consider as a single point in the group
    finger = []
    for i in range(0,len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
            if hull[i][0][1] <500 :
                finger.append(hull[i][0])
    
    #The fingertip points are 5 hull points with largest y coordinates  
    finger =  sorted(finger,key=lambda x: x[1])   
    fingers = finger[0:5]
    
    #Calculate distance of each finger tip to the center mass
    fingerDistance = []
    for i in range(0,len(fingers)):
        distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
        fingerDistance.append(distance)
    
    #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
    #than the distance of average finger webbing to center mass by 130 pixels
    result = 0
    for i in range(0,len(fingers)):
        if fingerDistance[i] > AverageDefectDistance+130:
            result = result +1
    
    #Print number of pointed fingers
    cv2.putText(frame,str(result),(100,100),font,2,(255,255,255),2)
    
    #show height raised fingers
    #cv2.putText(frame,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger8',tuple(finger[7]),font,2,(255,255,255),2)
        
    #Print bounding rectangle
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.drawContours(frame,[hull],-1,(255,255,255),2)
    
    ##### Show final image ########
    cv2.imshow('Dilation',frame)
    ###############################
    
    #Print execution time
    #print time.time()-start_time
    
    #close the output video by pressing 'ESC'
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
'''
'''
#! /usr/bin/env python

import cv

color_tracker_window = "Color Tracker"

class ColorTracker():

    def __init__(self):
        cv.NamedWindow( color_tracker_window, 1 )
        self.capture = cv.CaptureFromCAM(0)

    def run(self):
        while True:
            img = cv.QueryFrame( self.capture )

            #blur the source image to reduce color noise
            cv.Smooth(img, img, cv.CV_BLUR, 3);

            #convert the image to hsv(Hue, Saturation, Value) so its
            #easier to determine the color to track(hue)
            hsv_img = cv.CreateImage(cv.GetSize(img), 8, 3)
            cv.CvtColor(img, hsv_img, cv.CV_BGR2HSV)

            #limit all pixels that don't match our criteria, in this case we are
            #looking for purple but if you want you can adjust the first value in
            #both turples which is the hue range(120,140).  OpenCV uses 0-180 as
            #a hue range for the HSV color model
            thresholded_img =  cv.CreateImage(cv.GetSize(hsv_img), 8, 1)
            cv.InRangeS(hsv_img, (120, 80, 80), (140, 255, 255), thresholded_img)

            #determine the objects moments and check that the area is large
            #enough to be our object
            moments = cv.Moments(thresholded_img, 0)
            area = cv.GetCentralMoment(moments, 0, 0)

            #there can be noise in the video so ignore objects with small areas
            if(area > 100000):
                #determine the x and y coordinates of the center of the object
                #we are tracking by dividing the 1, 0 and 0, 1 moments by the area
                x = cv.GetSpatialMoment(moments, 1, 0)/area
                y = cv.GetSpatialMoment(moments, 0, 1)/area

                #print 'x\: ' + str(x) + ' y\: ' + str(y) + ' area\: ' + str(area)

                #create an overlay to mark the center of the tracked object
                overlay = cv.CreateImage(cv.GetSize(img), 8, 3)

                cv.Circle(overlay, (x, y), 2, (255, 255, 255), 20)
                cv.Add(img, overlay, img)
                #add the thresholded image back to the img so we can see what was
                #left after it was applied
                cv.Merge(thresholded_img, None, None, None, img)

            #display the image
            cv.ShowImage(color_tracker_window, img)

            if cv.WaitKey(10) == 27:
                break

if __name__=="__main__":
    color_tracker = ColorTracker()
    color_tracker.run()

'''


'''
import numpy as np
import cv2
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
        # write the flipped frame
#        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# Release everything if job is finished
cap.release()
#out.release()
cv2.destroyAllWindows()
'''