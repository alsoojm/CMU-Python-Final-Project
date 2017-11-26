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
# import imutils
import cv2
import numpy as np
import Tkinter
import tkFileDialog
import os

def doNothing(color):
    return None

# -------------------------------------------------------------------------------------
#                                   Color Tracker Class
# -------------------------------------------------------------------------------------
class colorTracker():
    # Initializers
    def __init__(self):
        # Capture video feed from the primary camera
        # CHANGE THIS VALUE TO SELECT WHICH CAMERA TO CAPTURE FROM
        self.cap = cv2.VideoCapture(0)

        # Marker Settings
        self.markerBlue = 0
        self.markerGreen = 225
        self.markerRed = 0
        self.markerColor = (self.markerBlue, self.markerGreen, self.markerRed)
        self.markerThickness = 10

        # Trackbar for choosing marker color
        cv2.namedWindow('Marker Properties')
        cv2.createTrackbar('Blue', 'Marker Properties', self.markerBlue, 225, doNothing)
        cv2.createTrackbar('Green', 'Marker Properties', self.markerGreen, 225, doNothing)
        cv2.createTrackbar('Red', 'Marker Properties', self.markerRed, 225, doNothing)
        cv2.createTrackbar('Thickness', 'Marker Properties', self.markerThickness, 100, doNothing)

        # Rectangle Settings
        self.rectangleBlue = 0
        self.rectangleGreen = 0
        self.rectangleRed = 255
        self.rectangleColor = (self.rectangleBlue, self.rectangleGreen, self.rectangleRed)
        self.rectangleThickness = 3
        self.rectangleHeight = 100
        self.rectangleWidth = 100

        # Trackbar for choosing rectangle color
        cv2.namedWindow('Rectangle Properties')
        cv2.createTrackbar('Height', 'Rectangle Properties', self.rectangleHeight, 480, doNothing)
        cv2.createTrackbar('Width', 'Rectangle Properties', self.rectangleWidth, 640, doNothing)
        cv2.createTrackbar('Blue', 'Rectangle Properties', self.rectangleBlue, 255, doNothing)
        cv2.createTrackbar('Green', 'Rectangle Properties', self.rectangleGreen, 255, doNothing)
        cv2.createTrackbar('Red', 'Rectangle Properties', self.rectangleRed, 255, doNothing)
        cv2.createTrackbar('Thickness', 'Rectangle Properties', self.rectangleThickness, 100, doNothing)

        # Text Settings
        self.textSize = 1
        self.textBlue = 255
        self.textGreen = 255
        self.textRed = 255
        self.textColor = (self.textBlue, self.textGreen, self.textRed)
        self.textThickness = 2

        # Trackbar for Text
        cv2.namedWindow('Text Properties')
        cv2.createTrackbar('Size', 'Text Properties', self.textSize, 100, doNothing)
        cv2.createTrackbar('Blue', 'Text Properties', self.textBlue, 255, doNothing)
        cv2.createTrackbar('Green', 'Text Properties', self.textGreen, 255, doNothing)
        cv2.createTrackbar('Red', 'Text Properties', self.textRed, 255, doNothing)
        cv2.createTrackbar('Thickness', 'Text Properties', self.textThickness, 25, doNothing)

        # Circle Settings
        self.circleBlue = 225
        self.circleGreen = 0
        self.circleRed = 0
        self.circleRadius = 50
        self.circleThickness = 2
        self.circleFill = 0
        self.circleColor = (self.circleBlue, self.circleGreen, self.circleRed)

        # Trackbar for choosing circle colors
        cv2.namedWindow('Circle Properties')
        cv2.createTrackbar('Blue', 'Circle Properties', self.circleBlue, 255, doNothing)
        cv2.createTrackbar('Green', 'Circle Properties', self.circleGreen, 255, doNothing)
        cv2.createTrackbar('Red', 'Circle Properties', self.circleRed, 255, doNothing)
        cv2.createTrackbar('Radius', 'Circle Properties', self.circleRadius, 500, doNothing)
        cv2.createTrackbar('Thickness', 'Circle Properties', self.circleThickness, 25, doNothing)
        cv2.createTrackbar('Fill', 'Circle Properties', self.circleFill, 1, doNothing)
        

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
        self.imgCanvas = np.zeros((480,640,3), np.uint8)

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
        READY = False
        continueCap = True              # Boolean to keep capturing if this is true
        paintNow = False                # This is used to trigger the painting command
        paintStr = "OFF"                # This will be used to show if there drawing is active or not
        programStatus = "OFF"
        file = ""
        drawingRect = False
        textReady = False
        circleReady = False


        while continueCap:
            # Get colors from trackbar
            self.markerBlue = cv2.getTrackbarPos("Blue", "Marker Properties")
            self.markerGreen = cv2.getTrackbarPos("Green", "Marker Properties")
            self.markerRed = cv2.getTrackbarPos("Red", "Marker Properties")
            self.markerColor = (self.markerBlue, self.markerGreen, self.markerRed)
            self.markerThickness = cv2.getTrackbarPos("Thickness", "Marker Properties")

            # Get properties of rectangle
            self.rectangleHeight = cv2.getTrackbarPos("Height", "Rectangle Properties")
            self.rectangleWidth = cv2.getTrackbarPos("Width", "Rectangle Properties")
            self.rectangleBlue = cv2.getTrackbarPos("Blue", "Rectangle Properties")
            self.rectangleGreen = cv2.getTrackbarPos("Green", "Rectangle Properties")
            self.rectangleRed = cv2.getTrackbarPos("Red", "Rectangle Properties")
            self.rectangleBlue = cv2.getTrackbarPos("Blue", "Rectangle Properties")
            self.rectangleThickness = cv2.getTrackbarPos("Thickness", "Rectangle Properties")
            self.rectangleColor = (self.rectangleBlue, self.rectangleGreen, self.rectangleRed)

            # Get text properties
            self.textSize = cv2.getTrackbarPos("Size", "Text Properties")
            self.textBlue = cv2.getTrackbarPos("Blue", "Text Properties")
            self.textGreen = cv2.getTrackbarPos("Green", "Text Properties")
            self.textRed = cv2.getTrackbarPos("Red", "Text Properties")
            self.textColor = (self.textBlue, self.textGreen, self.textRed)
            self.textThickness = cv2.getTrackbarPos("Thickness", "Text Properties")

            # Get circle properties
            self.circleBlue = cv2.getTrackbarPos("Blue", "Circle Properties")
            self.circleGreen = cv2.getTrackbarPos("Green", "Circle Properties")
            self.circleRed = cv2.getTrackbarPos("Red", "Circle Properties")
            self.circleRadius = cv2.getTrackbarPos("Radius", "Circle Properties")
            self.circleFill = cv2.getTrackbarPos("Fill", "Circle Properties")
            self.circleThickness = cv2.getTrackbarPos("Thickness", "Circle Properties")
            self.circleColor = (self.circleBlue, self.circleGreen, self.circleRed)

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

            if READY:
                for i in conts:
                    M = cv2.moments(i)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Draw the contour on the main display to show which objects exactly 
                    # are being tracked
                    cv2.drawContours(frame,conts, -1, (255,0,0), 3)
                    # cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                    # cv2.putText(frame, "center", (cX - 20, cY - 20), self.font, 0.5, (255, 255, 255), 2)

                # Go through the contours
                for i in range(len(conts)):
                    # get the bouding rectangle of that contour to get the x and y axis
                    x,y,w,h=cv2.boundingRect(conts[i])
                    cv2.circle(frame, (x,y), self.markerThickness, self.markerColor, -1)
                    xStart = x - 50
                    yStart = y - 50
                    # Show the objects requested on the screen
                    if drawingRect:
                        cv2.rectangle(frame,(xStart,yStart),(xStart + self.rectangleWidth, yStart + self.rectangleHeight),self.rectangleColor, self.rectangleThickness)
                    if textReady:
                        cv2.putText(frame, textFromUser, (xStart, yStart), self.font, self.textSize, self.textColor, self.textThickness)
                    if circleReady:
                        if self.circleFill == 1:
                            cv2.circle(frame, (x,y), self.circleRadius, self.circleColor, -1)
                        else:
                            cv2.circle(frame, (x,y), self.circleRadius, self.circleColor, self.circleThickness)

                    # Display the coordinates on the close mask screen
                    cv2.putText(maskClose, "(" + str(x) + ", " + str(y) + ")", (50, 50), self.font, 0.90, (255,255,255), 1)
                    # Draw when printNow is true (when the space key is pressed)
            
            # Display of it is ready to draw or not
            cv2.putText(maskClose, "Program: " + programStatus, (50, 100), self.font, 0.90, (255,255,255), 1)
            cv2.putText(maskClose, "Draw: " + paintStr, (50, 150), self.font, 0.90, (255,255,255), 1)
            if READY:
                cv2.putText(maskClose, "Press 'Space' to toggle drawing ON or OFF", (50, 200), self.font, 0.80, (255,255,255), 1)
                cv2.putText(maskClose, "Press 'r' to insert a rectangle. 'r' to apply", (50, 250), self.font, 0.80, (255,255,255), 1)
                cv2.putText(maskClose, "Press 'o' to insert a cirlce, 'o' to apply", (50, 300), self.font, 0.80, (255,255,255), 1)
                cv2.putText(maskClose, "Press 't' to insert text, 't' to apply", (50, 350), self.font, 0.80, (255,255,255), 1)
                cv2.putText(maskClose, "Press 'q' to quit", (50, 400), self.font, 0.80, (255,255,255), 1)

            # Marker Draw
            if ~paintNow:
                prev_x = 0
                prev_y = 0

            if paintNow:
                if prev_x != 0:
                    cv2.line(self.canvas, (prev_x, prev_y), (x,y), self.markerColor, 20)
                prev_x = x
                prev_y = y
                cv2.circle(self.canvas,(x,y), self.markerThickness, self.markerColor, -1)


            # Combine Canvas and Camera Window
            added = cv2.add(frame, self.canvas)

            # Show the windows
            if READY:
                cv2.imshow("Camera Window", added)
                #cv2.imshow("Original Gaussian Blurred", frame)
                cv2.imshow("HSV Converted", HSV_Frame)
                #cv2.imshow("Masked Frame", maskedFrame)
                # cv2.imshow("Result", result)
                cv2.imshow("Canvas", self.canvas)
                cv2.imshow("Information Window", maskClose)
            else:
                cv2.imshow("Origianl Camera", frame)
                cv2.imshow("Information Window", maskClose)
            

            # If the space key is pressed, invered the painting functionality
            key = cv2.waitKey(10)
            if key == 13:
                READY = True
                programStatus = "ON"
            if key == 32:
                paintNow = ~paintNow
                if paintNow:
                    paintStr = "ON"
                else:
                    paintStr = "OFF"

            # c clears the window from drawings
            elif key == 99:
                self.canvas = np.zeros((480,640,3), np.uint8)

            # i inserts an image
            elif key == 105:
                imageRead = True
                root = Tkinter.Tk()
                root.withdraw()
                currdir = os.getcwd()
                file = tkFileDialog.askopenfilename(parent=root, initialdir=currdir, title='Please select a file')
                img = cv2.imread(file)
                #height, width = added.shape[:2]
                #res = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)

            # r to insert a rectangle
            elif key == 114:
                drawingRect = ~drawingRect
                if drawingRect == False:
                    cv2.rectangle(self.canvas,(xStart,yStart),(xStart + self.rectangleWidth, yStart + self.rectangleHeight),self.rectangleColor, self.rectangleThickness)
                
            # t to insert text
            elif key == 116:
                textReady = ~textReady
                if textReady == False:
                    cv2.putText(self.canvas, textFromUser, (xStart, yStart), self.font, self.textSize, self.textColor, self.textThickness)
                else:
                    textFromUser = raw_input("Enter the text you would like to add: ")

            # o to insert circle
            elif key == 111:
                circleReady = ~circleReady
                if circleReady == False:
                    if self.circleFill == 1:
                        cv2.circle(self.canvas, (x,y), self.circleRadius, self.circleColor, -1)
                    else:
                        cv2.circle(self.canvas, (x,y), self.circleRadius, self.circleColor, self.circleThickness)
            
            # if l is pressed 
            elif key == 108:
                savedCanvas = cv2.copyMakeBorder(clearCanvas,0,0,0,0,cv2.BORDER_REPLICATE)
                newrect = cv2.rectangle(savedCanvas, (rect_pos_x_start+1, rect_pos_y_start), (rect_pos_x_end+1, rect_pos_y_end), (255, 0, 0), 3)
                rect_pos_x_start += 1
                rect_pos_x_end += 1
                self.canvas = savedCanvas

            # s to save
            elif key == 115:
                cv2.imwrite("ImageWithCam.png", added)
                cv2.imwrite("Canvas.png", self.canvas)
                
            # Exit when 'q' key is pressed
            elif key & 0xFF == ord('q'):
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