#! /usr/bin/env python

import cv2

color_tracker_window = "Color Tracker"

class ColorTracker():

    def __init__(self):
        cv2.NamedWindow( color_tracker_window, 1 )
        self.capture = cv2.CaptureFromCAM(0)

    def run(self):
        while True:
            img = cv2.QueryFrame( self.capture )

            #blur the source image to reduce color noise
            cv2.Smooth(img, img, cv2.cv2_BLUR, 3);

            #convert the image to hsv(Hue, Saturation, Value) so its
            #easier to determine the color to track(hue)
            hsv_img = cv2.CreateImage(cv2.GetSize(img), 8, 3)
            cv2.cv2tColor(img, hsv_img, cv2.cv2_BGR2HSV)

            #limit all pixels that don't match our criteria, in this case we are
            #looking for purple but if you want you can adjust the first value in
            #both turples which is the hue range(120,140).  Opencv2 uses 0-180 as
            #a hue range for the HSV color model
            thresholded_img =  cv2.CreateImage(cv2.GetSize(hsv_img), 8, 1)
            cv2.InRangeS(hsv_img, (120, 80, 80), (140, 255, 255), thresholded_img)

            #determine the objects moments and check that the area is large
            #enough to be our object
            moments = cv2.Moments(thresholded_img, 0)
            area = cv2.GetCentralMoment(moments, 0, 0)

            #there can be noise in the video so ignore objects with small areas
            if(area > 100000):
                #determine the x and y coordinates of the center of the object
                #we are tracking by dividing the 1, 0 and 0, 1 moments by the area
                x = cv2.GetSpatialMoment(moments, 1, 0)/area
                y = cv2.GetSpatialMoment(moments, 0, 1)/area

                #print 'x\: ' + str(x) + ' y\: ' + str(y) + ' area\: ' + str(area)

                #create an overlay to mark the center of the tracked object
                overlay = cv2.CreateImage(cv2.GetSize(img), 8, 3)

                cv2.Circle(overlay, (x, y), 2, (255, 255, 255), 20)
                cv2.Add(img, overlay, img)
                #add the thresholded image back to the img so we can see what was
                #left after it was applied
                cv2.Merge(thresholded_img, None, None, None, img)

            #display the image
            cv2.ShowImage(color_tracker_window, img)

            if cv2.WaitKey(10) == 27:
                break

if __name__=="__main__":
    color_tracker = ColorTracker()
    color_tracker.run()
