import cv2
import numpy as np 


class Eye():

    def __init__(self, tracker, feed):
        self.tracker = tracker
        self.cap = cv2.VideoCapture(feed)
        self.__set_frame_properties(640, 480)
        self.centroid = None
        self.normalized = None
        self.excentricity = 1.0


    def get_frame(self, side=None):
        ret, frame = self.cap.read()
        if ret:
            if side == 'l':
                frame = cv2.flip(frame, 0)
            elif side == 'r':
                frame = cv2.flip(frame, 1)
            self.__process_frame(frame, 640, 480, side)
            return frame
        return False, None


    def __set_frame_properties(self, width, height):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
    
    def __process_frame(self, frame, width, height, side=None):
        ellipse = self.tracker.find_pupil(frame)
        if ellipse is not None:
            cv2.ellipse(frame, ellipse, (0,255,0), 2)
            self.excentricity = ellipse[1][1]/ellipse[1][0]
            x = ellipse[0][0]/width
            y = ellipse[0][1]/height
            self.centroid = np.array([x,y], float)