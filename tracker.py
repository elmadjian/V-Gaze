import numpy as np
import cv2
from skimage import exposure

class Tracker():

    def __init__(self, confidence=0.85, cutout=600):
        self.centroids = np.empty((0,2), float)
        self.conf      = confidence
        self.cutout    = cutout


    def find_pupil(self, frame):
        '''
        Main method to track pupil position
        IN: BGR frame from live camera or video file
        OUT: ellipse 
        '''
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pp = exposure.rescale_intensity(img, in_range=(0,150))
        blur = cv2.GaussianBlur(img, (7,7), 5)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blur)
        ret, thresh = cv2.threshold(img, minVal+25, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        morph = cv2.erode(thresh, kernel, iterations=2)
        blob = self.__get_blob(morph)
        cnt = self.__get_contours(blob)
        if cnt is not None:
            cnt_e, ellipse = self.__get_ellipse(cnt, img)
            if cnt_e is not None:
                confidence = self.__get_confidence(cnt, cnt_e)
                if confidence > self.conf and confidence <= 1.0:
                    self.update_centroids(ellipse[0])
                    return ellipse


    def __inpaint(self, img, mask):
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        count = 0
        for s in stats:
            if s[4] < 1000:
                x0, y0 = s[0]-7, s[1]-7
                x1, y1 = x0+s[2]+7, y0+s[3]+7
                neighbor = []
                tmp = np.zeros(img.shape, dtype=np.uint8)
                for i in range(y0, y1):
                    for j in range(x0, x1):
                        tmp[i][j] = 255
                        if mask[i][j] == 0:
                            neighbor.append(img[i][j])
                median = np.median(neighbor)-10
                for i in range(y0, y1):
                    for j in range(x0, x1):
                        if mask[i][j] > 0:
                            img[i][j] = median
                blur = cv2.GaussianBlur(img, (13,13), 9)
                img = np.where(tmp == 255, blur, img)
            count += 1
        return img    
                

    def __get_blob(self, bin_img):
        '''
        IN: thresholded image with pupil as foreground
        OUT: blob area containing only de pupil (hopefully)
        '''
        #TODO: return more than one blob
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
        blob = np.zeros(bin_img.shape, np.uint8)
        stats = stats[1:]
        max_area, idx = 0, 0
        if len(stats) > 0:
            for i in range(len(stats)):
                if stats[i,4] > max_area and 1500 < stats[i,4] < 10500:
                    max_area = stats[i,4]
                    idx = i + 1
            blob[labels==idx] = 255
            return blob


    def __get_contours(self, blob):
        '''
        IN: pupil blob in a binary image
        OUT: OpenCV contours of this blob 
        '''
        #TODO: return more than one external contours if there is more than 1 blob
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        closing = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
        cnt, hiq = cv2.findContours(closing, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        if len(cnt) > 0:
            return cv2.convexHull(cnt[0])

    
    def __get_ellipse(self, contour, img):
        '''
        IN: pupil contours and image frame
        OUT: fitted ellipse around pupil and its contours
        '''
        #TODO: if there are multiple contours, return multiple ellipses
        mask = np.zeros(img.shape, np.uint8)
        ellipse = None
        for c in [contour]:
            if len(c) >= 5:
                ellipse = cv2.fitEllipseDirect(c)
                break
        if ellipse is not None:
            mask = cv2.ellipse(mask, ellipse, 255, 2)
            cnt, hiq = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_NONE)
            return cnt, ellipse
        return None, None

    
    def __get_confidence(self, blob_contour, ellipse_contour):
        '''
        Measures the rate of how certain we are that
        we found the pupil in a particular frame
        IN: original blob and actual fitted ellipse contours
        OUT: confidence index (0-1 float)
        '''
        #TODO: if there are multiple ellipses, choose the one with higher conf rate
        blob_area = cv2.contourArea(blob_contour)
        #print(len(blob_contour), len(ellipse_contour))
        if len(ellipse_contour) >= 1:
            ellipse_area = cv2.contourArea(ellipse_contour[0])
            if ellipse_area > 0:
                return blob_area/ellipse_area
        return 0

    
    def update_centroids(self, centroid):
        '''
        Manages the amount of centroids that have been
        calculated from detected ellipses so far
        IN: current detected ellipse
        OUT: None
        '''
        self.centroids = np.vstack((self.centroids, centroid))
        if len(self.centroids) > self.cutout:
            percentile = self.cutout//10
            self.centroids = self.centroids[percentile:, :]