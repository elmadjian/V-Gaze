import cv2
import sys
import numpy as np
import tracker
import view
import eye
import marker_detector
import calibrator
import depth
import time
import calib_screen
import threading
import scene
#import visualizer_3d
import videoio
import re
import network
from multiprocessing import Process, Pipe


class Controller():

    def __init__(self, argv, in3d=False):
        self.le_video = None
        self.re_video = None
        self.sc_video = None
        self.le_track = tracker.Tracker()
        self.re_track = tracker.Tracker()
        self.d_estimator  = depth.DepthEstimator()
        self.calibrations = {i:[None,None] for i in range(1,10)}
        self.calibrating  = False
        self.active  = False
        self.__setup_video_input(argv)
        self.left_e  = eye.Eye(self.le_track, self.le_video)
        self.right_e = eye.Eye(self.re_track, self.re_video)
        self.marker  = cv2.imread('marker2.png', cv2.IMREAD_GRAYSCALE)
        self.screen  = None
        self.in3d    = in3d
        self.pipe_father, self.pipe_child = Pipe()


    def __setup_video_input(self, argv):
        if "--cam" in argv or "--hol" in argv and len(argv) < 5:
            vid = videoio.VideoIO()
            self.le_video = vid.get_eye_id("left")
            self.re_video = vid.get_eye_id("right")
            if "--cam" in argv:
                self.sc_video = vid.get_rs_id()
            #print(self.le_video, self.re_video, self.sc_video)
        else:
            self.le_video = int(argv[2])
            self.re_video = int(argv[3])
            self.sc_video = int(argv[4])

    def exchange(self):
        temp = self.left_e.cap
        self.left_e.cap = self.right_e.cap
        self.right_e.cap = temp


    def calibrate(self, id):
        self.calibrating = id
        if self.in3d:
            calib = None
            if self.in3d == 'gpr':
                calib = calibrator.Calibrator(12, binocular=True, in3d=True)
            self.calibrations[id][0] = calib
            if self.in3d == 'hololens':
                return
            self.screen = Process(target=calib_screen.CalibrationScreen,
                    args=(1280,720,3,4,self.marker, self.pipe_child, 2,))
        else:
            calib_left  = calibrator.Calibrator(12, binocular=False)
            calib_right = calibrator.Calibrator(12, binocular=False) 
            self.calibrations[id][0] = calib_left
            self.calibrations[id][1] = calib_right
            self.screen = Process(target=calib_screen.CalibrationScreen,
                        args=(1920,1080,3,4,self.marker, self.pipe_child,))
        self.screen.start()
        cv2.namedWindow('calibration')


    def end_calibration(self):
        id = self.calibrating
        self.calibrations[id][0].clean_up_data(2)
        self.calibrations[id][0].estimate_gaze()
        if not self.in3d:
            self.calibrations[id][1].clean_up_data(2)
            self.calibrations[id][1].estimate_gaze()
        self.calibrating = False
        if self.in3d == 'hololens':
            return
        self.screen.join()
        cv2.destroyWindow('calibration')

    
    def use_calibration(self, id):
        if id in self.calibrations.keys():
            if self.calibrations[id][0] is not None:
                self.active = id
            else:
                print('No calibration has been found for id:', id)
        else:
            self.active = False


    def __collect_data(self, target, t_id, leye, reye):
        id = self.calibrating
        if target is not None and leye is not None and reye is not None and id:
            if self.in3d:
                #print('collecting:', target, leye, reye)
                self.calibrations[id][0].collect_data(target, leye, reye)
            else:
                self.calibrations[id][0].collect_data(target, t_id, leye)
                self.calibrations[id][1].collect_data(target, t_id, reye)


    def run(self, publish, ip=""):
        if self.in3d == 'gpr':
            self.run_3d(publish, ip)
        else:
            self.run_2d(publish, ip)


    def run_2d(self, publish, ip=""):
        scn = scene.SceneCamera(self.sc_video, 1280, 720) 
        kbd = view.View(self, self.pipe_father)
        scn.start()
        kbd.start()
        target_id = -1
        # net = network.Network()
        # if publish:
        #     net.create_connection(ip)
        while True:
            if scn.frame is not None:
                cv2.imshow('left', self.left_e.get_frame('l'))
                cv2.imshow('right', self.right_e.get_frame('r'))
                le_c = self.left_e.centroid
                re_c = self.right_e.centroid
                if self.calibrating:
                    target = scn.get_marker_position()
                    if self.pipe_father.poll():
                        target_img = self.pipe_father.recv()
                        target_id += 1
                        cv2.imshow('calibration', target_img)
                    self.__collect_data(target, target_id, le_c, re_c)
                elif self.active:
                    id = self.active
                    self.calibrations[id][0].plot_data()
                    self.calibrations[id][1].plot_data()
                    lcoord = self.calibrations[id][0].predict(le_c,w=1280,h=720)
                    rcoord = self.calibrations[id][1].predict(re_c,w=1280,h=720)
                    if lcoord is not None and rcoord is not None:
                        cv2.circle(scn.frame, lcoord, 12, (200,0,200),-1)
                        cv2.circle(scn.frame, rcoord, 12, (0,200,200),-1)
                cv2.imshow('scene', scn.frame)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                scn.quit = True
                break
        kbd.join()
        scn.join()
        #net.close()
        cv2.destroyAllWindows()



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#====================================================================
if __name__=="__main__":
    if '--h' in sys.argv or len(sys.argv) < 2:
        print("usage: <program> <input> [i-params] [options] [o-params]\n\n"
            + "INPUTS:\n"
            + "--cam: Connected video input devices\n"
            + "--vid: Load saved video files\n"
            + "--hol: Hololens as scene camera\n\n"
            + "OPTIONS:\n"
            + "--gpr: Gaussian Processes Regressor\n"
            + "--pub: Publish gaze estimation data\n\n"
            + "PARAMS (optional):\n"
            + "--cam 1 2 3: Left eye camera is loaded from /dev/video1,\n"
            + "             right from /dev/video2, and\n"
            + "             scene from /dev/video3\n"
            + "--vid f1 f2 f3: load video files for left and right\n"
            + "                eyes and scene camera\n"
            + "--hol 1 2: Left eye camera is loaded from /dev/video1,\n"
            + "           right from /dev/video2\n"
            + "--pub 1.1.1.1: send data to a specific address\n")
        sys.exit()
    elif '--gpr' in sys.argv:
        controller = Controller(sys.argv, 'gpr')
    else:
        controller = Controller(sys.argv)

    pub, ip = False, ""
    if '--pub' in sys.argv:
        pub = True
        ip  = re.findall("\d+\.\d+\.\d+\.\d+", str(sys.argv))[0]
    controller.run(pub, ip)






