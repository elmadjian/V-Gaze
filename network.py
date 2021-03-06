import socket
import time
import numpy as np
import zmq
from threading import Thread


class Network():

    def __init__(self):
        self.socket = None
        self.address = None
    

    def create_connection(self, ip, port=55502):
        self.terminate = False
        if ip == "":
            context = zmq.Context()
            self.socket = context.socket(zmq.PUB)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect('tcp://*:' + port)
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.address = (ip, port)
            self.socket.settimeout(0.5)
            unity = Thread(target=self.connect_to_device, args=())
            unity.start()
        
        

    def publish_vector(self, topic, le_vec, re_vec):
        le_eye = self.__convert_to_str(le_vec)
        re_eye = self.__convert_to_str(re_vec)
        msg = topic + " " + le_eye + " " + re_eye
        self.__send_msg(msg)


    def publish_coord(self, topic, coord):
        ncoord = self.__convert_to_str(coord)
        msg = topic + " " + ncoord
        self.__send_msg(msg)


    def recv_target(self):
        try:
            msg = self.socket.recv(1024)
            txt = msg.decode()
            if ';' in txt:
                return self.__convert_to_ndarray(txt)
        except Exception as e:
            return


    def connect_to_device(self):
        msg = ""
        while msg != "okay" and not self.terminate:
            try:
                self.__send_msg("eye_tracking_connection")
                time.sleep(2)
                msg = self.socket.recv(1024).decode()
            except Exception as e:
                print("No response from client")
        if msg == "okay":
            print("Connected to device")

    
    def __send_msg(self, msg):
        if self.address is None:
            self.socket.send_string(msg)
        else:
            self.socket.sendto(msg.encode(), self.address)


    def __convert_to_str(self, eye):
        e0 = "{:.8f}".format(eye[0])
        e1 = "{:.8f}".format(eye[1])
        if len(eye) == 3:
            e2 = "{:.8f}".format(eye[2])
            return e0 + ';' + e1 + ';' + e2
        return e0 + ';' + e1


    def __convert_to_ndarray(self, msg):
        msg = msg.replace(',', '.')
        coords = msg.split(';')
        x = float(coords[0])
        y = float(coords[1])
        z = float(coords[2])
        return np.array([x,y,z])


    def close(self):
        self.terminate = True
        self.socket.close()


if __name__=='__main__':
    net = Network()
    net.create_connection("192.168.1.74")
    while True:
        target = recv_target()
        print(target)
        time.sleep(0.5)
