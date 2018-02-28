# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:48:40 2018

@author: Ragulan
"""

import cv2
import dlib
import numpy as np
import time
import pyaudio
import wave
from threading import Thread


PREDICTOR_PATH = "C:\Users\Ragulan\Anaconda2\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_right_eye(landmarks):
    top_right_eye_pts = []
    top_right_eye_pts.append(landmarks[37])
    """for i in range(50,53):
        top_right_eye_pts.append(landmarks[i])
    for i in range(61,64):
        top_right_eye_pts.append(landmarks[i])"""
    top_right_eye_all_pts = np.squeeze(np.asarray(top_right_eye_pts))
    top_right_eye_mean = np.mean(top_right_eye_pts, axis=0)
    return int(top_right_eye_mean[:,1])

def bottom_right_eye(landmarks):
    bottom_right_eye_pts = []
    bottom_right_eye_pts.append(landmarks[41])
    """for i in range(65,68):
        bottom_right_eye_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_right_eye_pts.append(landmarks[i])"""
    bottom_right_eye_all_pts = np.squeeze(np.asarray(bottom_right_eye_pts))
    bottom_right_eye_mean = np.mean(bottom_right_eye_pts, axis=0)
    return int(bottom_right_eye_mean[:,1])



def top_left_eye(landmarks):
    top_left_eye_pts = []
    top_left_eye_pts.append(landmarks[43])
    """for i in range(50,53):
        top_right_eye_pts.append(landmarks[i])
    for i in range(61,64):
        top_right_eye_pts.append(landmarks[i])"""
    top_left_eye_all_pts = np.squeeze(np.asarray(top_left_eye_pts))
    top_left_eye_mean = np.mean(top_left_eye_pts, axis=0)
    return int(top_left_eye_mean[:,1])


def bottom_left_eye(landmarks):
    bottom_left_eye_pts = []
    bottom_left_eye_pts.append(landmarks[47])
    """for i in range(65,68):
        bottom_right_eye_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_right_eye_pts.append(landmarks[i])"""
    bottom_left_eye_all_pts = np.squeeze(np.asarray(bottom_left_eye_pts))
    bottom_left_eye_mean = np.mean(bottom_left_eye_pts, axis=0)
    return int(bottom_left_eye_mean[:,1])

def left_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    
    top_left_eye_center = top_left_eye(landmarks)
    bottom_left_eye_center = bottom_left_eye(landmarks)
    
    left_distance = abs(top_left_eye_center - bottom_left_eye_center)
    return image_with_landmarks, left_distance

def right_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    
    top_right_eye_center = top_right_eye(landmarks)
    bottom_right_eye_center = bottom_right_eye(landmarks)
    
    rigth_distance = abs(top_right_eye_center - bottom_right_eye_center)
    return image_with_landmarks, rigth_distance

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)#"C:/Users/Ragulan/MasterOpenCV/small.mp4")
#cap.set(cv2.cv.CV_CAP_PROP_FPS, 15)
#cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 400)
#cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 225)
#frame_width, 533
#frame_height, 400
yawns = 0
eye_status = False 
started = False
start = 0
end = 0
chunk = 1024

def alarm():
    f = wave.open("C:/Users/Ragulan/MasterOpenCV/test1.WAV")
    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),channels = f.getnchannels(), rate = f.getframerate(), output = True)
    data = f.readframes(chunk)
    while(data):
        stream.write(data)
        data = f.readframes(chunk)

while True:
    ret, frame = cap.read()   
    image_landmarks, rigth_distance = right_open(frame)
    same_img, left_distance = left_open(frame)
    
    prev_eye_status = eye_status  
    
    if rigth_distance <= 7 and left_distance <= 7:
        eye_status = True 
        
        cv2.putText(frame, "Subject is closing eyes", (50,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

        #output_text = " Yawn Count: " + str(yawns + 1)

        #cv2.putText(frame, output_text, (50,50),
                    #cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        
        
    else:
        eye_status = False 
        #print("new frame")
        #print(rigth_distance)
        #print(left_distance)
        
    if eye_status and not started:
        start = time.time()
        started = True
        
    elif not eye_status and started: # Øyne åpne, men lukket sist.
        started=False
        start = time.time()
        
    elif eye_status and started:
        end = time.time()
        
        
    if end-start > 2:
        print('Subject is sleeping')
        t = Thread(target = alarm)
        t.deamon = True
        t.start()
        #cv2.putText(frame, "Subject is sleeeeeeping", (400,50), 
                    #cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        
        
        
    
    
    
    
         
    if prev_eye_status == True and eye_status == False:
        yawns += 1

    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Yawn Detection', frame )
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows() 