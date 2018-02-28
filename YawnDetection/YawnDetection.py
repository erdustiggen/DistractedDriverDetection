import cv2
import dlib
import numpy as np
import pyaudio
import wave
from threading import Thread
import time


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
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

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FPS, 15)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 533)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 400)
chunk = 1024
yawns = 0
yawn_status = False 

alerted = False
yawn_intervals = list()
yawns_in_interval = 0

def alarm():
    f = wave.open("/Users/vemundwoien/Documents/GENALARM.WAV")
    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),channels = f.getnchannels(), rate = f.getframerate(), output = True)
    data = f.readframes(chunk)
    while(data):
        stream.write(data)
        data = f.readframes(chunk)

start = time.time()
timeint = 5
buckets = 5
alarms = 0

while True:
    ret, frame = cap.read()
    image_landmarks, lip_distance = mouth_open(frame)
    
    prev_yawn_status = yawn_status  
    
    if lip_distance > 35:
        yawn_status = True
        '''
        if (yawns > 1 and (yawns+1)%5 == 0 and alerted == False):
            cv2.putText(frame, " Yawn Count: " + str(yawns + 1), (100,100),cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),2)
            t = Thread(target = alarm)
            t.deamon = True
            t.start()
            alerted = True
        elif(yawns > 1 and (yawns+1)%5 == 0):
            cv2.putText(frame, " Yawn Count: " + str(yawns + 1), (100,100),cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),2)
        else:
            cv2.putText(frame, " Yawn Count: " + str(yawns + 1), (50,50), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),2)
        '''
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1
        yawns_in_interval += 1
        alerted = False

    diff = time.time()-start
    if diff%timeint <= 0.22:
        print(str(diff//timeint) + ":   " + str(round(diff,2)))
        if(len(yawn_intervals)<buckets):
            if yawns_in_interval > 0:
                yawn_intervals.append(True)
            else:
                yawn_intervals.append(False)
        else: 
            if yawns_in_interval > 0:
                yawn_intervals[int((diff//timeint)%buckets)-1] = True
            else:
                yawn_intervals[int((diff//timeint)%buckets)-1] = False
        yawns_in_interval = 0
        print(yawn_intervals)
        acc = 0
        for i in yawn_intervals: 
            if i == True: 
                acc += 1
        if acc>=3: 

            if alarms < 1:
                cv2.putText(frame, " First:   ", (100,100),cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),2)
                t = Thread(target = alarm)
                t.deamon = True
                t.start()
            elif (alarms >= 1 and alarms < 3):
                cv2.putText(frame, " More serious:   ", (100,100),cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),2)
                t = Thread(target = alarm)
                t.deamon = True
                t.start()

            alerted = True
            for i in range(len(yawn_intervals)):
                yawn_intervals[i] = False
            alarms += 1

    cv2.imshow('Yawn Detection', frame)
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows() 
