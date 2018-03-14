import cv2
import dlib
import numpy as np
from pygame import mixer
import wave
from threading import Thread
import time
import math
from collections import OrderedDict


PREDICTOR_PATH = "C:\\Users\\ebras\\MasterOpenCV\\face-alignment\\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 0)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

#Yawn code
def yawn_logic(lip_distance):
   
    global yawn_status, yawns, alerted, yawns_in_interval, prevInt, alarms, buckets, timeint, yawn_intervals
   
    prev_yawn_status = yawn_status
    if lip_distance > 35:
        yawn_status = True
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1
        yawns_in_interval += 1
        alerted = False

    diff = time.time()-start
    if diff%timeint <= 0.2 and diff//timeint != prevInt:
        prevInt = diff//timeint
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
        #print(yawn_intervals)
        acc = 0
        for i in yawn_intervals: 
            if i == True: 
                acc += 1
            # To trigger the alarm
            if acc>=3: 
                if alarms < 1:
                    #cv2.putText(frame, " First:   ", (100,100),cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),2)
                    t = Thread(target = alarm1)
                    t.deamon = True
                    t.start()
                elif (alarms >= 1 and alarms < 3):
                    #cv2.putText(frame, " More serious:   ", (100,100),cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),2)
                    t = Thread(target = alarm2)
                    t.deamon = True
                    t.start()

                alerted = True
                for i in range(len(yawn_intervals)):
                    yawn_intervals[i] = False
                alarms += 1


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
    
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return lip_distance

yawns = 0
yawn_status = False 

yawn_intervals = list()
yawns_in_interval = 0

start = time.time()
timeint = 5
buckets = 5
alarms = 0
prevInt = 0

#Eye code

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
    
    top_left_eye_center = top_left_eye(landmarks)
    bottom_left_eye_center = bottom_left_eye(landmarks)
    
    left_distance = abs(top_left_eye_center - bottom_left_eye_center)
    return left_distance

def right_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    top_right_eye_center = top_right_eye(landmarks)
    bottom_right_eye_center = bottom_right_eye(landmarks)
    
    rigth_distance = abs(top_right_eye_center - bottom_right_eye_center)
    return rigth_distance

def eye_logic(rigth_distance, left_distance):
    global eye_status, started, start1, end
    prev_eye_status = eye_status   

    #Eye logic
    if rigth_distance <= 6 and left_distance <= 6:
        eye_status = True
    else: 
        eye_status = False

    if eye_status and not started:
        start1 = time.time()
        started = True
        
    elif not eye_status and started: # Oyne apne, men lukket sist.
        started=False
        start1 = time.time()
        
    elif eye_status and started:
        end = time.time()
                
    if end-start1 > 2:
        print('Subject is sleeping')
        t = Thread(target = alarm2)
        t.deamon = True
        t.start()
eye_status = False 
started = False
start1 = 0
end = 0

#Angle code

def lookForFaces(frame):
    rects = detector(frame, 0)
    if len(rects) == 0:
        return False
    return True

def checkForAngleAlarms(pitch, yaw, roll):
    if (yaw>35 or yaw<-35):
        return True
    return False

def get_angles(frame):
    # Getting heigth and width of video frame, putting them in array size.
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    size = (int(height), int (width))
    
    rects = detector(frame, 0)

    # Identifying landmarks on face, putting them in landmark_matrix
    landmark_matrix = get_landmarks(frame)

    # Filling an array with the important landmarks used for calculating yaw, pitch and roll. 
    image_points = np.array([
                            (landmark_matrix[30,0], landmark_matrix[30,1]),     # Nose tip
                            (landmark_matrix[8,0], landmark_matrix[8,1] ),     # Chin
                            (landmark_matrix[45,0], landmark_matrix[45,1]),     # Left eye left corner
                            (landmark_matrix[36,0], landmark_matrix[36,1]),     # Right eye right corne
                            (landmark_matrix[54,0], landmark_matrix[54,1]),     # Left Mouth corner
                            (landmark_matrix[48,0], landmark_matrix[48,1])      # Right mouth corner
                        ], dtype="double")
    nose = (landmark_matrix[30,0], landmark_matrix[30,1])    
        
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])

    # This Part might need change, better to use: focal_length = center[0] / np.tan(60/2 * np.pi / 180) ???
    # focal_length = center[0] / np.tan(60/2 * np.pi / 180) is maybe essential
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])

    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), nose, pitch, roll, yaw

unatentive = False
timer = 9999999999

#For all

def alarm1():
    mixer.init()
    mixer.music.load("C:\\Users\\ebras\\MasterOpenCV\\alarm.mp3")
    mixer.music.play()

def alarm2():
    mixer.init()
    mixer.music.load("C:\\Users\\ebras\\MasterOpenCV\\alarm.mp3")
    mixer.music.play()

def alarm3():
    mixer.init()
    mixer.music.load("C:\\Users\\ebras\\MasterOpenCV\\alarm.mp3")
    mixer.music.play()





cap = cv2.VideoCapture("C:\\Users\\ebras\\MasterOpenCV\\driver.MP4")
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 533)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 400)

alerted = False

while True:
    ret, frame = cap.read()

    lip_distance = mouth_open(frame)
    rigth_distance = right_open(frame)
    left_distance = left_open(frame)
        
    eye_logic(rigth_distance, left_distance)

    #Yawn logic
    yawn_logic(lip_distance)  

    if (lookForFaces(frame) == True):
        imgpts, modelpts, rotate_degree, nose, pitch, roll, yaw = get_angles(frame) 
        cv2.putText(frame, rotate_degree[0] + ' '+ rotate_degree[1]+ ' '+ rotate_degree[2], (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
        thickness=1, lineType=1)
    
        prev_unatentive = unatentive
    
        unatentive = checkForAngleAlarms(pitch, yaw, roll)
    
        if (prev_unatentive == False and unatentive == True):
            timer = time.time()
    
        if ((time.time() - timer) > 3):
            t = Thread(target = alarm3)
            t.deamon = True
            t.start()
    
        if (prev_unatentive == True and unatentive == False):
            timer = 9999999999


    cv2.imshow('Yawn and Eye Detection', frame)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows() 
