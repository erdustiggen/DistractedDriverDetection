import cv2
import dlib
import numpy as np
import time
import math
from collections import OrderedDict

PREDICTOR_PATH = "Path to predictor"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_angles(frame):
    # Getting heigth and width of video frame, putting them in array size.
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    size = (int(height), int (width))

    # Identifying landmarks on face, putting them in landmark_matrix
    rects = detector(frame, 1)
    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    landmark_matrix = np.matrix([[p.x, p.y] for p in predictor(frame, rects[0]).parts()])

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
    

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), nose, yaw

cap = cv2.VideoCapture('Path to video')


while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    imgpts, modelpts, rotate_degree, nose, jaw = get_angles(frame)

    cv2.line(frame, nose, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
    cv2.line(frame, nose, tuple(imgpts[0].ravel()), (255,0,), 3) #BLUE
    cv2.line(frame, nose, tuple(imgpts[2].ravel()), (0,0,255), 3) #RED
    
    cv2.putText(frame, rotate_degree[2], (5, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
    thickness=1, lineType=1)           


    cv2.imshow('frame',frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
