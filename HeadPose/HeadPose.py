import cv2
import dlib
import numpy as np
import time
from collections import OrderedDict

PREDICTOR_PATH = "C:\\Users\\ebras\\MasterOpenCV\\face-alignment\\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])



class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

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
                    fontScale=0.1,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 1, color=(0, 255, 255))
    return im

def get_imagepoints(landmarks):
    image_points = np.array([
                            (landmark_matrix[30,0], landmark_matrix[30,1]),     # Nose tip
                            (landmark_matrix[8,0], landmark_matrix[8,1] ),     # Chin
                            (landmark_matrix[45,0], landmark_matrix[45,1]),     # Left eye left corner
                            (landmark_matrix[36,0], landmark_matrix[36,1]),     # Right eye right corne
                            (landmark_matrix[54,0], landmark_matrix[54,1]),     # Left Mouth corner
                            (landmark_matrix[48,0], landmark_matrix[48,1])      # Right mouth corner
                        ], dtype="double")
    return image_points

def get_modelpoints():
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])
    return model_points

def get_camera_matrix():
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
    return camera_matrix

cap = cv2.VideoCapture('C:\\Users\\ebras\\MasterOpenCV\\lele.mp4')
width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) 

# Not sure if heigth or weigth should be first
size = (int(height), int (width))

# Not sure what this does, and if frame is a correct parameter. This probaly have to run in the while loop
# before calculations.



while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(frame, 1)
    landmark_matrix = get_landmarks(frame)
    
    image_points = get_imagepoints(frame)
    
    model_points = get_modelpoints()
    
    camera_matrix = get_camera_matrix()
        
    
    #image_with_landmarks = annotate_landmarks(frame, landmark_matrix)
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    for p in image_points:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1) 
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
 
    cv2.line(frame, p1, p2, (255,0,0), 2)
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
