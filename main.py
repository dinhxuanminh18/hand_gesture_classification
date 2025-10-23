import cv2 as cv
import numpy as np
import mediapipe as mp
import csv
import copy
import itertools

from model import KeyPointClassifier
from model import AslClassifier

##########################################################
# press o to turn off current mode
# press k to turn on mode 1 (normal gesture recognize)
# press l to turn on mode 2 (asl sign for number from 0 to 9)
# press m then press num from 0 to 9 for data collection (mode 3)





csv_path = r'D:\dxm\dxm_code\dxm_hand_gest\model\asl_classification\test_data.csv'

label = ['open', 'close', 'pointer', '_']
label2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def main():
    cap = cv.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    keypoint_classifier = KeyPointClassifier()
    asl_classifier = AslClassifier()

    hands = mp_hands.Hands(
        static_image_mode = True,
        max_num_hands = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )
    mode = 0
    while True:
        ret, frame = cap.read()
        key = cv.waitKey(10)
        if not ret:
            break
        if key == 27:
            break

        mode, number = logging_mode(key, mode)
        
        
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb = cv.flip(frame_rgb, 1)

        results = hands.process(frame_rgb)
        frame = cv.flip(frame, 1)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = caculate_landmark_list(frame, hand_landmarks)
                preprocessed_landmark_list = preprocessed_landmark(landmark_list)
                logging_csv(mode, number, preprocessed_landmark_list)
            
                result = keypoint_classifier.__cal__(preprocessed_landmark_list)
                result2 = asl_classifier.__cal__(preprocessed_landmark_list)

                bound = bounding_rect(frame, hand_landmarks)
                frame = draw_rectangle(frame, bound)
                if mode == 1:
                    frame = draw_text(frame, label[result], bound)
                if result2 < 10 and mode == 2:
                    frame = draw_text(frame, label2[result2], bound)
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        if mode == 1:
            cv.putText(frame, 'Record : on' , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        if mode == 2:
            cv.putText(frame, "Number-rec : on", (5, 15), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        if mode == 3:
            cv.putText(frame, "Data Collection : on", (5, 15), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
            if (48 <= key <= 57):
                cv.putText(frame, "OK", (10, 30), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        cv.imshow('Hand Landmark Detection', frame)
    cap.release()
    cv.destroyAllWindows()

def logging_mode(key, mode):
    number = -1
    if (48 <= key <= 57):
        number = key - 48
    
    if (key == 111):
        mode = 0
    if (key == 107):
        mode = 1
    if (key == 108):
        mode = 2
    if (key == 109):
        mode = 3
    return mode, number

def logging_csv(mode, number, landmark_list, csv_path = csv_path):
    if (mode == 0 or mode == 1 or mode == 2):
        pass
    elif (mode == 3) and (number != -1):
        with open(csv_path, 'a', newline = '') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow([number, *landmark_list])
    return

def bounding_rect(image, hand_landmarks):

    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0,2), int)

    for _, landmark in enumerate(hand_landmarks.landmark):
        landmark_x = int(landmark.x * image_width)
        landmark_y = int(landmark.y * image_height)

        landmark_point = np.array([landmark_x, landmark_y])
        landmark_array = np.append(landmark_array, [landmark_point], axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return[x, y, w, h]
def draw_rectangle(image, bounding_rect):
    cv.rectangle(image, (bounding_rect[0], bounding_rect[1]), (bounding_rect[2] + bounding_rect[0], bounding_rect[3] + bounding_rect[1]), (83, 99, 93), 1)
    return image

def draw_text(image, text, bounding_rect):
    cv.rectangle(image, (bounding_rect[0], bounding_rect[1]), (bounding_rect[0] + bounding_rect[2], bounding_rect[1] - 20), (83, 99, 93), -1)
    cv.putText(image, 'Mode: ' + text, (bounding_rect[0], bounding_rect[1]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image
    
def caculate_landmark_list(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = []
    for _, landmark in enumerate(hand_landmarks.landmark):
        landmark_x = int(landmark.x * image_width)
        landmark_y = int(landmark.y * image_height)
        landmark_array.append([landmark_x, landmark_y])

    return landmark_array 


def preprocessed_landmark(landmark_list):
    temp_landmark = copy.deepcopy(landmark_list)

    x_base_coordinates, y_base_coordinates = 0, 0

    for index, landmark_point in enumerate(temp_landmark):
        if index == 0:
            x_base_coordinates = landmark_point[0]
            y_base_coordinates = landmark_point[1]

        temp_landmark[index][0] = landmark_point[0] - x_base_coordinates
        temp_landmark[index][1] = landmark_point[1] - y_base_coordinates
    temp_landmark = list(itertools.chain.from_iterable(temp_landmark))

    def normalize(n):
        return n / max_value

    max_value = max(list(map(abs, temp_landmark)))
    temp_landmark = list(map(normalize, temp_landmark))
    return temp_landmark
    
main()