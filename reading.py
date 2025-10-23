import cv2 as cv
import numpy as np
import mediapipe as mp
import csv
import copy
import itertools
import os

csv_path = 'model/asl_classification/new.csv'
csv_data = 'asl_dataset'


mp_hands = mp.solutions.hands


def main():
    data_label = os.listdir(csv_data)
    count = 0
    for index in data_label:
        if index == '9' and index == '8': # because the data is too large, i handle two data labels at one time
            file_path = os.path.join(csv_data, index)
            for root, dirs, files in os.walk(file_path):
                for item in files:
                    path = os.path.join(file_path, item)
                    path = path.replace("\\", "/")
                    image = cv.imread(path)
                    if image is None:
                        print("error")
                        return
                    else:
                        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                        image_rgb = cv.flip(image_rgb, 1)
                        hands = mp_hands.Hands(
                            static_image_mode = True,
                            max_num_hands = 1,
                            min_detection_confidence = 0.5,
                            min_tracking_confidence = 0.5)

                        results = hands.process(image_rgb)
                        if(results.multi_hand_landmarks):
                            for hand_landmarks in results.multi_hand_landmarks:
                                landmark_list = caculate_landmark_list(image, hand_landmarks)
                                preprocessed_landmark_list = preprocessed_landmark(landmark_list)
                                logging_csv(str(index), preprocessed_landmark_list)




def caculate_landmark_list(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = []
    for _, landmark in enumerate(hand_landmarks.landmark):
        landmark_x = int(landmark.x * image_width)
        landmark_y = int(landmark.y * image_height)
        landmark_array.append([landmark_x, landmark_y])

    return landmark_array 

def logging_csv(text, landmark_list, csv_path = csv_path):
    with open(csv_path, 'a', newline = '') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow([text, *landmark_list])
    return

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
