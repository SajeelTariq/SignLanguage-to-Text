import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

single_hand_data = []
single_hand_labels = []
multi_hand_data = []
multi_hand_labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            num_hands_detected = len(results.multi_hand_landmarks)  # Get the number of hands detected
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []  # New data array for each hand
                x_, y_, z_ = [], [], []

                # Collect x and y coordinates for normalization
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    x_.append(x)
                    y_.append(y)
                    z_.append(z)

                # Calculate bounding box dimensions
                bbox_width = max(x_) - min(x_)
                bbox_height = max(y_) - min(y_)
                bbox_depth = max(z_) - min(z_)

                # Normalize landmarks and store in data_aux
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    x_normalized = (x - min(x_)) / bbox_width
                    y_normalized = (y - min(y_)) / bbox_height
                    z_normalized = (z - min(z_)) / bbox_depth
                    data_aux.append(x_normalized)
                    data_aux.append(y_normalized)
                    data_aux.append(z_normalized)

                # Append each hand's landmarks as a separate data entry
                if num_hands_detected == 1:
                    single_hand_data.append(data_aux)
                    single_hand_labels.append(dir_)  
                elif num_hands_detected == 2:
                    multi_hand_data.append(data_aux)
                    multi_hand_labels.append(dir_) 

# Save data to separate pickle files based on number of hands detected
if single_hand_data:
    with open('single_hand.pickle', 'wb') as f:
        pickle.dump({'data': single_hand_data, 'labels': single_hand_labels}, f)

if multi_hand_data:
    with open('multi_hand.pickle', 'wb') as f:
        pickle.dump({'data': multi_hand_data, 'labels': multi_hand_labels}, f)
