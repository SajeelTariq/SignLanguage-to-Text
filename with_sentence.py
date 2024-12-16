import pickle
import cv2
import mediapipe as mp
import numpy as np


def sign_to_sentence():
    single_model_dict = pickle.load(open('./single.p', 'rb'))
    single_model = single_model_dict['model']

    multi_model_dict = pickle.load(open('./multi.p', 'rb'))
    multi_model = multi_model_dict['model']

    # Set up the video capture
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    # Define a mapping for predictions to labels
    labels_dict = {
        0: 'What', 
        1: 'How', 
        2: 'You', 
        3: 'Nice', 
        4: 'Your', 
        5: 'Name', 
        6: 'Where', 
        7: 'From', 
        8: 'Day', 
        9: 'Going', 
        10: 'Like', 
        11: 'Major', 
        12: 'No', 
        13: 'Yes', 
        14: 'Thank You', 
        15: 'I', 
        16: 'Love', 
        17: 'Year', 
        18: 'Which', 
        19: 'Class'
    }

    sentence = []
    last_character = None

    while True:
        data_aux = []
        x_all = []  # Store x coordinates for all hands
        y_all = []  # Store y coordinates for all hands

        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there's an issue with the frame capture

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results     = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Count the number of hands detected in the frame
            num_hands = len(results.multi_hand_landmarks)

            # Choose the model based on the number of hands detected
            if num_hands == 1:
                model_to_use = single_model
            elif num_hands == 2:
             model_to_use = multi_model
            else:
                model_to_use = None  # No model if the number of hands is not 1 or 2 (can be adjusted as needed)

            # Process each hand detected
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []  # Reset for each hand
                y_ = []  # Reset for each hand
                z_ = []
                data_aux = []  # Reset for each hand

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Collect x and y coordinates
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

                # Normalize landmarks for the current hand
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

             # Collect bounding box coordinates
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

            # If model_to_use is valid, make prediction for the current hand
                if model_to_use is not None:
                    prediction = model_to_use.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Draw bounding box and label for the current hand
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                    if len(sentence)==0:
                        sentence.append(predicted_character)
                        # print(sentence)    
                    elif sentence[-1]!=predicted_character:
                        sentence.append(predicted_character)
                        # print(sentence)


                    # if cv2.waitKey(30) & 0xFF == ord('n'):
                    #     if len(sentence)==0:
                    #         sentence.append(predicted_character)
                    #         print(sentence)    
                    #     elif sentence[-1]!=predicted_character:
                    #         sentence.append(predicted_character)
                    #         print(sentence)
                    #     last_character=predicted_character
                    # else:
                    #     last_character = predicted_character


        # Display the frame with predictions
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow quitting with 'q' key
            break


    # print(sentence)
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return sentence
