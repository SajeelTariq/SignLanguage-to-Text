from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import cv2
import pickle
import mediapipe as mp
import numpy as np
from text_to_handsign import extract_keywords, resize_and_pad
from with_sentence import sign_to_sentence  # Import sign-to-text function
from text_correction_AI import sentence_correction_AI  # Import sentence correction function
import cv2

app = Flask(__name__)
IMAGE_FOLDER = './static'

# Load models
single_model_dict = pickle.load(open('./single.p', 'rb'))
single_model = single_model_dict['model']

multi_model_dict = pickle.load(open('./multi.p', 'rb'))
multi_model = multi_model_dict['model']

# Labels mapping
labels_dict = {
    0: 'What', 1: 'How', 2: 'You', 3: 'Nice', 4: 'Your', 5: 'Name', 6: 'Where', 
    7: 'From', 8: 'Day', 9: 'Going', 10: 'Like', 11: 'Major', 12: 'No', 13: 'Yes', 
    14: 'Thank You', 15: 'I', 16: 'Love', 17: 'Year', 18: 'Which', 19: 'Class'
}

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Global variable to store the detected sentence
detected_sentence = []
current_prediction = None
last_prediction = None
stop_camera = False

@app.route('/get_current_prediction', methods=['GET'])
def get_current_prediction():
    global last_prediction
    return jsonify({'current_prediction': last_prediction or ""})

@app.route('/append_word', methods=['POST'])
def append_word():
    global detected_sentence, last_prediction
    if last_prediction:  # Only append if there is a valid prediction
        if not detected_sentence or detected_sentence[-1] != last_prediction:
            detected_sentence.append(last_prediction)
    return jsonify({'sentence': ' '.join(detected_sentence)})


# Route: Process Sign-to-Text
@app.route('/process-sign-to-text', methods=['POST'])
def process_sign_to_text():
    detected_sentence = sign_to_sentence()
    corrected_sentence = sentence_correction_AI(detected_sentence)
    return jsonify({'detected': detected_sentence, 'corrected': corrected_sentence})

# Stream the live camera feed and perform hand detection
def generate_camera_feed():
    global detected_sentence, stop_camera, current_prediction, last_prediction
    detected_sentence = []  # Reset sentence
    cap = cv2.VideoCapture(0)

    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_prediction = None

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            model_to_use = single_model if num_hands == 1 else multi_model if num_hands == 2 else None

            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []  # Reset for each hand
                y_ = []  # Reset for each hand
                z_ = []
                data_aux = []  # Reset for each hand

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
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

                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

                if model_to_use:
                    prediction = model_to_use.predict([np.asarray(data_aux)])
                    current_prediction = labels_dict[int(prediction[0])]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, current_prediction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

                    if current_prediction != None:
                        last_prediction = current_prediction


        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Send the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Route: Stream Camera Feed
@app.route('/camera_feed')
def camera_feed():
    return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route: Stop Camera Feed
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global stop_camera
    stop_camera = True
    return jsonify({'status': 'Camera stopped'})

# Route: Get Detected Sentence
@app.route('/get_sentence', methods=['POST'])
def get_sentence():
    global detected_sentence
    corrected_sentence = sentence_correction_AI(detected_sentence)
    return jsonify({'detected': ' '.join(detected_sentence), 'corrected': corrected_sentence})

@app.route('/convert', methods=['POST'])
def convert_text():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Extract keywords
    keywords = extract_keywords(text)
    print("Extracted Keywords:", keywords)

    # Check if images exist for each keyword
    images = []
    for keyword in keywords:
        image_path = os.path.join(IMAGE_FOLDER, f"{keyword}.jpg")
        if os.path.exists(image_path):
            images.append(f"/static/{keyword}.jpg")
        else:
            print(f"No image found for: {keyword}")
    
    return jsonify({'keywords': keywords, 'images': images})

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # This assumes you have an 'index.html' in a 'templates' folder

# Route: Sign-to-Text Page
@app.route('/sign-to-text')
def sign_to_text():
    global stop_camera
    stop_camera = False  # Reset the stop flag when the page reloads
    return render_template('sign-to-text.html')

@app.route('/text-to-sign')
def text_to_sign():
    return render_template('text-to-sign.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
