import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, Response, jsonify

TFLITE_PATH = "./models/model_mobilenet_v2.tflite"
IMAGE_SIZE = (160, 160)
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]

TARGET_FRAME_COUNT = 3
TARGET_CONSECUTIVE_PREDICTIONS = 4
TARGET_PREDICTION_SCORE = 0.92

app = Flask(__name__)
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
classify_lite = interpreter.get_signature_runner("serving_default")

frame_count = 0
previous_predictions = {letter: 0 for letter in CLASS_NAMES}
text_output = ""
x1, y1 = 100, 100
x2, y2 = (x1 + IMAGE_SIZE[0]), (y1 + IMAGE_SIZE[1])

def get_image_array(image_data):
    img_array = tf.keras.utils.img_to_array(image_data)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict(image_array):
    score_lite = classify_lite(input_2=image_array)["outputs"]
    predicted_char = CLASS_NAMES[np.argmax(score_lite)]
    prediction_score = np.max(score_lite)
    return predicted_char, prediction_score

def max_predicted(predictions):
    return max(predictions.items(), key=lambda k: k[1])

def generate_frames():
    global frame_count, previous_predictions, text_output
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        predicted_char = ""
        prediction_score = 0.0

        if frame_count == TARGET_FRAME_COUNT:
            frame_count = 0
            cropped = frame[y1:y2, x1:x2]
            image_data = Image.fromarray(cropped)
            image_array = get_image_array(image_data)
            predicted_char, prediction_score = predict(image_array)

            if prediction_score >= TARGET_PREDICTION_SCORE:
                previous_predictions[predicted_char] += 1

            letter, count = max_predicted(previous_predictions)
            if count >= TARGET_CONSECUTIVE_PREDICTIONS:
                previous_predictions = {letter: 0 for letter in CLASS_NAMES}

                if letter == "space":
                    text_output += " "
                elif letter == "del":
                    text_output = text_output[:-1]
                else:
                    text_output += letter

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, predicted_char.upper(), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {prediction_score:.2f}", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    global text_output
    text_output = ""  # Clear text when home loads
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/text_output')
def get_text():
    return jsonify({"text": text_output})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

