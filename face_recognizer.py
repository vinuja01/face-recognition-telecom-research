import cv2
import numpy as np
import json
import os
import requests
import time
import datetime

# Function to send POST request to server with timestamp
def send_recognition_data(name, confidence):
    url = 'http://localhost:5001/api/face-recognition/notify'  # Your server's URL
    timestamp = datetime.datetime.now().isoformat()
    data = {'user': name, 'confidence': confidence, 'timestamp': timestamp}
    print(f"Preparing to send data: {data}")  # Debugging
    try:
        response = requests.post(url, json=data)
        print("Data sent to server:", response.text)
    except requests.exceptions.RequestException as e:
        print("Failed to send data to server:", e)

if __name__ == "__main__":
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.exists('trainer.yml'):
        print("Error: 'trainer.yml' not found. Please train the model first.")
        exit(1)

    recognizer.read('trainer.yml')
    print("Model loaded successfully.")

    face_cascade_Path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(face_cascade_Path):
        print(f"Error: Haar cascade file '{face_cascade_Path}' not found.")
        exit(1)

    faceCascade = cv2.CascadeClassifier(face_cascade_Path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if not os.path.exists('names.json'):
        print("Error: 'names.json' not found.")
        exit(1)

    with open('names.json', 'r') as fs:
        try:
            names_dict = json.load(fs)
            # Convert to list where index corresponds to label ID
            names = ["Unknown"]  # Ensure names[0] is "Unknown"
            sorted_items = sorted(names_dict.items(), key=lambda item: int(item[0]))
            for id_str, name in sorted_items:
                id_int = int(id_str)
                # Fill the list up to the current id
                while len(names) <= id_int:
                    names.append("Unknown")
                names[id_int] = name
            print(f"Names List: {names}")  # Debugging
        except json.JSONDecodeError:
            print("Error: 'names.json' is not a valid JSON file.")
            exit(1)

    CONFIDENCE_THRESHOLD = 70  # Adjust as needed
    cam = cv2.VideoCapture('http://192.168.8.130:8080/video')
    if not cam.isOpened():
        print("Error: Cannot open webcam.")
        exit(1)

    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    last_response_time = 0  # Initialize last response time

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        # Get current time
        current_time = time.time()

        # Check if 5 seconds have passed since the last response
        if current_time - last_response_time >= 5:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # Enhance contrast

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH))
            )

            recognized_names = []  # To store names recognized in this interval

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                print(f"Predicted ID: {id_}, Confidence: {confidence}")

                if confidence < CONFIDENCE_THRESHOLD:
                    if id_ < len(names):
                        name = names[id_]
                    else:
                        name = "Unknown"
                    confidence_text = f"{round(confidence)}%"
                    recognized_names.append((name, round(confidence)))
                    print(f"Recognized {name} with confidence {confidence_text}")
                else:
                    name = "Unknown"
                    confidence_text = "N/A"
                    recognized_names.append((name, confidence_text))
                    print(f"Unknown face detected with confidence {confidence_text}")

                cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            # Prepare data to send
            if recognized_names:
                for name, confidence in recognized_names:
                    send_recognition_data(name, confidence)
            else:
                # If no faces detected, you can choose to send a notification or skip
                print("No faces detected in this interval.")

            # Update last response time
            last_response_time = current_time

        # Display the frame
        cv2.imshow('camera', img)
        k = cv2.waitKey(1) & 0xff
        if k == 27 or k == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
