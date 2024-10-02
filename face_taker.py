import numpy as np
import json
import cv2
import os


def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_face_id(directory: str) -> int:
    user_ids = []
    for filename in os.listdir(directory):
        number = int(os.path.split(filename)[-1].split("-")[1])
        user_ids.append(number)
    user_ids = sorted(list(set(user_ids)))
    max_user_ids = 1 if len(user_ids) == 0 else max(user_ids) + 1
    for i in sorted(range(0, max_user_ids)):
        if i not in user_ids:
            return i
    return max_user_ids


def save_name(face_id: int, face_name: str, filename: str) -> None:
    names_json = {}
    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            try:
                names_json = json.load(fs)
            except json.JSONDecodeError:
                names_json = {}

    names_json[str(face_id)] = face_name

    with open(filename, 'w') as fs:
        json_dump = json.dumps(names_json, ensure_ascii=False, indent=4)
        fs.write(json_dump)


if __name__ == '__main__':
    directory = 'images'
    cascade_classifier_filename = 'haarcascade_frontalface_default.xml'
    names_json_filename = 'names.json'

    # Create 'images' directory if it doesn't exist
    create_directory(directory)

    # Load the pre-trained face cascade classifier
    faceCascade = cv2.CascadeClassifier(cascade_classifier_filename)

    # Check if camera can be opened with the URL
    cam = cv2.VideoCapture('http://192.168.8.130:8080/video')
    if not cam.isOpened():
        print("Error: Cannot open webcam stream.")
        exit(1)

    # Set camera dimensions
    cam.set(3, 640)
    cam.set(4, 480)

    # Initialize face capture variables
    count = 0
    face_name = input('\nEnter user name and press <return> -->  ')
    face_id = get_face_id(directory)
    save_name(face_id, face_name, names_json_filename)
    print('\n[INFO] Initializing face capture. Look at the camera and wait...')

    while True:
        # Read a frame from the camera
        ret, img = cam.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Increment the count for naming the saved images
            count += 1

            # Save the captured image into the 'images' directory
            cv2.imwrite(f'./images/Users-{face_id}-{count}.jpg', gray[y:y + h, x:x + w])

            # Display the image with rectangles around faces
            cv2.imshow('image', img)

        # Press Escape to end the program
        k = cv2.waitKey(100) & 0xff
        if k == 27:  # Escape key to exit
            break

        # Take 30 face samples and stop video. You may increase or decrease the number of
        # images. The more, the better while training the model.
        elif count >= 30:
            break

    print('\n[INFO] Success! Exiting Program.')

    # Release the camera
    cam.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
