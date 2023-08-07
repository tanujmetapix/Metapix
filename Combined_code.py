import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import pygame
import time
from pygame import mixer
from PIL import Image

# Initialize pygame mixer
mixer.init()

# Load the music file
mixer.music.load(r"D:\sample Projects\PlayMusic\2018-11-27_-_Happy_Tree_-_FesliyanStudios.com_By_Stephen_Bennett.mp3")

# Initialize the music playback flag
play_music1 = False

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the hand gesture recognition model
model = load_model(r'D:\sample Projects\hand-gesture-recognition-code\mp_hand_gesture')

# Load class names
with open("D:\sample Projects\hand-gesture-recognition-code\gesture.names", 'r') as f:
    classNames = f.read().split('\n')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the MirNet model
mirnet_model = tf.lite.Interpreter(model_path='lite-model_mirnet-fixed_fp16_1.tflite')
mirnet_model.allocate_tensors()

while True:
    # Read a frame from the webcam
    _, frame = cap.read()

    # Hand gesture recognition with music effect

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    # Post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]
            print(className, play_music1)
            if className == "thumbs up" and not play_music1:
                play_music1 = True
                time.sleep(5)
                mixer.music.play(-1)
            elif className == "stop" and play_music1:
                mixer.music.stop()
                play_music1 = False
                time.sleep(5)

    # Show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    # Background correction

    # Preprocess the image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((400, 400), Image.LANCZOS)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Run the MirNet model on the preprocessed image
    mirnet_model.set_tensor(mirnet_model.get_input_details()[0]['index'], image)
    mirnet_model.invoke()
    output_image = mirnet_model.get_tensor(mirnet_model.get_output_details()[0]['index'])

    # Convert the output image to a format compatible with OpenCV
    output_image = np.squeeze(output_image, axis=0)
    output_image = output_image.clip(0, 1) * 255
    output_image = output_image.astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Display the corrected image
    cv2.imshow('Corrected Image', output_image)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        mixer.music.stop()
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

