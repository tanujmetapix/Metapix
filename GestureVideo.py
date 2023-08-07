# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:38:16 2023

@author: hp
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import pygame
import time
from pygame import mixer
import pygame

mixer.init()
mixer.music.load(r"D:\sample Projects\PlayMusic\2018-11-27_-_Happy_Tree_-_FesliyanStudios.com_By_Stephen_Bennett.mp3")
#mixer.music.play()
play_music1 = False
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model(r'D:\sample Projects\hand-gesture-recognition-code\mp_hand_gesture')

# Load class names
f = open("D:\sample Projects\hand-gesture-recognition-code\gesture.names", 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)
# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])
                #print(lmx,lmy)

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
            print(className,play_music1)
            if className == "thumbs up" and not play_music1:
                play_music1 = True
                #print("hello")
                time.sleep(5)
                mixer.music.play(-1)  # set the loop to -1 to play the file continuously
            elif className == "stop" and play_music1:
                mixer.music.stop()
                play_music1 = False
                time.sleep(5)
        

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        mixer.music.stop()
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()