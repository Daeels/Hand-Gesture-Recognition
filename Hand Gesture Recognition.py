#!/usr/bin/env python
# coding: utf-8

# # 1. Install and Import Dependencies

# In[1]:


import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time
import autopy 
import math
import _thread
import sys
import json


# In[2]:


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# # 2. Draw Hands
# <img src=https://i.imgur.com/qpRACer.png />

# In[3]:


# def click_timer(name):
#     while True :
#         if (flag == 1) :
#             autopy.mouse.click()
#             time.sleep(0.1)


# In[4]:


prev_x = 0.5
prev_y = 0.5
S = 0.2 #sensibility 
L = 9 #landmark
smoothening = 7
wCam = 1280
hCam = 720
wScr , hScr = autopy.screen.size()
print(wScr)
print(hScr)
plocX, plocY = 0, 0
clocX, clocY = 0, 0
frameR = 200
# global flag
# flag = 0
# _thread.start_new_thread( click_timer, ("thread", ) )


# In[5]:


with open(sys.argv[1]+'.json', 'r') as outfile:     
    slides = json.load(outfile)


# In[ ]:


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        
#         print(frame.shape)
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detections
        results = hands.process(image)
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Rendering results
        if results.multi_hand_landmarks:
#             cv2.rectangle(image, (frameR, frameR), ( wCam - frameR, hCam - frameR),(255, 0, 255), 2)
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )

                x1 = hand.landmark[L].x * wCam
                y1 = hand.landmark[L].y * hCam
#                      
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR , hCam -frameR ), (0, hScr))

#                     print(x1)
#                     print(x3)


                clocX = plocX + (x3 - plocX) / 7
                clocY = plocY + (y3 - plocY) / 7

                autopy.mouse.move(wScr - clocX , clocY )
                plocX, plocY = clocX, clocY

#                     print(plocX, " ", plocY)
#                     print(clocX, " ", clocY)
                H = hand.landmark[L].x - prev_x
                V = hand.landmark[L].y - prev_y
                prev_x = hand.landmark[L].x
                prev_y = hand.landmark[L].y
                if (H > S):
                    exec(slides["right"])
                if (H < -S):
                    exec(slides["left"])
                if (V > S):
                    exec(slides["down"])
                if (V < -S):
                    exec(slides["up"])
                click_dist = math.sqrt((hand.landmark[12].x - hand.landmark[9].x)**2 + (hand.landmark[12].y - hand.landmark[9].y)**2)
                if (click_dist < 0.03 ) :
                    exec(slides["click"])
                    time.sleep(0.2)
                pinch_dist = math.sqrt((hand.landmark[8].x - hand.landmark[4].x)**2 + (hand.landmark[8   ].y - hand.landmark[4].y)**2)
                if (pinch_dist < 0.03 ) :
                    exec(slides["pinch"])
                    time.sleep(0.2)
                    

        try :
            cv2.imshow('Hand Tracking', image)
        except : 
            cap.release()
            cv2.destroyAllWindows()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




