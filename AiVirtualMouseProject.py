import cv2
import numpy as np
import HandTrackingModule2 as htm
import time
import pyautogui

##############################################
wcam,hcam = 640,480
ptime=0
screenWidth,screenHeight = pyautogui.size()
#print(screenWidth,screenHeight)
frameR = 100 #Frame Reduction
smoothening = 1.5
plocX,plocY =0,0
clocX,clocY = 0,0
##############################################
cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
detector = htm.HandDetector(max_hands=1)
while True:
    #1.Find hand landmarks
    success, img = cap.read()
    img = detector.draw_hands(img)
    landmarks_list,bbox = detector.find_landmarks(img)
    #2.Get the tip of index and middle fingers
    if len(landmarks_list)!=0:
        x1,y1 = landmarks_list[8][1:]
        x2,y2 = landmarks_list[12][1:]
        #print(x1,y1,x2,y2)
        #3.Check which fingers are up
        fingers = detector.fingers_up()
        #print(fingers)
        cv2.rectangle(img,(frameR,frameR),(wcam-frameR,hcam-frameR),(255,0,255),2)
        #4.Only Index Finger: Moving Mode
        if fingers[1]==1 and fingers[2]==0:
            #5.Convert Coordinates
            x3 = np.interp(x1,(frameR,wcam-frameR),(0,screenWidth))
            y3 = np.interp(y1,(frameR,hcam-frameR),(0,screenHeight))
            #6.Smoothen Values
            clocX = plocX + (x3-plocX)/smoothening
            clocY = plocY + (y3-plocY)/smoothening
            #7.Move Mouse
            pyautogui.moveTo(screenWidth-clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        #8.Both Index and Middle Fingers are up: Clicking Mode
        if fingers[1]==1 and fingers[2]==1:
            # 9.Find Distance between fingers
            length,img,info =detector.find_distance(8,12,img)
            print(length)
            if length<40:
                # 10.Click Mouse if distance is short
                cv2.circle(img,(info[4],info[5]),15,(0,255,0),cv2.FILLED)
                pyautogui.click()


        #11.Frame Rate
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,f'FPS:{int(fps)}',(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    #12.Display
    cv2.imshow('Image',img)
    cv2.waitKey(1)