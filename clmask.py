import cv2
import numpy as np
import pandas as pd
from scipy.stats import norm
import random

frame_width = 1280
frame_height = 720

def livecap(cap,lower,upper):
#takes in a live video capture to track an orange

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (640, 360))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold of orange in HSV space
        lower=np.array(lower)
        upper=np.array(upper)
        # preparing the mask to overlay
        mask = cv2.inRange(hsv, lower, upper)
        
        result = cv2.bitwise_and(frame, frame, mask = mask)

        #de noising
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        #find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in cnts:
            # Get the area of the contour
            area = cv2.contourArea(contour)
            
            # Filter out small areas to avoid noise
            if area > 500:
                # Get the center and radius of the bounding circle for the contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Draw the circle around the orange object
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)  # Green circle

        cv2.imshow('frame', frame)
        cv2.imshow('result', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def initcap(cap):
    i=1
    while i<6:
        ret, frame = cap.read()
        if not ret:
            print('Camera failed')
            exit()

        screen=frame.copy()
        x,y =frame_width//2,frame_height//2
        cv2.circle(screen,(x,y),140,(0,255,0),2)
        cv2.putText(screen,f'Take photo of ball {6-i} times',(600,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
        screen = cv2.resize(screen, (640, 360))
        cv2.imshow('frame',screen)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite(f'testimg{i}.jpg',frame)
            i+=1
            if i==6:
                break
        

def hsv():
    with open('colour.csv','w')as infile:
        infile.write('H,S,V\n')
    for i in range(5):
        hsvimg=cv2.imread(f'testimg{i+1}.jpg')
        hsvimg=cv2.cvtColor(hsvimg,cv2.COLOR_BGR2HSV)
        lst=[]
        for i in range(150):
            x,y = random.randint(505,775),random.randint(225,495)
            while (x,y) in lst or (x>750 and (y>470 or y<250)) or (x<530 and (y>470 or y<250)):
                x,y = random.randint(500,780),random.randint(220,500)
            lst.append((x,y))
            h,s,v=hsvimg[y,x]
            with open('colour.csv','a') as infile:
                infile.write(f'{h},{s},{v}\n')

    lower,upper=[],[]
    df = pd.read_csv('colour.csv')
    
    columns = ['H', 'S', 'V']

    for col in columns:
        data = df[col].dropna()
        mean, std = norm.fit(data)
        if col == 'H':
            lower.append(int(mean)-int(std))
            upper.append(int(mean)+int(std))
        else:
            lower.append(int(mean)-int(std))
            upper.append(int(mean)+int(std)+int(std))

    return lower,upper



def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    initcap(cap)
    lower,upper = hsv()
    livecap(cap,lower,upper)
    cv2.destroyAllWindows()
    cap.release()

main()



