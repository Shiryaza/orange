import cv2
import numpy as np
import pandas as pd
from scipy.stats import norm
import random
from collections import deque
from filterpy.kalman import KalmanFilter
import copy

frame_width = 1280
frame_height = 720

def smooth_path(path, window_size=3):
    smoothed = []
    for i in range(len(path)):
        window = path[max(i - window_size + 1, 0):i + 1]
        valid_points = [p for p in window if p is not None]
        if valid_points:
            x = int(np.mean([p[0] for p in valid_points]))
            y = int(np.mean([p[1] for p in valid_points]))
            smoothed.append((x, y))
        else:
            smoothed.append(None)
    return smoothed

def livecap(cap,lower,upper):
#takes in a live video capture to track an orange
    # --- Kalman Filter Setup ---
    kf = KalmanFilter(dim_x=4, dim_z=2)  # [x, y, dx, dy]
    kf.x = np.array([0, 0, 0, 0])        # Initial state
    kf.F = np.array([[1, 0, 1, 0],       # State transition matrix
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],       # Measurement function
                    [0, 1, 0, 0]])
    kf.P *= 1000.                        # Initial covariance
    kf.R = np.array([[5, 0], [0, 5]])    # Measurement noise
    kf.Q = np.eye(4) * 0.01              # Process noise

    path = deque(maxlen=32)

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
        measurement=None
        for contour in cnts:
            # Get the area of the contour
            area = cv2.contourArea(contour)
            
            # Filter out small areas to avoid noise
            if area > 500:
                # Get the center and radius of the bounding circle for the contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                measurement = np.array([x, y])
                # Draw the circle around the orange object
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)  # Green circle
        
        if measurement is not None:
            kf.update(measurement)

        kf.predict()

        predicted = kf.x
        px, py = int(predicted[0]), int(predicted[1])

        cv2.circle(frame, (px, py), 8, (0, 80, 255), -1)

        # Predict the next 1 second of motion (30 frames ahead)
        future_kf = copy.deepcopy(kf)
        future_positions = []

        for _ in range(30):  # Simulate 1 second at 30 FPS
            future_kf.predict()
            fx, fy = int(future_kf.x[0]), int(future_kf.x[1])
            future_positions.append((fx, fy))

        # Draw the future path
        for point in future_positions:
            cv2.circle(frame, point, 8, (0, 80, 255), -1)

        path.appendleft((int(x),int(y))) #add center to path

        newpath = smooth_path(list(path))
        
        #draw path
        for i in range(1, len(newpath)):
            if newpath[i - 1] is None or newpath[i] is None or not cnts:
                continue
            thickness = int(np.sqrt(len(newpath) / float(i + 1)) * 2.5)
            cv2.line(frame, newpath[i - 1], newpath[i], (255, 70, 0), thickness)

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
        cv2.putText(screen,f"'P' to take photo of ball {6-i} times",(570,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
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