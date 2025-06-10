import cv2
import numpy as np

cap = cv2.VideoCapture(0)
frame_width = 1280
frame_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

def livecap():
#takes in a live video capture to track an orange

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold of orange in HSV space
        lower = np.array([5,130,95])

        upper = np.array([24,255,255])
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


    
    

def initcap():
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
    hsvimg=cv2.imread('testimg1.jpg')
    hsvimg=cv2.cvtColor(hsvimg,cv2.COLOR_BGR2HSV)
    x,y =frame_width//2,frame_height//2
    h,s,v=hsvimg[y,x]
    print(h,s,v)
        

def main():
    initcap()
    livecap()
    cv2.destroyAllWindows()
    cap.release()

main()





