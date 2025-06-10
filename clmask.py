import cv2
import numpy as np

#takes in a live video capture to track an orange
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

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


cv2.destroyAllWindows()
cap.release()