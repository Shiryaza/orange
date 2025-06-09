import cv2
import numpy as np

def nothing(x):
    pass

# Load the image
image = cv2.imread('IMG_1681.jpeg')  # Replace with your actual image path
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create trackbars for HSV value adjustment
cv2.namedWindow('Trackbars')
cv2.createTrackbar('LH', 'Trackbars', 5, 179, nothing)   # Lower Hue
cv2.createTrackbar('LS', 'Trackbars', 100, 255, nothing) # Lower Saturation
cv2.createTrackbar('LV', 'Trackbars', 100, 255, nothing) # Lower Value
cv2.createTrackbar('UH', 'Trackbars', 25, 179, nothing)  # Upper Hue
cv2.createTrackbar('US', 'Trackbars', 255, 255, nothing) # Upper Saturation
cv2.createTrackbar('UV', 'Trackbars', 255, 255, nothing) # Upper Value

# Resize scale (percentage of original size)
scale_percent = 50  # Change this to resize differently

# Main loop
while True:
    # Get HSV range from trackbars
    lh = cv2.getTrackbarPos('LH', 'Trackbars')
    ls = cv2.getTrackbarPos('LS', 'Trackbars')
    lv = cv2.getTrackbarPos('LV', 'Trackbars')
    uh = cv2.getTrackbarPos('UH', 'Trackbars')
    us = cv2.getTrackbarPos('US', 'Trackbars')
    uv = cv2.getTrackbarPos('UV', 'Trackbars')

    lower_orange = np.array([lh, ls, lv])
    upper_orange = np.array([uh, us, uv])

    # Create a binary mask and extract orange areas
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Resize outputs for display
    width = int(image.shape[1] * scale_percent / 230)
    height = int(image.shape[0] * scale_percent / 230)
    dim = (width, height)

    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
    resized_result = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)

    # Show the outputs
    cv2.imshow('Original Image', resized_image)
    cv2.imshow('Orange Mask', resized_mask)
    cv2.imshow('Detected Orange Areas', resized_result)

    # Break the loop when 'Esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cv2.destroyAllWindows()
