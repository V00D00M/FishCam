import numpy as np
import cv2 as cv

# TODO: Create tracker

# Initialize the camera
roi_points = []
roi_set = False

# Open the camera
cap = cv.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
#Mouse callback function
def mouse_callback(event, x, y, flags, params):
    global roi_points, roi_set
    if event == cv.EVENT_LBUTTONDOWN:
        roi_points.append = [(x, y)]
        print(f"Mouse position - Height: {y} Width: {x}")
        if len(roi_points) == 2:
            roi_set = True
        
cv.namedWindow("FishCam")
cv.setMouseCallback("FishCam", mouse_callback)

# Fish detector
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# mask = object_detector.apply(object_detector)
# _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    if roi_set:
        # Draw the ROI rectangle on the frame
        cv.rectangle(frame, (roi_points[0], roi_points[1]), (0, 255, 0), 2)
        
        # Crop the ROI from the frame
        x1, y1 = roi_points[0]
        x2, y2 = roi_points[1]
        roi = frame[y1:y2, x1:x2]
    
    # Extract the region of interest
    # TODO: Extract the region of interest (ROI) from the frame
    # roi = frame[100:300, 100:300]
    
    # Apply the object detector to the roi
    mask = object_detector.apply(frame)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calculate the area & remove small elements
        area = cv.contourArea(cnt)
        if area > 50:
            cv.drawContours(frame, [cnt], 0, (0,255,0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            # cv.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)
    

    # Display the resulting frame
    # cv.imshow('ROI', roi)
    cv.imshow('FishCam', frame)
    cv.imshow("Mask", mask)

    # Exit if the user presses 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()