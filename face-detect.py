import cv2 as cv

face_model = cv.CascadeClassifier('face-detect-model1.xml')
cap = cv.VideoCapture(0) # open the default camera

while True:
    ret, frame = cap.read() # read a frame from the camera
    if not ret: # if there was an error reading the frame
        break
    
    gray_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_model.detect1MultiScale(gray_scale)
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'): # exit on pressing 'q' key
        break

cap.release()
cv.destroyAllWindows()
