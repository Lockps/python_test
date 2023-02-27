import cv2 as cv

face_model = cv.CascadeClassifier('face-detect-model.xml')
cap = cv.VideoCapture(0) 

while True:
    ret, frame = cap.read() 
    if not ret: 
        break
    
    gray_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray_scale)
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
