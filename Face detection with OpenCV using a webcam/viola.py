import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Haar trained model on frontal faces only; leveraging OpenCV's property to access the classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while(True):
    ret, frame = cap.read() # read frame by frame
    # A frame is an array of 3 matrices where each matrix is for the respective color blue, green, red

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the image into gray scale
    # gray is a single matrix now
    # detectMultiScale function detects faces and returns an array of position coordinates and sizes
    # 1.3 is scaled factor, if its high we may miss some pixels and thus, faces
    # 5 is minNeighbors: defines how many neighbor rectangles should be identified to retain it;
    # high value == less false positives
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # draw a rectangle if the faces are detected, passed red in BGR format
    for (x,y,w,h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0 , 0, 255), 2)
    cv2.imshow('Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):#q terminates script
        break

# destroy all opened windows
cap.release()
cv2.destroyAllWindows()
