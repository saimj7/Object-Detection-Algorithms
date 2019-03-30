import cv2

cap = cv2.VideoCapture(0)

# haar trained model on frontal faces only
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while(True):

    ret, frame = cap.read() #read frame by frame
    #A frame is a array of 3 matrices where each matrix is for the respective color blue, green, red

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert into gray

    #gray is a single matrix now

    #detectMultiScale function detect faces and return an array of position coordinates and sizes
    # 1.3 is scaled factor, if its high we may miss some pixels and thus faces
    # 5 is minNeighbors, defines how many neighbor rectangles should be identified to retain it, higher value = less false positives
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)



    #define rectangle if faces detected, passed red in BGR format
    for (x,y,w,h) in faces:

      cv2.rectangle(frame, (x, y), (x+w, y+h), (0 , 0, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):#q terminates script

        break

cap.release()

cv2.destroyAllWindows()
