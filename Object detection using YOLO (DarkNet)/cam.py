import cv2
from darkflow.net.build import TFNet
import numpy as np
import time #how fast is the processing

options = {
    'model': 'cfg/yolo.cfg', #model in cfg folder with yolo model
    'load': 'bin/yolo.weights', #loading weights from bin folder downloaded from darknet
    'threshold': 0.2, #confidence factor to draw bounding box, > 0.2 to show in image, low = tons of boxes which is not good
    'gpu': 1.0 #uses GPU to render images
}

tfnet = TFNet(options) #object called tfnet to run options
#displays info in command window like trained, layer description, output size

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)] # list of random colors for variations to boxes, array with 3 elements long
# for in range(10) = sicne we want few entries; we wanna go from 0 t0 255

#0 for webcam 1, diff no for multiple cams
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #full hd
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    stime = time.time() #start time
    ret, frame = capture.read() # ret is true while playing, false when ends; frame is actual frame; reading from capture device
    if ret:
        results = tfnet.return_predict(frame) # return_predict operation predicts whats in the video frame; passing the frame

        # define bounding box and display label on the frame
        #looping over all items, all results, adding box for each thing it detects; loops over all predictions
        for color, result in zip(colors, results): # zip makes a list and then each item is gonna be a tuple with one color and one result
            #create tuples with two corners x,y in it
            tl = (result['topleft']['x'], result['topleft']['y']) #take first element as top left from result
            br = (result['bottomright']['x'], result['bottomright']['y']) #bottom right
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100) # string, float with 0 decimals; format with label, conf and mul with 100 to get %

            #redefine frame with cv2 rectangle, pass frame first, then tl,br,color we want, line width
            frame = cv2.rectangle(frame, tl, br, color, 5)
            #add label by redifing frame with cv2 puttext; pass img first, text=label, font, font size, color, line width
            frame = cv2.putText( #puttext on the frame
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame) #display frame
        print('FPS {:.1f}'.format(1 / (time.time() - stime))) #.1f = one decimal place, calculation
    if cv2.waitKey(1) & 0xFF == ord('q'): #boilder plate code, if hit q key it ends
        break

capture.release()
cv2.destroyAllWindows()
