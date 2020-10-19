import cv2, time
from darkflow.net.build import TFNet
import numpy as np

options = {
    'model': 'cfg/yolo.cfg', # cfg folder with the yolo model
    'load': 'bin/yolov2.weights', # loading weights from bin folder downloaded from darknet
    'threshold': 0.2, # confidence factor to draw bounding box; > 0.2 to show in image, low == tons of boxes which is not good
    'gpu': 0.5 # uses GPU to render the video; 1.0 == 100% memory, so keep that in mind
}

tfnet = TFNet(options) # object called tfnet to run options
# displays the info in command window like trained, layer description, output size

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)] # list of random colors for variations to boxes, array with 3 elements long
# for in range(10) == since we want few entries; we go from 0 t0 255

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    stime = time.time() # start time
    ret, frame = capture.read() # ret is true while playing, false when ends; frame is actual frame; reading from capture device
    if ret:
        results = tfnet.return_predict(frame) # return_predict operation predicts what is in the video frame; passing the frame

        # define the bounding box and display label on the frame
        # looping over all items, all results, adding bbox for everything we detect; loops over all predictions
        for color, result in zip(colors, results): # zip makes a list and then each item is a tuple with one color and one result
            # create tuples with two corners x,y in it
            tl = (result['topleft']['x'], result['topleft']['y']) # take first element as top left from result
            br = (result['bottomright']['x'], result['bottomright']['y']) # bottom right
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100) # string, float with 0 decimals; format with label, conf and mul with 100 to get %

            # redefine frame with cv2 rectangle, pass frame first; then tl, br, color we want, line width etc
            frame = cv2.rectangle(frame, tl, br, color, 5)
            # add label by redefining frame with cv2 puttext; pass img first, text=label, font, font size, color, line width
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS: {:.1f}'.format(1 / (time.time() - stime))) #.1f = one decimal place, calculation
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# destroy/close all opened windows
capture.release()
cv2.destroyAllWindows()
