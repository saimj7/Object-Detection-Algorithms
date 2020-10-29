from mylib.object_detection import non_max_suppression
from mylib.object_detection import ObjectDetector
from mylib.descriptors import HOG
from mylib.utils import Conf
import numpy as np
import imutils, argparse, pickle, cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to be classified")
args = vars(ap.parse_args())

#------------------------------------------------------------------------------#
""" To test with/without HNM """
HNM = False
#------------------------------------------------------------------------------#

# load the configuration file
conf = Conf(args["conf"])

# load the classifier, then initialize the Histogram of Oriented Gradients descriptor
# and the object detector
if HNM:
	model = pickle.loads(open(conf["classifier_path"], "rb").read())
else:
	model = pickle.loads(open(conf["classifier_path0"], "rb").read())

hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
	cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"], block_norm="L1")
od = ObjectDetector(model, hog)

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(260, image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect objects in the image and apply non-maxima suppression to the bounding boxes
(boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["window_step"],
	pyramidScale=conf["pyramid_scale"], minProb=conf["min_probability"])
pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])

# loop over the allowed bounding boxes and draw them
for (startX, startY, endX, endY) in pick:
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output image
if HNM:
	cv2.imshow("With HNM", image)
else:
	cv2.imshow("Without HNM", image)
cv2.waitKey(0)
