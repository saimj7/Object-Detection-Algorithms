# Custom object detection with dlib

Training/Testing a custom object detector (from scratch) with dlib (uses HOG to extract features from the images and a linear SVM to classify them).

> We can manually label/annotate any object (faces in this example) with dlib's imglab tool.


Training Images       |  Test Output
:-------------------------:|:-------------------------:
![Train](mylib/utils/train.png?raw=true "Training images")  |  ![Test](mylib/utils/test.gif?raw=true "Testing output")


---

## Installation

- First up, dlib has to be installed: ```pip install dlib```. Make sure visual studio and cmake are also installed.
- Imglab should be built by downloading the latest dlib version from [**here**](http://dlib.net/).
- Extract the contents and then follow the commands:

```
$ cd dlib-version-folder/tools/imglab
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build . --config Release
```
- Set the system path environment variable (usually C:\Users\xxx\dlib-19.21\tools\imglab\build\Release).

## Labeling

- Navigate to the folder which contains the images you would like to label. In my case: ```$ cd Custom object detection```.
- Create/Initialize the annotations 'xml' file: ```$ imglab -c mylib/newfile.xml mylib/faces```.

> Training images are in 'mylib/faces' and testing images are in 'mylib/testing' folders.

- Note: I have already provided the annotated xml file for this example, make sure you rename/create as per your wish in the command.
- Start imglab: ```$ imglab face_detector/newfile.xml```. The following GUI should open up:

<div align="center">
<img src= mylib/utils/imglab.png?raw=true width=550>
</div>

- To start labeling, hold the 'shift' key and drag a bounding box around the object. 
- It is important that all the examples of objects are labeled to avoid false positives/bad accuracies. 

> If there is an ROI that you are unsure about and want to be ignored entirely during the training process, simply double-click the bounding box and press the 'i' key. This will cross out the bounding box and mark it as 'ignored'.

- After the process is done, click file > save. The annotated file should look like this:

<div align="center">
<img src= mylib/utils/annotated.png?raw=true width=450>
</div>


## Inference

- To train a new detector:

```
python train_detector.py --xml mylib/newfile.xml --detector mylib/detector.svm
```

> Training images are from the 'MIT + CMU frontal images' dataset.

- To test the trained detector:

```
python test_detector.py --detector mylib/detector.svm --testing mylib/testing
```

> Testing images are from the 'Caltech 10,000 Web Faces' dataset.

## References

- HOG (Histograms of oriented gradients) paper: https://ieeexplore.ieee.org/document/1467360
- SVM (Support vector machines): https://scikit-learn.org/stable/modules/svm.html
- Dlib paper: https://arxiv.org/abs/1502.00046
- Official dlib doc: http://dlib.net/compile.html
- Caltech web faces dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/

---

saimj7/ 30-10-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
