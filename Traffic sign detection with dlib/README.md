# Traffic sign detection with dlib

Training and testing a custom object detector with dlib (which uses HOG to extract features from the images and a linear SVM to classify them).

<div align="center">
<img src=https://imgur.com/JXdZCar.gif" width=500>
<p>Output</p>
</div>

---

- First up, download the desired dataset from [**here**](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) and arrange the images; annotations into respective folders.
- We are testing a stop sign class from the Caltech 101 dataset.
- Install the dlib library with ``` pip install dlib ``` (make sure to install cmake and visual studio as well).
- To train the detector:

```
python train_detector.py --class stop_sign_images --annotations stop_sign_annotations --output output/stop_sign_detector.svm
```

- To test the trained detector:

```
python test_detector.py --detector output/stop_sign_detector.svm --testing stop_sign_testing
```


## References

- HOG (Histograms of oriented gradients) paper: https://ieeexplore.ieee.org/document/1467360
- Dlib paper: https://arxiv.org/abs/1502.00046
- Official dlib doc: http://dlib.net/compile.html
- Caltech 101 dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
- More solid theory: https://www.pyimagesearch.com/pyimagesearch-gurus/


---

saimj7/ 20-10-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
