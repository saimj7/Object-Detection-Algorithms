# Object detection with Hard-negative mining (HNM)

Testing a custom object detector with HNM (used HOG to extract features from the images and a linear SVM to classify them).

#### HNM is the process of 'mining' our negative example images for false-positive detections. The HOG feature vectors associated with these false-positive detections are then used as additional training data for our Linear SVM.

> By applying hard-negative mining, we can reduce the number of false-positives and thus increase the overall detection accuracy.

Without HNM         |  With HNM
:-------------------------:|:-------------------------:
![Without](mylib/utils/without.png?raw=true "Without")  |  ![With](mylib/utils/with.png?raw=true "With")

---

## Inference

- We are testing a car class from the Caltech 101 dataset.
- Setup your dataset paths and configurations in 'conf/cars.json' file.
- Non-maxima suppression is used to reduce the overlapping bounding boxes.
- The model is also trained with 'Sceneclass13' dataset to aid in accurate class detections.
- To test the trained detector ```(replace --image with any image from the dataset)``` :

```
python test_model.py --conf conf/cars.json --image datasets/caltech101/101_ObjectCategories/car_side/image_0016.jpg
```
- To test without HNM: ```set HNM = False``` in 'test_model.py'.


## References

- HOG (Histograms of oriented gradients) paper: https://ieeexplore.ieee.org/document/1467360
- SVM (Support vector machines): https://scikit-learn.org/stable/modules/svm.html
- Caltech 101 dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech101/


---

saimj7/ 29-10-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
