# Object detection with Non-maxima suppression (NMS)

Testing a custom object detector with NMS (used HOG to extract features from the images and a linear SVM to classify them).

### NMS is used to reduce overlapping bounding boxes to only a single bounding box, thus representing the true detection of the object. Having overlapping boxes is not exactly practical and ideal, especially if we need to count the number of objects in an image.

Without NMS         |  With NMS
:-------------------------:|:-------------------------:
![Without](mylib/utils/without.png?raw=true "Without")  |  ![With](mylib/misc/with.png?raw=true "With")

---

## Inference

- We are testing a car class from the Caltech 101 dataset.
- Setup your dataset paths and configurations in 'conf/cars.json' file.
- Download the weights file from [**here**](https://drive.google.com/file/d/1bSJo8cU_gyzttScRskb5F45aeLu4LF3_/view?usp=sharing) and place it in 'output/cars'.
- The model is also trained with 'Sceneclass13' dataset to aid in accurate class detections.
- To test the trained detector (replace --image with any image from the dataset):

```
python test_model.py --conf conf/cars.json --image datasets/caltech101/101_ObjectCategories/car_side/image_0007.jpg
```


## References

- HOG (Histograms of oriented gradients) paper: https://ieeexplore.ieee.org/document/1467360
- SVM (Support vector machines): https://scikit-learn.org/stable/modules/svm.html
- Caltech 101 dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
- NMS implementations: [**Felzenszwalb et al**](https://github.com/rbgirshick/voc-dpm/blob/master/test/nms.m); [**Malisiewicz et al. method**](https://www.computervisionblog.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html).
- More solid theory: https://www.pyimagesearch.com/pyimagesearch-gurus/


---

saimj7/ 27-10-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
