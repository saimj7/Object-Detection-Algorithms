# Object detection using template matching

Simple object detection using OpenCV's template matching function (template is the object image we would like to detect in a source image).

Template         |  Source
:-------------------------:|:-------------------------:
![Template](mylib/template.jpg?raw=true "template")  |  ![Source](mylib/detection.png?raw=true "detection")

> However, template matching only works under very specific and/or controlled conditions. It is obviously not adaptable to dramatic changes in viewpoint, scale, occlusion, etc.

---

## Inference

- To run (provide your desired image paths): 

```
python template_matching.py --source mylib/source_01.jpg --template mylib/template.jpg
```

## References

- Template matching doc: https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
- More solid theory: https://www.pyimagesearch.com/pyimagesearch-gurus/


---

saimj7/ 19-10-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
