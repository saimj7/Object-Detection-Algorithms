# Object detection and tracking
### Using HSV color spaces

> Benefits: Simple code; Fast object tracking in real-time, easily obtaining 32+ FPS on modern hardware systems.

- To test on a video: ```python run.py --video video.mp4```.

- To test on webcam: ```python run.py```.

> Where video.mp4 should be a video file in your disk. Note that the script is intended to find balls (green and blue) based only on their colors.

- One example to define such a color space (line 12-14 in run.py) is by using the HSV chart:

<div align="center">
<img src= misc/hsv.png?raw=true width=500>
</div>

- Notice how values fall in the range [0, 360]. Simply divide by 2 to bring values into the range [0, 180] which is what OpenCV expects.

> As we know, colors can appear dramatically different depending on our lighting conditions. Instead, an approach of gathering data, performing experiments, and validating is better.

## References
- For solid theory: https://www.pyimagesearch.com/pyimagesearch-gurus/

---

saimj7/ 26-05-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
