---
layout: post
title: Briefly Machine Learning
excerpt: A brief introduction to Machine Learning in general
images:
  - url: /images/cover_brief_ml.jpg

---

## Introduction : A Little Bit About Pixel

Computer store images as a mosaic of tiny squrares. This is like the ancient art form of tile mosaic, or the melting bead kits kids play with today. Now, if these square tiles are too big, it's then hard to make smooth edges and curves. The more and smaller tiles we use, the smoother or as we say less pixelated, image will be. These sometimes gets referred to as resoulution of the images.

Vector graphics are somewhat different method of storing images that aims to avoid pixel related issues. But even vecotr images, in the end, are displayed as a mosaic of pixels. The word pixel means a picture element. A simple way to describe each pixel is using a combination of three colors, namely Red, Green, Blue. This is what we call an RGB image.

In an RGB image, each pixel is represented by three 8 bit numbers associated to the values for Red, Green, Blue respectively. Eventually using a magnifying glass, if we zoom a picture, we'll see the pictue is made up of tiny dots of little light or more specifically the pixels and what more interesting is to see that tose tiny dots of little light are actually multiple tiny dots of little light of different colors which are nothing but Red Green Blue channels.

Pixel together from far away, create an image and upfront they're just little lights that are ON and OFF. The combination of those create images and basically what we see on screen every single day.


```python
from IPython.display import Image
with open('F:/zom_pic.gif','rb') as img:
    display(Image(data = img.read(), format = 'png'))
```
![Alt Text](/images/zom_pic.gif)

Every photograph, in digital form, is made up of pixels. They are the smallest unit of information that makes up a picture. Usually round or square, they are typically arranged in a 2-dimensional grid.
