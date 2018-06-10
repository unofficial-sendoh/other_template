---
layout: post
title: Image Data Analysis Using Numpy & OpenCV - Part 1
excerpt: Up and Running with basic Image processing in Python
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
![Pixel Zooming](/images/zom_pic.gif)

Every photograph, in digital form, is made up of pixels. They are the smallest unit of information that makes up a picture. Usually round or square, they are typically arranged in a 2-dimensional grid.

```python
Image("F:/rgb_explain.png")
```
![RGB Explain](/images/rgb_explain.png)

Now, if all three values are at full intensity, that means they're 255, it then shows as white and if all three colors are muted, or has the value of 0, the color shows as black. The combination of these three will, in turn, give us a specific shade of the pixel color. Since each number is an 8-bit number, the values range from 0-255.

```python
Image("F:/dec_to_bin.png")
```
![Dec to Bin](/images/dec_to_bin.png)

Combination of these three color will posses tends to the hightest value among them. Since each value can have 256 different intesity or brightness value, it makes 16.8 million total shades.

```python
with open('F:/rig_gif.gif','rb') as f:
    display(Image(data=f.read(), format='gif'))
```
![RGB_Gif](/images/rig_gif.gif)
