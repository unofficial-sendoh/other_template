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


## Importing Image

Now let's load an image and observe its various properties in general.

```python
if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt
    %matplotlib inline

    pic = imageio.imread('F:/demo_2.jpg')
    plt.figure(figsize = (15,15))

    plt.imshow(pic)
```
![Loaded Image](/images/demo_2.jpg)


## Observe Basic Properties of Image

```python
print('Type of the image : ' , type(pic))
print()
print('Shape of the image : {}'.format(pic.shape))
print('Image Hight {}'.format(pic.shape[0]))
print('Image Width {}'.format(pic.shape[1]))
print('Dimension of Image {}'.format(pic.ndim))
```

> Output

```
Type of the image :  <class 'imageio.core.util.Image'>

Shape of the image : (562, 960, 3)
Image Hight 562
Image Width 960
Dimension of Image 3
```

The shape of the ndarray show that it is a three layered matrix. The first two numbers here are length and width, and the third number (i.e. 3) is for three layers: Red, Green, Blue. So, if we calculate the size of a RGB image, the total size will be counted as height x width x 3

```python
print('Image size {}'.format(pic.size))
```
> Output
`Image size 1618560`

```python
print('Maximum RGB value in this image {}'.format(pic.max()))
print('Minimum RGB value in this image {}'.format(pic.min()))
```
> Output
```
Maximum RGB value in this image 255
Minimum RGB value in this image 0
```
These values are important to verify since the eight bit color intensity is, can not be outside of the 0 to 255 range.

Now, using the picture assigned variable we can also access any particular pixel value of an image and further can access each RGB channel separetly.

```python
'''
Let's pick a specific pixel located at 100 th Rows and 50 th Column. 
And view the RGB value gradually. 
'''
pic[ 100, 50 ]
```
> Output
`Image([109, 143,  46], dtype=uint8)`

In these case: R = 109 ; G = 143 ; B = 46 and we can realize that this particular pixel has a lot of GREEN in it. And now we could have also selected one of this number specifically by giving the index value of these three channel. Now we know for this

- `0` index value for **Red** channel
- `1` index value for **Green** channel
- `2` index value for **Blue** channel

But good to know that in OpenCV, Images takes as not RGB but BGR. `imageio.imread` loads image as RGB (or RGBA), but OpenCV assumes the image to be [BGR or BGRA](https://docs.opencv.org/trunk/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce) (BGR is the default OpenCV colour format).

```python
# A specific pixel located at Row : 100 ; Column : 50 
# Each channel's value of it, gradually R , G , B
print('Value of only R channel {}'.format(pic[ 100, 50, 0]))
print('Value of only G channel {}'.format(pic[ 100, 50, 1]))
print('Value of only B channel {}'.format(pic[ 100, 50, 2]))
```
> Output
```
Value of only R channel 109
Value of only G channel 143
Value of only B channel 46
```

OK, now let's take a quick view of each channels in the whole image.

```python
plt.title('R channel')
plt.ylabel('Height {}'.format(pic.shape[0]))
plt.xlabel('Width {}'.format(pic.shape[1]))

plt.imshow(pic[ : , : , 0])
plt.show()
```
![red_channel](/images/red_chn.JPG)

```python
plt.title('G channel')
plt.ylabel('Height {}'.format(pic.shape[0]))
plt.xlabel('Width {}'.format(pic.shape[1]))

plt.imshow(pic[ : , : , 1])
plt.show()
```
![red_channel](/images/green_chn.JPG)

```python
plt.title('B channel')
plt.ylabel('Height {}'.format(pic.shape[0]))
plt.xlabel('Width {}'.format(pic.shape[1]))

plt.imshow(pic[ : , : , 2])
plt.show()
```
![red_channel](/images/blue_chn.JPG)


Now, here we can also able to change the number of RGB values. As an example, let's set the Red, Green, Blue layer for following Rows values to full intensity.

- R channel: Row-  50 to 150
- G channel: Row- 200 to 300
- B channel: Row- 350 to 450

We'll load the image once, so that we can visualize each change simultaneously.

```python
pic = imageio.imread('F:/demo_2.jpg')

pic[50:150 , : , 0] = 255 # full intensity to those pixel's R channel
plt.figure( figsize = (10,10))
plt.imshow(pic)
plt.show()
```
![red_chn](/images/R_chn.JPG)

```python
pic[200:300 , : , 1] = 255 # full intensity to those pixel's G channel
plt.figure( figsize = (10,10))
plt.imshow(pic)
plt.show()
```
![red_chn](/images/red_grn_chn.JPG)

```python
pic[350:450 , : , 2] = 255 # full intensity to those pixel's B channel
plt.figure( figsize = (10,10))
plt.imshow(pic)
plt.show()
```
![red_chn](/images/three_chn.JPG)



