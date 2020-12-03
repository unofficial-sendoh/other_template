
In this article, we will try to understand the Model Sub-Classing API and Custom Training Loop from Scratch in TensorFlow 2. 
It may not be a beginner or advance introduction but aim to get rough intuition of what they are all about. 
The post is divided into three parts:

- Comparable Modelling Strategies in TensorFlow 2
- Build an Inception Network with Model Sub-Classing API
- End-to-End Training with Custom Training Loop from Scratch

So, at first, we will see how many ways to define models using TensorFlow 2 and how they differ from each other. Next, 
we will see how feasible it is to build a complex neural architecture using the model subclassing API which is introduced in TF 2. 
And then we will implement a custom training loop and train these subclassing model end-to-end from scratch. We will also use Tensorboard 
in our custom training loop to track the model performance for each batch. We will also see how to save and load the model after training. 
In the end, we will measure the model performance via the confusion matrix and 
classification report, etc.

# Comparable Modelling Strategies in TensorFlow 2

In `TF.Keras` there are basically three-way we can define a neural network, namely
- Sequential API
- Functional API
- Model Subclassing API

Among them, Sequential API is the easiest way to implement but comes with certain limitations. For example, with this API, we can’t 
create a model that shares feature information to another layer except to its subsequent layer. In addition, multiple input and output 
are not possible to implement either. In this point, Functional API does solve these issues greatly. A model like Inception or ResNet 
is feasible to implement in Functional API. But often deep learning researcher wants to have more control over every nuance of the 
network and on the training pipelines and that’s exactly what Model Subclassing API serves. Model Sub-Classing is a fully customizable 
way to implement the feed-forward mechanism for our custom-designed deep neural network in an object-oriented fashion.

Let’s create a very basic neural network using these three API. It will be the same neural architecture and will see what are the 
implementation differences. This of course will not demonstrate the full potential, especially for Functional and Model Sub-Classing API.
The architecture will be as follows:

```python
Input - > Conv - > MaxPool - > BN - > Conv -> BN - > Droput - > GAP -> Dense
```

Simple enough. As mentioned, let’s create the neural nets with Sequential, Functional, and Model Sub-Classing respectively.

## Sequential API

```python
# declare input shape 
seq_model = tf.keras.Sequential()
seq_model.add(tf.keras.Input(shape=imput_dim))

# Block 1
seq_model.add(tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu"))
seq_model.add(tf.keras.layers.MaxPooling2D(3))
seq_model.add(tf.keras.layers.BatchNormalization())

# Block 2
seq_model.add(tf.keras.layers.Conv2D(64, 3, activation="relu"))
seq_model.add(tf.keras.layers.BatchNormalization())
seq_model.add(tf.keras.layers.Dropout(0.3))

# Now that we apply global max pooling.
seq_model.add(tf.keras.layers.GlobalMaxPooling2D())

# Finally, we add a classification layer.
seq_model.add(tf.keras.layers.Dense(output_dim))
```

## Functional API

```python
# declare input shape 
input = tf.keras.Input(shape=(imput_dim))

# Block 1
x = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu")(input)
x = tf.keras.layers.MaxPooling2D(3)(x)
x = tf.keras.layers.BatchNormalization()(x)

# Block 2
x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Now that we apply global max pooling.
gap = tf.keras.layers.GlobalMaxPooling2D()(x)

# Finally, we add a classification layer.
output = tf.keras.layers.Dense(output_dim)(gap)

# bind all
func_model = tf.keras.Model(input, output)
```

## Model Sub-Classing API

```python
class ModelSubClassing(tf.keras.Model):
    def __init__(self, num_classes):
        super(ModelSubClassing, self).__init__()
        # define all layers in init
        # Layer of Block 1
        self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu")
        self.max1  = tf.keras.layers.MaxPooling2D(3)
        self.bn1   = tf.keras.layers.BatchNormalization()

        # Layer of Block 2
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu")
        self.bn2   = tf.keras.layers.BatchNormalization()
        self.drop  = tf.keras.layers.Dropout(0.3)

        # GAP, followed by Classifier
        self.gap   = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(num_classes)


    def call(self, input_tensor, training=False):
        # forward pass: block 1 
        x = self.conv1(input_tensor)
        x = self.max1(x)
        x = self.bn1(x)

        # forward pass: block 2 
        x = self.conv2(x)
        x = self.bn2(x)

        # droput followed by gap and classifier
        x = self.drop(x)
        x = self.gap(x)
        return self.dense(x)
```

In Model Sub-Classing there are two most important functions `__init__` and call. Basically, we will define all the `tf.keras layers` or 
custom implemented layers inside the `__init__` method and call those layers based on our network design inside the call method which is 
used to perform a forward propagation. It’s quite the same as the forward method that is used to build the model in `PyTorch` anyway.


Let’s run these models on the MNIST data set. We will load from `tf.keras.datasets`. However, the input image is `28` by `28` and in
grayscale shape. We will repeat the axis three times so that we can feasibly experiment with the pretrained weight later on if necessary.

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# x_train.shape, y_train.shape: (60000, 28, 28) (60000,)
# x_test.shape,  y_test.shape : (10000, 28, 28) (10000,)

# train set / data 
x_train = np.expand_dims(x_train, axis=-1)
x_train = np.repeat(x_train, 3, axis=-1)
x_train = x_train.astype('float32') / 255
# train set / target 
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)


# validation set / data 
x_test = np.expand_dims(x_test, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)
x_test = x_test.astype('float32') / 255
# validation set / target 
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


# ------------------------------------------------------------------

# compile 
print('Sequential API')
seq_model.compile(
          loss      = tf.keras.losses.CategoricalCrossentropy(),
          metrics   = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())
# fit 
seq_model.fit(x_train, y_train, batch_size=128, epochs=1)



# compile 
print('\nFunctional API')
func_model.compile(
          loss      = tf.keras.losses.CategoricalCrossentropy(),
          metrics   = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())
# fit 
func_model.fit(x_train, y_train, batch_size=128, epochs=1)



# compile 
print('\nModel Sub-Classing API')
sub_classing_model = ModelSubClassing(10)
sub_classing_model.compile(
          loss      = tf.keras.losses.CategoricalCrossentropy(),
          metrics   = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())
# fit 
sub_classing_model.fit(x_train, y_train, batch_size=128, epochs=1);
```

Output

```
Sequential API
469/469 [==============================] - 2s 3ms/step - loss: 7.5747 - categorical_accuracy: 0.2516

Functional API
469/469 [==============================] - 2s 3ms/step - loss: 8.1335 - categorical_accuracy: 0.2368

Model Sub-Classing API
469/469 [==============================] - 2s 3ms/step - loss: 5.2695 - categorical_accuracy: 0.1731
```

# Build an Inception Network with Model Sub-Classing API

The core data structures in `TF.Keras` is layers and model classes. A layer encapsulates both state (weight) and the transformation 
from inputs to outputs, i.e. the call method that is used to define the forward pass. However, these layers are also recursively 
composable. It means if we assign a `tf.keras.layers.Layer` instances as an attribute of another `tf.keras.layers.Layer`, the outer 
layer will start tracking the weights matrix of the inner layer. So, each layer will track the weights of its sublayers, both trainable 
and non-trainable. Such functionality is required when we need to build such a layer of a higher level of abstraction.

n this part, we will be building a small Inception model by subclassing the layers and model classes. Please see the diagram below. 
It’s a small **Inception** network, [src](https://arxiv.org/pdf/1611.03530.pdf). If we give a close look we’ll see that it mainly consists of three 
special modules, namely:

1. ***Conv Module***
2. ***Inception Module***
3. ***Downsample Module***

## Conv Module

From the diagram we can see, it consists of one convolutional network, one batch normalization, and one relu activation. Also, 
it produces C times feature maps with `K` x `K` filters and `S` x `S` strides. Now, it would be very inefficient if we simply go with the 
sequential modeling approach because we will be re-using this module many times in the complete network. So, define a functional 
block would be efficient and simple enough. But this time, we will prefer layer subclassing which is more pythonic and more efficient. 
To do that, we will create a class object that will inherit the `tf.keras.layers.Layer`classes.

```python
class ConvModule(tf.keras.layers.Layer):
	def __init__(self, kernel_num, kernel_size, strides, padding='same'):
		super(ConvModule, self).__init__()
        # conv layer
		self.conv = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, 
                        strides=strides, padding=padding)

        # batch norm layer
		self.bn   = tf.keras.layers.BatchNormalization()


	def call(self, input_tensor, training=False):
		x = self.conv(input_tensor)
		x = self.bn(x, training=training)
		x = tf.nn.relu(x)
		
        return x
```

Now, We can also initiate the object of this class and see the following properties.

```
cm = ConvModule(96, (3,3), (1,1))
y = cm(tf.ones(shape=(2,32,32,3))) # first call to the `cm` will create weights

print("weights:", len(cm.weights))
print("trainable weights:", len(cm.trainable_weights))


# output
weights: 6
trainable weights: 4
```

## Inception Module

Next comes the **Inception** module. According to the above graph, it consists of two **convolutional modules** and then merges together. 
Now as we know to merge, here we need to ensure that the output feature maps dimension ( height and width ) needs to be the same.

```python
class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, kernel_size1x1, kernel_size3x3):
        super(InceptionModule, self).__init__()
        
        # two conv modules: they will take same input tensor 
        self.conv1 = ConvModule(kernel_size1x1, kernel_size=(1,1), strides=(1,1))
        self.conv2 = ConvModule(kernel_size3x3, kernel_size=(3,3), strides=(1,1))
        self.cat   = tf.keras.layers.Concatenate()


    def call(self, input_tensor, training=False):
        x_1x1 = self.conv1(input_tensor)
        x_3x3 = self.conv2(input_tensor)
        x = self.cat([x_1x1, x_3x3])
        return x 
```

Here you may notice that we are now hard-coded exact `kernel size` and `strides` number for both convolutional layers according to the 
network (diagram). And also in ConvModule, we have already set padding to the ‘same’, so that the dimension of the feature maps will be 
the same for both (`self.conv1` and `self.conv2`); which is required in order to concatenate them to the end.

Again, in this module, two variable performs as the placeholder, **kernel_size1x1**, and **kernel_size3x3**. This is on the purpose of course. 
Because we will need different numbers of feature maps to the different stages of the entire model. If we look into the diagram of the model,
we will see that InceptionModule takes a different number of filters at different stages in the model.

## Downsample Module

Lastly the downsampling module. The main intuition for downsampling is that we hope to get more relevant feature information that highly 
represents the inputs to the model. As it tends to remove the unwanted feature so that model can focus on the most relevant. There are many 
ways we can reduce the dimension of the feature maps (or inputs). For example: using **strides 2** or using the conventional **pooling** operation.
There are many types of pooling operation, namely: **MaxPooling**, **AveragePooling**, **GlobalAveragePooling**.

From the diagram, we can see that the downsampling module contains one convolutional layer and one max-pooling layer which later merges together. 
Now, if we look closely at the diagram (top-right), we will see that the convolutional layer takes 3 x 3 size filter with strides **2 x 2**. And the 
pooling layer (here MaxPooling) takes pooling size **3 x 3** with strides **2 x 2**. Fair enough, however, we also ensure that the dimension coming from 
each of them should be the same in order to merge at the end. Now, if we remember when we design the **ConvModule** we purposely set the value of 
padding argument to `same`. But in this case, we need to set to `valid`.

```python
class DownsampleModule(tf.keras.layers.Layer):
	def __init__(self, kernel_size):
		super(DownsampleModule, self).__init__()

        # conv layer
		self.conv3 = ConvModule(kernel_size, kernel_size=(3,3), 
                         strides=(2,2), padding="valid") 

        # pooling layer 
		self.pool  = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), 
                         strides=(2,2))
		self.cat   = tf.keras.layers.Concatenate()


	def call(self, input_tensor, training=False):

        # forward pass 
		conv_x = self.conv3(input_tensor, training=training)
		pool_x = self.pool(input_tensor)
	

        # merged
        return self.cat([conv_x, pool_x])
```
