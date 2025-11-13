# Deep-Learning-Exp4

**Implement a Transfer Learning concept in Image Classification**

**AIM**

To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

**THEORY**

Neural Network Model

<img width="814" height="324" alt="image" src="https://github.com/user-attachments/assets/3b5ab9d2-6624-407a-81e8-d1a3bab7aa8f" />


**DESIGN STEPS**

STEP 1: We begin by importing the necessary Python libraries, including TensorFlow for deep learning, data preprocessing tools, and visualization libraries.

STEP 2: To leverage the power of GPU acceleration, we configure TensorFlow to allow GPU processing, which can significantly speed up model training.

STEP 3: We load the dataset, consisting of cell images, and check their dimensions. Understanding the image dimensions is crucial for setting up the neural network architecture.

STEP 4: We create an image generator that performs data augmentation, including rotation, shifting, rescaling, and flipping. Data augmentation enhances the model's ability to generalize and recognize malaria-infected cells in various orientations and conditions.

STEP 5: We design a convolutional neural network (CNN) architecture consisting of convolutional layers, max-pooling layers, and fully connected layers. The model is compiled with appropriate loss and optimization functions.

STEP 6: We split the dataset into training and testing sets, and then train the CNN model using the training data. The model learns to differentiate between parasitized and uninfected cells during this phase.

STEP 7: We visualize the training and validation loss to monitor the model's learning progress and detect potential overfitting or underfitting.

STEP 8: We evaluate the trained model's performance using the testing data, generating a classification report and confusion matrix to assess accuracy and potential misclassifications.

STEP 9: We demonstrate the model's practical use by randomly selecting and testing a new cell image for classification.

**PROGRAM**

**Name:vasanth kumar v**

**Register Number:2305002027**

**PROGRAM**
```python
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix

%matplotlib inline
my_data_dir = 'dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'

os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[0]

para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])

plt.imshow(para_img)
# Checking the image dimensions
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)

image_shape = (130,130,3)
# help(ImageDataGenerator)

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degreeswidth_shift_range=0.10, # Shift the pic width by a max of 5% height_shift_range=0.10,
                               # Shift the pic height by a max of 5 % rescale=1/255, # Rescale the image by normalzing it.
                                shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

image_gen.flow_from_directory(train_path)

image_gen.flow_from_directory(test_path)



model = models.Sequential()
model.add(keras.Input(shape=(image_shape)))
model.add(layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
batch_size = 16

# help(image_gen.flow_from_directory)

train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')

train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=5,
                              validation_data=test_image_gen)
                             
model.save('cell_model.h5')
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)


import random
list_dir=["Un Infected","parasitized"]
dir_=(random.choice(list_dir))
p_img=imread(train_path+'/'+dir_+'/'+os.listdir(train_path+'/'+dir_)[random.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(p_img))
img = tf.image.resize(img,(130,130))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred  else "Un Infected")
			+"\nActual Value: "+str(dir_))
plt.axis("off")
plt.imshow(img)
plt.show()


```

**OUTPUT**

**Training Loss, Validation Loss Vs Iteration Plot**

<img width="855" height="537" alt="image" src="https://github.com/user-attachments/assets/dd94a949-3b22-407b-a6d9-774ab562f124" />


**Confusion Matrix**

<img width="610" height="146" alt="image" src="https://github.com/user-attachments/assets/d4a8dcb7-bb3a-468d-b680-8877f3cc64b6" />

**Classification Report**

<img width="625" height="499" alt="image" src="https://github.com/user-attachments/assets/93517994-3563-4ce8-963e-e6d3bdde9489" />


**New Sample Data Prediction**

<img width="648" height="151" alt="image" src="https://github.com/user-attachments/assets/f5d873ec-a961-4d89-bd31-27c4ad36acf8" />
<img width="633" height="630" alt="image" src="https://github.com/user-attachments/assets/d23b22ea-c847-4f78-9621-444ce4a4ee42" />


**RESULT**
The model's performance is evaluated through training and testing, and it shows potential for assisting healthcare professionals in diagnosing malaria more efficiently and accurately
