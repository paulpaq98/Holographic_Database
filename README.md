# Holographic Database

This repository contains the minimal datset for the neural networks described in

Spatio-temporal based deep learning for rapid detection and identification of bacterial colonies through lens-free microscopy time-lapses
Paul Paquin, Claire Durmort, Caroline Paulus, Thierry Vernet, Pierre R. Marcoux, Sophie Morales

for detection and classification tasks

Our models were implemented using python 3.7.0 together with Tensorflow 2.1.0 and the 2.3.1 Keras interface.

# Data

The data in this repository is a sample from a database of holographic 12h time-lapses (one image = 30 minutes) of seven pathogenic bacterial species taken with Iprasense lens-free imaging system (Cytonote6) on thin-layer BHI Agar mediums (150 microns) collected by the Institut de Biologie Structurelle (IBS, France).


# Detection Model

The detection model used for this dataset is a Mask-RCNN model as described in 

He K, Gkioxari G, Dollár P, Girshick R. Mask R-CNN. ArXiv170306870 Cs. 2018.
Available: http://arxiv.org/abs/1703.06870

Matterport's implementation was used to train our model

https://github.com/matterport/Mask_RCNN

# Classification Model

The classification model used for this dataset is a CONV3D + LSTM2D model

For a 7-frame classification model, the implementation in Keras would be 

```
class Conv3DModel(tf.keras.Model):
    package_frame = 0

    def __init__(self,package_frame):
        self.package_frame = package_frame
        super(Conv3DModel, self).__init__()
        
          # Convolutions
          self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (2, 2, 2), activation='relu', name="conv1", data_format='channels_last')
          self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
          self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (1, 1, 1), activation='relu', name="conv1", data_format='channels_last')
          self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')
        
        # Dense layers
        # LSTM & Flatten
        self.convLSTM =tf.keras.layers.ConvLSTM2D(30, (3, 3))
        self.flatten =  tf.keras.layers.Flatten(name="flatten")
        self.d1 = tf.keras.layers.Dense(64, activation='relu', name="d1")
        

    def call(self, x):

        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)      
        x = self.convLSTM(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.out(x)
```
