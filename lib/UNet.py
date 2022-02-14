import numpy as np
import random
import tensorflow as tf
# tf_core sliences pylance warnings for some reason
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from imutils import paths
import cv2
# import glob
from skimage.io import imshow
import matplotlib.pyplot as plt
import argparse


def train(path, split, epochs, batch_size, r_patience, e_patience):

    # Verify split
    if split <= .5 and split > 0:
        split = split
    else: 
        # print("UNet.py: invalid argument \"--split {}\": split should be between 0 and .5".format(args.split))
        split = .05
    
    print("UNet.py: using split={}".format(split))

    ################### GENERAL HYPERPARAMETERS ######################
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ################### SPECIFIC HYPERPARAMETERS #####################
    r_patience = r_patience
    e_patience = e_patience
    epochs = epochs
    # split = 0.05
    LR = 0.0001
    #img_size = 256
    #path = 'soft_reduced'
    #path = 'test'
    # path = 'hardwoods'
    batch_size = batch_size
    n_channels = 3
    filters = [96, 128, 256, 386, 256, 64] #Working solution is [96, 128, 256, 386, 256, 128]
    desired_dim = 256

    ######################## LOAD DATA FUNCTION #####################
    def analyze_load(path):
        # global desired_dim
        print('[INFO]: Loading images and labels from \'{}\'\n'.format(path))
        #loop over the folder and grab images
        imagepaths = list(paths.list_images("../../" + path))
        # print(imagepaths)
        data = []
        labels = []
        for imagePath in imagepaths:
            #grab the labels
            label = imagePath.split(os.path.sep)[-2]
            #grab the images
            #image = cv2.imread(imagePath,0) # Used for grayscale images
            image = cv2.imread(imagePath)
            # print(image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = cv2.resize(image, (img_size, img_size)) #used for grayscale
            ########### RESIZE IMAGE TRICK #####################
            y = (image.shape[0] - desired_dim) // 2
            x = (image.shape[1] - desired_dim) // 2
            h=desired_dim
            w=desired_dim   
            image = image[y:y+h, x:x+w]
            #image = image.reshape((image.shape[0], image.shape[1], n_channels)) #Used for grayscale images
            
            # Determine shape
            # TODO proper image cropping/resizing
            if image.shape != (256, 256, 3):
                print("="*20 + 
                    "\n{}: bad size, image.shape = {}, should be (256, 256, 3)".format(imagePath, image.shape) + 
                    "\nNot including image in train/test set\n" + "="*20)
            else:
                data.append(image)
                labels.append(label)
                
        #sanity test for labels and images
        print('[INFO]:',len(labels), 'labels have been assigned to images\n',)
        print('[INFO]:',len(data), 'images have been loaded to memory\n')

        #convert both data and labels to a list of numpy array
        data = np.array(data, dtype='float32') / 255.0
        # rnd = random.randint(0, data.shape[0])
        # img = cv2.imshow('test_image',data[rnd]) #Press ESC
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        labels = np.array(labels)
        classes = np.unique(labels)
        classes = list(classes)
        n_classes = len(np.unique(labels))
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        labels = tf.keras.utils.to_categorical(labels, n_classes)
        # print(labels, labels.shape, len(labels))

        return data, labels, n_classes, classes

    images, labels, _, __ = analyze_load(path)

    images_training, images_val, y_train, y_test = train_test_split(images, labels, 
        test_size = split, random_state = seed)

    with open('labels_test.npy', 'wb') as l_test:
        np.save(l_test, y_test)

    with open('labels_train.npy', 'wb') as l_tr:
        np.save(l_tr, y_train)

    print(type(images_training), len(images_training), images_training.shape)


    def get_model():

    #========================== ENCODER ======================================

        inputs = Input((images.shape[1],images.shape[2],n_channels))

        conv1 = Conv2D(filters[0], 3, 1, padding="same", activation = 'relu')(inputs)
        maxp1 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv1)

        conv2 = Conv2D(filters[1], 3, 1, activation = 'relu', padding="same")(maxp1)
        maxp2 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv2)

        conv3 = Conv2D(filters[2], 3, 1, activation = 'relu', padding="same")(maxp2)
        maxp3 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv3)

        conv4 = Conv2D(filters[3], 3, 1, activation = 'relu', padding="same")(maxp3)
        maxp4 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv4)

        conv5 = Conv2D(filters[4], 3, 1, activation = 'relu', padding="same")(maxp4)
        maxp5 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv5)

        conv6 = Conv2D(filters[5], 3, 1, activation = 'relu', padding="same")(maxp5)
        maxp6 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv6)

        flat = Flatten(name = 'flatten')(maxp6)

        shape_before_flatten = tf.keras.backend.int_shape(maxp6)[1:]

        den = Dense(np.prod(shape_before_flatten), name = 'dense')(flat)

    #=================================DECODER======================================

        resh = Reshape((shape_before_flatten))(den)

        convt2 = Conv2DTranspose(filters[5], (2,2), strides = (2,2))(resh)
        merge1 = concatenate([convt2, conv6])
        conv9 = Conv2D(filters[5], 3, 1, activation = 'relu', padding="same")(merge1)

        convt3 = Conv2DTranspose(filters[4], (2,2), strides = (2,2))(conv9)
        merge2 = concatenate([convt3, conv5])
        conv10 = Conv2D(filters[4], 3, 1, activation = 'relu', padding="same")(merge2)

        convt4 = Conv2DTranspose(filters[3], (2,2), strides = (2,2))(conv10)
        merge3 = concatenate([convt4, conv4])
        conv11 = Conv2D(filters[3], 3,1,  activation = 'relu', padding="same")(merge3)

        convt5 = Conv2DTranspose(filters[2], (2,2), strides = (2,2))(conv11)
        merge4 = concatenate([convt5, conv3])
        conv12 = Conv2D(filters[2], 3,1,  activation = 'relu', padding="same")(merge4)

        convt6 = Conv2DTranspose(filters[1], (2,2), strides = (2,2))(conv12)
        merge5 = concatenate([convt6, conv2])
        conv13 = Conv2D(filters[1], 3,1,  activation = 'relu', padding="same")(merge5)

        convt7 = Conv2DTranspose(filters[0], (2,2), strides = (2,2))(conv13)
        merge6 = concatenate([convt7, conv1])
        conv14 = Conv2D(filters[0], 3,1,  activation = 'relu', padding="same")(merge6)

        output = Conv2D(n_channels, 1, 1, padding="valid", activation = 'sigmoid')(conv14)#<---- IT NEEDS TO BE SIGMOID. 

        autoencoder = Model(inputs=[inputs], outputs=[output])

        print(autoencoder.summary())
        #===================================================================

        plot_model(autoencoder, to_file='architecture.png', show_shapes=True, show_layer_names=True)

        opt = Adam(lr=LR)
        autoencoder.compile(optimizer=opt, loss='mse') #<--- IT NEEDS TO BE MSE. 

        return autoencoder
    ############################# DISTRIBUTED STRATEGY ################################

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        autoencoder = get_model()

    ######################### MODEL FIT #############################################

    tensorboard_log_dir = "logs/fit/"
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss',patience = r_patience, verbose = 1),
        tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = e_patience, verbose = 1),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1),
    ]

    print('[INFO]: Training net for {} epochs'.format(epochs))

    history = autoencoder.fit(
        images_training, images_training, validation_data = (images_val, images_val), 
        batch_size = batch_size, epochs = epochs, callbacks = callbacks,
    )

    ############################## PLOTTING HISTORY ####################################
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss.png')

    ########################### PLOTTING AUTOENCODER IMAGE GENERATION ###########################

    # batch predict (doesn't work ?)
    # pred = []
    rnd = random.randint(0, images_training.shape[0])
    pred = autoencoder.predict(images_training, batch_size=128)
    # pred = autoencoder(np.array([images_training[rnd]], ndim=4))

    fig, ax = plt.subplots(1,2)
    # pred = autoencoder(images_training)

    ax[0].imshow(np.squeeze(images_training[rnd]))
    ax[0].set_title('Ground-truth image')
    ax[1].imshow(np.squeeze(pred[rnd])) #cmap = 'gray' # removed rnd index from pred for single prediction
    ax[1].set_title('Autoencoder image')
    plt.savefig('image_generation.png')

    ########################## GETTING LATENT VECTOR ####################################
    get_latent = tf.keras.models.Model(autoencoder.inputs, autoencoder.get_layer("dense").output)
    latent = get_latent.predict(images_training)
    print("Latent type and shape are: ", type(latent), latent.shape)

    with open('latent.npy', 'wb') as f:
        np.save(f, latent)
        
    return latent, y_test, y_train