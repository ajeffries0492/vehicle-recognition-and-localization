import numpy as np
from tensorflow import keras

from utils.summarize_data import get_image, rescale_image


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels,input_location='.' ,batch_size=32, dim=(100,100),n_channels=3,
    n_classes=10, shuffle=True,XBoxes=None):
        'Initialization'
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.input_location = input_location
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.boxes = XBoxes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim,self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            

            # Get Image from File Location
            image = get_image(f"{self.input_location}/{ID}")
            if self.boxes is not None:
                # Grab Box Index
                min_x, min_y,max_x, max_y = self.boxes[ID]
                # Crop Image
                image = image[min_y:max_y,min_x:max_x,:]

            
            
            # DownSample Data and Store sample
            X[i,] =  rescale_image(image,dim=self.dim)

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
