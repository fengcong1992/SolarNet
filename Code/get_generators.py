import numpy as np
import keras
import pickle

class DataGenerator_SCNN(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe, batch_size=2, dim=(256, 256, 3), channel_IMG = 3, shuffle=True, iftest = False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe.reset_index(drop=True)
        self.channel_IMG = channel_IMG
        self.shuffle = shuffle
        self.on_epoch_end()
        self.iftest = iftest
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        dataframe_temp = self.dataframe.iloc[indexes]
        X, y = self.__data_generation(dataframe_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, dataframe_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim), dtype=float)
        y = np.empty((self.batch_size), dtype=float)
        file_name = []
        dataframe_temp2 = dataframe_temp.reset_index(drop=True)
        for [n_index, vector_row] in dataframe_temp2.iterrows():
            img_row = vector_row.Image
            value_row = vector_row.Target
            for i_IMG in range(len(img_row)): # stack images in channels
                f = open(img_row[i_IMG], 'rb')
                X[n_index, :, :, 0 + self.channel_IMG * i_IMG:self.channel_IMG * (i_IMG + 1)] = pickle.load(f,encoding='uint8') / 255.0
                f.close()
            y[n_index] = value_row
            file_name.append(img_row)
        if self.iftest == True:
            return X, y, file_name
        elif self.iftest == False:
            return X, y
        
        
        
        
class DataGenerator_CNN3D(keras.utils.Sequence):
    # My first self-defined data generator
    # used to generate batches of [one sky image, a value] from pickles
    'Generates data for Keras'
    def __init__(self, dataframe, batch_size=2, dim=(256, 256, 2, 3), channel_IMG = 3, shuffle=True, iftest = False):
        'Initialization'
        self.dim = dim
        #print(dim)
        self.batch_size = batch_size
        self.dataframe = dataframe.reset_index(drop=True)
        self.channel_IMG = channel_IMG
        self.shuffle = shuffle
        self.on_epoch_end()
        self.iftest = iftest
        #print(dim)
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        dataframe_temp = self.dataframe.iloc[indexes]
        X, y = self.__data_generation(dataframe_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, dataframe_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        #print('Indicator1')
        X = np.empty((self.batch_size, *self.dim), dtype=float)
        y = np.empty((self.batch_size), dtype=float)
        file_name = []
        dataframe_temp2 = dataframe_temp.reset_index(drop=True)
        for [n_index, vector_row] in dataframe_temp2.iterrows():
            img_row = vector_row.Image
            value_row = vector_row.Target
            for i_IMG in range(len(img_row)): # stack images in channels
                f = open(img_row[i_IMG], 'rb')
                X[n_index, :, :, i_IMG, :] = pickle.load(f,encoding='uint8') / 255.0
                f.close()
            y[n_index] = value_row
            file_name.append(img_row)
        if self.iftest == True:
            return X, y, file_name
        elif self.iftest == False:
            return X, y