from keras.layers import Conv2D, MaxPooling2D
from keras import Sequential
from keras import models
from keras import layers

def SCNN(input_shape=(256, 256, 3)):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    return model


def CNN3D(input_shape=(256, 256, 2, 3)):
    model = models.Sequential()
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape, padding='same', data_format = 'channels_last'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(layers.Conv3D(128, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(layers.Conv3D(128, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(layers.Conv3D(256, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(layers.Conv3D(256, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    return model


