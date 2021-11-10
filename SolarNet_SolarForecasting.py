import pandas as pd
import os, sys, pickle
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
import tensorflow as tf
import subprocess, argparse

from get_model import *
from get_generators import *

n_GPUs = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.clear_session()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default = 256, help="Input sky image resolution")
    parser.add_argument("--no_img", type=int, default= 2, help="Length of input image sequence")
    parser.add_argument("--train_batchsize", type=int, default= 64)
    parser.add_argument("--validation_batchsize", type=int, default= 64)
    parser.add_argument("--test_batchsize", type=int, default= 1)
    args = parser.parse_args()
    return args

args = get_args()
img_size = args.img_size
no_img = args.no_img
train_batchsize = args.train_batchsize
validation_batchsize = args.validation_batchsize
test_batchsize = args.test_batchsize

start_train = pd.to_datetime('2012-01-01')
end_train = pd.to_datetime('2014-12-31')
end_validation = pd.to_datetime('2015-12-31')
end_test = pd.to_datetime('2017-12-31')

file = open('Data.pkl','rb')
df_train = pickle.load(file)
df_validation = pickle.load(file)
df_test = pickle.load(file)
file.close()

params1 = {'batch_size': train_batchsize,
           'dim': (img_size, img_size, 3 * no_img),
           'channel_IMG': 3,
           'shuffle': True,
           'iftest': False}
params2 = {'batch_size': validation_batchsize,
           'dim': (img_size, img_size, 3 * no_img),
           'channel_IMG': 3,
           'iftest': False}
params3 = {'batch_size': test_batchsize,
           'dim': (img_size, img_size, 3 * no_img),
           'channel_IMG': 3,
           'shuffle': False,
           'iftest': False}

train_generator = DataGenerator_SCNN(df_train, **params1)
validation_generator = DataGenerator_SCNN(df_validation, **params2)
test_generator = DataGenerator_SCNN(df_test, **params3)

conv_base = SCNN(input_shape=(img_size, img_size, 3 * no_img))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation='linear'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='linear'))
model.summary()
conv_base.summary()

# compile model
optimizer = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
class MyModelCheckPoint(ModelCheckpoint):
    def __init__(self, singlemodel, *args, **kwargs):
        self.singlemodel = singlemodel
        super(MyModelCheckPoint, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.singlemodel
        super(MyModelCheckPoint, self).on_epoch_end(epoch, logs)
checkpointer1 = MyModelCheckPoint(model, filepath="BestSingleModel.hdf5", save_best_only=True, verbose=vb)
# HPC parallel computing
parallel_model = multi_gpu_model(model, gpus=n_GPUs)
parallel_model.compile(loss='mean_absolute_error',  # mean_squared_error
                       optimizer=optimizer,
                       metrics=['mae'])
history = parallel_model.fit_generator(train_generator,
                                       steps_per_epoch=int(df_train.shape[0] / train_batchsize),
                                       epochs=60,
                                       validation_data=validation_generator,
                                       validation_steps=int(df_validation.shape[0] / validation_batchsize),
                                       callbacks=[checkpointer1],
                                       verbose=2,
                                       workers=6,
                                       use_multiprocessing=False)
model.save(os.path.join('Last_SingleModel.hdf5'))

# predict
try:
    model.load_weights(os.path.join('BestSingleModel.hdf5'))
    print('Succss in loading the best single model')
except:
    print('Fail to load the best single model')
    pass

y_hat = model.predict_generator(generator=test_generator, steps=df_test.shape[0] / test_batchsize)



