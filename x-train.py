from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback

from model import get_model
from utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation

import gc

import os

import click

DATA_DIR = '../'

def load_train_data(data_prefix, seed):
    """
    Load training data from .npy files.
    """
    X = np.load(data_prefix + 'X-train.npy')
    y = np.load(data_prefix + 'y-train.npy')

    X = X.astype(np.float32, copy=False)
    X /= 255

    # seed = np.random.randint(1, 10e6)
    # add seed to name
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X, y


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test


def hard_train(data_prefix, prefix, seed, col):
    what = ['systole', 'diastole'][col % 2]
    print('We are going to train hard {} {}'.format(what, col))
    print('Loading training data...')

    X, y = load_train_data(data_prefix, seed)
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.2)

    model = get_model()

    nb_iter = 200
    epochs_per_iter = 1
    batch_size = 32

    min_val = sys.float_info.max


    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


    print('-'*50)
    print('Training...')
    print('-'*50)

    datagen.fit(X_train)


    checkpointer_best = ModelCheckpoint(filepath=prefix + "weights_{}_best.hdf5".format(what), verbose=1, save_best_only=True)
    checkpointer = ModelCheckpoint(filepath=prefix + "weights_{}.hdf5".format(what), verbose=1, save_best_only=False)

    hist = model.fit_generator(datagen.flow(X_train, y_train[:, col], batch_size=batch_size),
                                           samples_per_epoch=X_train.shape[0],
                                           nb_epoch=nb_iter, show_accuracy=False,
                                           validation_data=(X_test, y_test[:, col]),
                                           callbacks=[checkpointer, checkpointer_best],
                                           nb_worker=4)

    loss = hist.history['loss'][-1]
    val_loss = hist.history['val_loss'][-1]

    with open(prefix + 'val_loss.txt', mode='w+') as f:
        f.write(str(min(hist.history['val_loss'])))
        f.write('\n')





def train(data_prefix, prefix, seed, run):
    """
    Training systole and diastole models.
    """
    print('Loading training data...')
    X, y = load_train_data(data_prefix, seed)


    print('Loading and compiling models...')
    model_systole = get_model()
    model_diastole = get_model()

    # split to training and test
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.2)

    nb_iter = 200
    epochs_per_iter = 1
    batch_size = 32
    calc_crps = 1  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_systole = sys.float_info.max
    min_val_loss_diastole = sys.float_info.max

    print('-'*50)
    print('Training...')
    print('-'*50)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)

    datagen.fit(X_train)

    systole_checkpointer_best = ModelCheckpoint(filepath=prefix + "weights_systole_best.hdf5", verbose=1, save_best_only=True)
    diastole_checkpointer_best = ModelCheckpoint(filepath=prefix + "weights_diastole_best.hdf5", verbose=1, save_best_only=True)
    systole_checkpointer = ModelCheckpoint(filepath=prefix + "weights_systole.hdf5", verbose=1, save_best_only=False)
    diastole_checkpointer = ModelCheckpoint(filepath=prefix + "weights_diastole.hdf5", verbose=1, save_best_only=False)


    if run == 0 or run == 1:
        print('Fitting Systole Shapes')
        hist_systole = model_systole.fit_generator(datagen.flow(X_train, y_train[:, 2], batch_size=batch_size),
                                                   samples_per_epoch=X_train.shape[0],
                                                   nb_epoch=nb_iter, show_accuracy=False,
                                                   validation_data=(X_test, y_test[:, 2]),
                                                   callbacks=[systole_checkpointer, systole_checkpointer_best],
                                                   nb_worker=4)

    if run == 0 or run == 2:
        print('Fitting Diastole Shapes')
        hist_diastole = model_diastole.fit_generator(datagen.flow(X_train, y_train[:, 2], batch_size=batch_size),
                                                     samples_per_epoch=X_train.shape[0],
                                                     nb_epoch=nb_iter, show_accuracy=False,
                                                     validation_data=(X_test, y_test[:, 2]),
                                                     callbacks=[diastole_checkpointer, diastole_checkpointer_best],
                                                     nb_worker=4)

    if run == 0 or run == 1:
        loss_systole = hist_systole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]

    if run == 0 or run == 2:
        loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_diastole = hist_diastole.history['val_loss'][-1]

    if calc_crps > 0 and run == 0:
        print('Evaluating CRPS...')
        pred_systole = model_systole.predict(X_train, batch_size=batch_size, verbose=1)
        val_pred_systole = model_systole.predict(X_test, batch_size=batch_size, verbose=1)

        pred_diastole = model_diastole.predict(X_train, batch_size=batch_size, verbose=1)
        val_pred_diastole = model_diastole.predict(X_test, batch_size=batch_size, verbose=1)

        # CDF for train and test data (actually a step function)
        cdf_train = real_to_cdf(np.concatenate((y_train[:, 0], y_train[:, 1])))
        cdf_test = real_to_cdf(np.concatenate((y_test[:, 0], y_test[:, 1])))

        # CDF for predicted data
        cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
        cdf_val_pred_systole = real_to_cdf(val_pred_systole, val_loss_systole)

        cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)
        cdf_val_pred_diastole = real_to_cdf(val_pred_diastole, val_loss_diastole)

        # evaluate CRPS on training data
        crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
        print('CRPS(train) = {0}'.format(crps_train))

        # evaluate CRPS on test data
        crps_test = crps(cdf_test, np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
        print('CRPS(test) = {0}'.format(crps_test))

    # save best (lowest) val losses in file (to be later used for generating submission)
    with open(prefix + 'val_loss.txt', mode='w+') as f:
        if run == 0 or run == 1:
            f.write(str(min(hist_systole.history['val_loss'])))
            f.write('\n')
        if run == 0 or run == 2:
            f.write(str(min(hist_diastole.history['val_loss'])))


@click.command()
@click.option('--col', default=0)
def main(col):
    seed = 19595
    data_prefix = 'dry-run/pre-'
    prefix = 'dry-run/{}-{}-mm2-'.format(seed, col)
    hard_train(data_prefix, prefix, seed, col)


if __name__ == "__main__":
    main()