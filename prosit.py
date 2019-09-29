
from lib import io
from lib import losses
from lib import model as model_lib
from lib import constants
from lib import layers


import numpy as np

from lib import sanitize

import keras

import scipy
from keras.callbacks import Callback
import sklearn

import pickle

import pandas as pd

import matplotlib
matplotlib.use("agg")
import argparse
import sys


import os
import gc
import keras.backend as K
import tensorflow as tf
import json


## only for RT prediction
def rt_predict(test_file, model_dir, batch_size=64, out_dir="./", prefix="test"):

    model_para_file = model_dir+"/model.json"
    with open(model_para_file, "r") as read_file:
        model_list = json.load(read_file)

    model = load_prediction_model(model_dir)
    test_x = io.processing_prediction_data(model_para_file,test_file)
    y_pred = model.predict(test_x, batch_size=batch_size)
    y_pred.reshape([y_pred.shape[0], 1])

    rt_pred = minMaxScoreRev(y_pred, model_list['min_rt'], model_list['max_rt'])

    input_data = pd.read_csv(test_file, sep="\t", header=0, low_memory=False)
    input_data['y_pred'] = rt_pred

    ## output
    out_file = out_dir + "/" + prefix + ".csv"
    input_data.to_csv(out_file, sep="\t", index=False)


    if "y" in input_data.columns.values:
        y_true = minMaxScale(np.asarray(input_data['y']),model_list['min_rt'], model_list['max_rt'])
        evaluation_res = model.evaluate(test_x, y_true)
        print("Metrics:")
        print(evaluation_res)
        out_prefix = prefix + "_" + "evaluate"
        evaluate_model(y_true, y_pred, para=model_list, plot=True, out_dir=out_dir, prefix=out_prefix)


## only for RT prediction
def evaluate_model(y_t, y_p, para=None, plot=True, out_dir="./", prefix="test"):
    y_t = minMaxScoreRev(y_t, para['min_rt'], para['max_rt'])
    y_p = minMaxScoreRev(y_p, para['min_rt'], para['max_rt'])
    y2 = pd.DataFrame({"y": y_t, "y_pred": y_p.reshape(y_p.shape[0])})
    cor = scipy.stats.pearsonr(y2['y'], y2['y_pred'])[0]
    mae = sklearn.metrics.mean_absolute_error(y2['y'], y2['y_pred'])
    r2 = sklearn.metrics.r2_score(y2['y'], y2['y_pred'])
    abs_median = float(np.median(np.abs(y2['y'] - y2['y_pred'])))
    d_t95 = calc_delta_t95(y2['y'], y2['y_pred'])
    print('Cor: %s, MAE: %s, R2: %s, abs_median_e: %s, dt95: %s' % (
        str(round(cor, 4)), str(round(mae, 4)),
        str(round(r2, 4)), str(round(abs_median, 4)),
        str(round(d_t95, 4))), end=100 * ' ' + '\n')

    ## output
    out_file = out_dir + "/" + prefix + ".csv"
    y2.to_csv(out_file)


## only for RT prediction
class RegCallback(Callback):
    """
    Calculate AUROC for each epoch
    """

    def __init__(self, X_train, X_test, y_train, y_test, min_rt=0, max_rt=120):
        self.x = X_train
        self.y = minMaxScoreRev(y_train,min_rt,max_rt)
        self.x_val = X_test
        self.y_val = minMaxScoreRev(y_test,min_rt,max_rt)
        self.min_rt = min_rt
        self.max_rt = max_rt

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs=None):

        ## training data
        y_pred = self.model.predict(self.x)
        y_pred_rev = minMaxScoreRev(y_pred, self.min_rt, self.max_rt)

        y1 = pd.DataFrame({"y": self.y, "y_pred": y_pred_rev.reshape(y_pred_rev.shape[0])})
        cor1 = scipy.stats.pearsonr(y1['y'],y1['y_pred'])[0]
        mae1 = sklearn.metrics.mean_absolute_error(y1['y'],y1['y_pred'])
        r21 = sklearn.metrics.r2_score(y1['y'],y1['y_pred'])
        abs_median1 = np.median(np.abs(y1['y'] - y1['y_pred']))
        d_t951 = calc_delta_t95(y1['y'], y1['y_pred'])
        ## test data
        y_pred_val = self.model.predict(self.x_val)
        y_pred_val_rev = minMaxScoreRev(y_pred_val, self.min_rt, self.max_rt)
        y2 = pd.DataFrame({"y": self.y_val, "y_pred": y_pred_val_rev.reshape(y_pred_val_rev.shape[0])})
        cor2 = scipy.stats.pearsonr(y2['y'], y2['y_pred'])[0]
        mae2 = sklearn.metrics.mean_absolute_error(y2['y'], y2['y_pred'])
        r22 = sklearn.metrics.r2_score(y2['y'], y2['y_pred'])
        abs_median2 = np.median(np.abs(y2['y'] - y2['y_pred']))
        d_t952 = calc_delta_t95(y2['y'], y2['y_pred'])
        print('\nCor: %s - Cor_val: %s, MAE: %s - MAE_val: %s, R2: %s - R2_val: %s, MedianE: %s - MedianE_val: %s, dt95: %s - dt95_val: %s' % (str(round(cor1, 4)), str(round(cor2, 4)),
                                                                                       str(round(mae1, 4)), str(round(mae2, 4)),
                                                                                       str(round(r21, 4)), str(round(r22, 4)),
                                                                                       str(round(abs_median1, 4)), str(round(abs_median2, 4)),
                                                                                       str(round(d_t951, 4)), str(round(d_t952, 4))), end=100 * ' ' + '\n')

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def calc_delta_t95(obs, pred):
    q95 = int(np.ceil(len(obs) * 0.95))
    return 2 * sorted(abs(obs - pred))[q95 - 1]

def minMaxScale(x, min=0,max=120):
    new_x = 1.0*(x-min)/(max-min)
    return new_x

def minMaxScoreRev(x,min=0,max=120):
    old_x = x * (max - min) + min
    return old_x



def get_callbacks(model_dir_path):
    import keras

    loss_format = "{val_loss:.5f}"
    epoch_format = "{epoch:02d}"
    weights_file = "{}/weight_{}_{}.hdf5".format(
        model_dir_path, epoch_format, loss_format
    )
    save = keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True)
    stop = keras.callbacks.EarlyStopping(patience=10)
    decay = keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2)

    model_chk_path = model_dir_path + "/best_model.hdf5"

    mcp = keras.callbacks.ModelCheckpoint(model_chk_path, save_best_only=True,
                          save_weights_only=False,
                          verbose=1)

    return [save, stop, decay, mcp]


def train(tensor, model_dir, out_dir="./", refine_model=False):
    import keras

    callbacks = get_callbacks(out_dir)
    model = load_model(model_dir=model_dir, refine_model=refine_model)
    #print(model_config)
    print(tensor)

    if isinstance(tensor,dict):
        x = tensor['X_train']
        y = tensor['Y_train']
    else:
        print("TODO")
        #x = io.get_array(tensor, model_config["x"])
        #y = io.get_array(tensor, model_config["y"])

    print("start to train ...")
    print("epoch: %d" % (constants.TRAIN_EPOCHS))
    print("batch size: %d" % (constants.TRAIN_BATCH_SIZE))
    print("validation split: %f" % (1 - constants.VAL_SPLIT))
    print("X shape:")
    print(type(x))
    print(x)
    print(np.asarray(x).shape[1:])
    print("Range of x: %f - %f" % (np.asarray(x).min(),np.asarray(x).max()))
    print("Y shape:")
    print(type(y))
    print(y)
    print(np.asarray(y).shape[1:])
    print("Range of y: %f - %f" % (np.asarray(y).min(), np.asarray(y).max()))

    # save training data to files
    with open('train_x.pkl', 'wb') as x2file:
        pickle.dump(x, x2file)

    with open('train_y.pkl', 'wb') as y2file:
        pickle.dump(y, y2file)


    if isinstance(tensor,dict):
        my_callbacks = RegCallback(tensor['X_train'], tensor['X_test'], tensor['Y_train'], tensor['Y_test'], min_rt=tensor['min_rt'], max_rt=tensor['max_rt'])
        callbacks.append(my_callbacks)
        model.fit(
            x=x,
            y=y,
            epochs=constants.TRAIN_EPOCHS,
            batch_size=constants.TRAIN_BATCH_SIZE,
            validation_split=1 - constants.VAL_SPLIT,
            callbacks=callbacks,
        )
        ## save model information to files
        para = dict()
        para['min_rt'] = tensor['min_rt']
        para['max_rt'] = tensor['max_rt']
        para['max_x_length'] = tensor['max_x_length']
        para['aa'] =  tensor['aa']

        ## save result
        model_json = out_dir + "/model.json"
        with open(model_json, 'w') as f:
            json.dump(para, f)


    else:
        ## original paper hdf5 format data
        model.fit(
            x=x,
            y=y,
            epochs=constants.TRAIN_EPOCHS,
            batch_size=constants.TRAIN_BATCH_SIZE,
            validation_split=1 - constants.VAL_SPLIT,
            callbacks=callbacks,
        )


    keras.backend.get_session().close()


def load_model(model_dir, refine_model=False):
    print("Refine model: " + str(refine_model))
    model, model_config = model_lib.load(model_dir, trained=refine_model)
    model.summary()
    print(model_config)
    print(model_config["loss"])

    if isinstance(model_config["loss"], list):
        loss = [losses.get(l) for l in model_config["loss"]]
    else:
        loss = losses.get(model_config["loss"])
    optimizer = model_config["optimizer"]
    print(optimizer)
    model.compile(optimizer=optimizer, loss=loss)
    return model

def load_prediction_model(model_dir):
    import keras

    model_file = model_dir + "/best_model.hdf5"
    model = keras.models.load_model(model_file, custom_objects={"Attention": layers.Attention})
    model.summary()
    return model



def main():

    if len(sys.argv) == 1:
        print("python prosit.py [train, predict]")
        sys.exit(0)
    else:

        mode = sys.argv[1]

        if mode == "train":

            print("Run training!")
            parser = argparse.ArgumentParser(description='Prosit')
            parser.add_argument('-i', '--input', default=None, type=str, required=True,
                                help="Input data for training")

            parser.add_argument('-md', '--model_dir', default=None, type=str, required=True,
                                help="Model directory")

            parser.add_argument('-rf', '--refine_model', action='store_true')

            parser.add_argument('-t', '--test', default=None, type=str,
                                help="Input data for testing")

            parser.add_argument('-o', '--out_dir', default="./", type=str,
                                help="Output directory")

            parser.add_argument('-e', '--epochs', default=None, type=int)
            parser.add_argument('-b', '--batch_size', default=None, type=int)
            parser.add_argument('-r1', '--min_rt', default=0, type=int)
            parser.add_argument('-r2', '--max_rt', default=0, type=int)
            parser.add_argument('-l', '--max_length', default=0, type=int)
            parser.add_argument('-m', '--mod', default=None, type=str)
            parser.add_argument('-u', '--unit', default="s", type=str)

            if len(sys.argv) == 1:
                parser.print_usage()
                sys.exit(0)

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            input_file = args.input
            model_dir = args.model_dir
            refine_model = args.refine_model
            test_file = args.test
            out_dir = args.out_dir
            max_rt = args.max_rt
            min_rt = args.min_rt
            max_length = args.max_length
            mod = args.mod
            unit = args.unit

            if mod is not None:
                mod = mod.split(",")

            epochs = args.epochs
            if epochs is None:
                epochs = constants.TRAIN_EPOCHS
                print("Use default number of epochs: %d" % (epochs))
            else:
                print("Use user-defined number of epochs: %d" % (epochs))

            batch_size = args.batch_size
            if batch_size is None:
                batch_size = constants.TRAIN_BATCH_SIZE
                print("Use default batch size: %d" % (batch_size))
            else:
                print("Use user-defined batch size: %d" % (batch_size))

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            tensor = None
            if input_file.endswith("hdf5"):
                tensor = io.from_hdf5(input_file)
            else:
                # tensor is a dict object which contains training data and other useful information about the data
                tensor = io.data_processing(input_data=input_file, mod=mod, unit=unit, max_x_length=max_length, out_dir=out_dir, min_rt = min_rt, max_rt = max_rt)

            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # turn off tf logging
            # data_path = constants.DATA_PATH
            # model_dir = constants.MODEL_DIR
            ## save best model

            train(tensor, model_dir, out_dir=out_dir, refine_model=refine_model)

        elif mode == "predict":

            print("Run training!")
            parser = argparse.ArgumentParser(description='Prosit')
            parser.add_argument('-i', '--input', default=None, type=str, required=True,
                                help="Input data for prediction")

            parser.add_argument('-md', '--model_dir', default=None, type=str, required=True,
                                help="Model directory")

            parser.add_argument('-o', '--out_dir', default="./", type=str,
                                help="Output directory")

            parser.add_argument('-b', '--batch_size', default=64, type=int)
            parser.add_argument('-u', '--unit', default="s", type=str)

            if len(sys.argv) == 1:
                parser.print_usage()
                sys.exit(0)

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            input_file = args.input
            model_dir = args.model_dir
            out_dir = args.out_dir
            unit = args.unit

            batch_size = args.batch_size


            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            rt_predict(input_file, model_dir, batch_size=batch_size, out_dir=out_dir, prefix="test")



if __name__ == "__main__":
    main()

