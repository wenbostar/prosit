from . import utils

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("agg")
import os
import json



def get_array(tensor, keys):
    utils.check_mandatory_keys(tensor, keys)
    return [tensor[key] for key in keys]


def to_hdf5(dictionary, path):
    import h5py

    with h5py.File(path, "w") as f:
        for key, data in dictionary.items():
            f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")


def from_hdf5(path):
    import h5py

    with h5py.File(path, "r") as f:
        data = {k: f[k][...] for k in f.keys()}
    return data


letterDict = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    #"M(ox)": 21,
}

#letterDict_s = {integer: char for char, integer in letterDict.items()}


def add_mod(mod=None):

    i=max(letterDict.values())+1
    for m in mod:
        if str(m) not in letterDict:
            letterDict[str(m)] = i
            print("New aa: %s -> %d" % (str(m),i))
            i = i + 1

def load_aa(file):
    print("Load aa coding data from file %s" % (file))
    dat = pd.read_table(file, sep="\t", header=0, low_memory=False)
    letterDict.clear()
    for i, row in dat.iterrows():
        letterDict[row['aa']] = row['i']

def save_aa(file):
    print("Save aa coding data to file %s" % (file))
    with open(file,"w") as f:
        f.write("aa\ti\n")
        for aa in letterDict.keys():
            f.write(aa+"\t"+str(letterDict[aa])+"\n")

def peptideEncode(sequence, max_length=50):
    '''
    Encode peptide
    :param sequences: A list of peptides
    :return:
    '''
    array = np.zeros([max_length], dtype=int)
    #print(sequence)
    #print(letterDict)
    for i in range(len(sequence)):
        #print(sequence[i])
        array[i] = letterDict[sequence[i]]
    return array

## This is only used to process training data
def data_processing(input_data: str, test_file=None, mod=None, max_x_length = 50, min_rt=0, max_rt=120, unit="s",
                    out_dir="./", aa_file=None):

    res = dict()
    if aa_file is not None:
        ## read aa information from file
        load_aa(aa_file)
        res['aa'] = aa_file
    else:
        if mod is not None:
            add_mod(mod)

        aa2file = out_dir + "/aa.tsv"
        save_aa(aa2file)
        res['aa'] = aa2file

    ##
    siteData = pd.read_table(input_data, sep="\t", header=0, low_memory=False)

    ## x is peptide sequence and y is rt
    if "x" not in siteData.columns:
        siteData.columns = ['x','y']

    ## convert second to minute
    if unit.startswith("s"):
        siteData['y'] = siteData['y']/60.0

    ## get max rt
    if max_rt < siteData['y'].max():
        max_rt = siteData['y'].max() + 1.0

    ## get min rt
    if min_rt > siteData['y'].min():
        min_rt = siteData['y'].min() - 1.0

    # aaMap = getAAcodingMap()
    n_aa_types = len(letterDict)
    print("AA types: %d" % (n_aa_types))

    ## all aa in data
    all_aa = set()

    ## get the max length of input sequences

    longest_pep_training_data = 0
    for pep in siteData["x"]:
        if max_x_length < len(pep):
            max_x_length = len(pep)

        if longest_pep_training_data < len(pep):
            longest_pep_training_data = len(pep)

        ##
        for aa in pep:
            all_aa.add(aa)

    print("Longest peptide in training data: %d\n" % (longest_pep_training_data))

    ## test data
    test_data = None
    longest_pep_test_data = 0
    if test_file is not None:
        print("Use test file %s" % (test_file))
        test_data = pd.read_table(test_file, sep="\t", header=0, low_memory=False)
        if "x" not in test_data.columns:
            test_data.columns = ['x', 'y']
        if unit.startswith("s"):
            test_data['y'] = test_data['y'] / 60.0

        if max_rt < test_data['y'].max():
            max_rt = test_data['y'].max() + 1.0

        if min_rt > test_data['y'].min():
            min_rt = test_data['y'].min() - 1.0

        for pep in test_data["x"]:
            if max_x_length < len(pep):
                max_x_length = len(pep)

            if longest_pep_test_data < len(pep):
                longest_pep_test_data = len(pep)

            for aa in pep:
                all_aa.add(aa)

        print("Longest peptide in test data: %d\n" % (longest_pep_test_data))

    print(sorted(all_aa))

    siteData = siteData.sample(siteData.shape[0], replace=False, random_state=2018)

    #train_data = np.zeros((siteData.shape[0], max_x_length, n_aa_types))
    train_data = np.zeros((siteData.shape[0], max_x_length))
    k = 0
    for i, row in siteData.iterrows():
        peptide = row['x']
        # train_data[k] = encodePeptideOneHot(peptide, max_length=max_x_length)
        train_data[k] = peptideEncode(peptide, max_length=max_x_length)
        k = k + 1

    #train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2])

    X_test = np.empty(1)
    Y_test = np.empty(1)

    print("RT range: %d - %d\n" % (min_rt,max_rt))

    if test_data is None:
        X_train, X_test, Y_train, Y_test = train_test_split(train_data,
                                                            #to_categorical(pos_neg_all_data['y'], num_classes=2),
                                                            minMaxScale(siteData['y'],min_rt,max_rt),
                                                            test_size=0.1, random_state=100)
    else:
        X_train = train_data
        #Y_train = to_categorical(pos_neg_all_data['y'], num_classes=2)
        Y_train = siteData['y']
        Y_train = minMaxScale(Y_train, min_rt, max_rt)
        if len(Y_train.shape) >= 2:
            Y_train = Y_train.reshape(Y_train.shape[1])

        #X_test = np.zeros((test_data.shape[0], max_x_length, n_aa_types))
        X_test = np.zeros((test_data.shape[0], max_x_length))
        k = 0
        for i, row in test_data.iterrows():
            peptide = row['x']
            X_test[k] = peptideEncode(peptide, max_length=max_x_length)
            k = k + 1

        Y_test = minMaxScale(test_data['y'],min_rt,max_rt)


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    print("X_train shape:")
    print(X_train.shape)
    print("X_test shape:")
    print(X_test.shape)
    print("Modeling start ...")


    res['X_train'] = X_train
    res['Y_train'] = Y_train
    res['X_test'] = X_test
    res['Y_test'] = Y_test
    res['min_rt'] = min_rt
    res['max_rt'] = max_rt
    res['max_x_length'] = max_x_length

    #return [X_train, Y_train, X_test, Y_test, min_rt, max_rt]
    return res


def processing_prediction_data(model_file: str, input_data: str):
    '''

    :param model_file: model file in json format
    :param input_data: prediction file
    :return: A numpy matrix for prediction
    '''

    with open(model_file, "r") as read_file:
        model_list = json.load(read_file)

    model_folder = os.path.dirname(model_file)
    aa_file = model_folder + "/" + os.path.basename(model_list['aa'])
    load_aa(aa_file)

    ##
    siteData = pd.read_csv(input_data, sep="\t", header=0, low_memory=False)

    n_aa_types = len(letterDict)
    print("AA types: %d" % (n_aa_types))

    ## all aa in data
    all_aa = set()

    ## get the max length of input sequences
    max_x_length = model_list['max_x_length']

    longest_pep_len = 0
    for pep in siteData["x"]:
        #if max_x_length < len(pep):
        #    max_x_length = len(pep)
        if longest_pep_len < len(pep):
            longest_pep_len = len(pep)
        ##
        for aa in pep:
            all_aa.add(aa)

    print("Longest peptide in input data: %d\n" % (longest_pep_len))

    print(sorted(all_aa))

    # siteData = siteData.sample(siteData.shape[0], replace=False, random_state=2018)

    pred_data = np.zeros((siteData.shape[0], max_x_length))
    k = 0
    for i, row in siteData.iterrows():
        peptide = row['x']
        # train_data[k] = encodePeptideOneHot(peptide, max_length=max_x_length)
        pred_data[k] = peptideEncode(peptide, max_length=max_x_length)
        k = k + 1

    pred_data = pred_data.astype('float32')

    return pred_data



def minMaxScale(x, min=0,max=120):
    new_x = 1.0*(x-min)/(max-min)
    return new_x

def minMaxScoreRev(x,min=0,max=120):
    old_x = x * (max - min) + min
    return old_x



def encodePeptideOneHot(peptide: str, max_length=None):  # changed add one column for '1'

    AACategoryLen = len(letterDict)
    peptide_length = len(peptide)
    use_peptide = peptide
    if max_length is not None:
        if peptide_length < max_length:
            use_peptide = peptide + "X" * (max_length - peptide_length)

    en_vector = np.zeros((len(use_peptide), AACategoryLen))

    i = 0
    for AA in use_peptide:
        if AA in letterDict.keys():
            try:
                en_vector[i][letterDict[AA]] = 1
            except:
                print("peptide: %s, i => aa: %d, %s, %d" % (use_peptide,i, AA, letterDict[AA]))
                exit(1)
        else:
            en_vector[i] = np.full(AACategoryLen,1/AACategoryLen)

        i = i + 1

    return en_vector
