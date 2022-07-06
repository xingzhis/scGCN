import os
import numpy as np
import random
import pandas as pd
import time as tm
from operator import itemgetter
from sklearn.model_selection import train_test_split
import pickle as pkl
import scipy.sparse
from graph import *

#' data preperation
def input_data(DataDir,Rgraph=True,test_on_train=True):
    # TODO this `test_on_train` is just a temperary setting, should make explicit interface for train and test dataset input!
    if Rgraph==False:
        graph_construct(outputdir='process_data')

    DataPath1 = '{}/Data1.csv'.format(DataDir)
    DataPath2 = '{}/Data2.csv'.format(DataDir)
    LabelsPath1 = '{}/Label1.csv'.format(DataDir)
    LabelsPath2 = '{}/Label2.csv'.format(DataDir)

    #' read the data
    data1 = pd.read_csv(DataPath1, index_col=0, sep=',')
    data2 = pd.read_csv(DataPath2, index_col=0, sep=',')
    lab_label1 = pd.read_csv(LabelsPath1, header=0, index_col=False, sep=',')
    lab_label2 = pd.read_csv(LabelsPath2, header=0, index_col=False, sep=',')

    lab_data1 = data1.reset_index(drop=True)  #.transpose()
    lab_data2 = data2.reset_index(drop=True)  #.transpose()
    lab_label1.columns = ['type']
    lab_label2.columns = ['type']

    types = np.unique(lab_label1['type']).tolist()

    random.seed(123)
    p_data = []
    p_label = []
    for i in types:
        tem_index = lab_label1[lab_label1['type'] == i].index
        tem_label = lab_label1[lab_label1['type'] == i]
        tem_data = lab_data1.iloc[tem_index]
        num_to_select = len(tem_data)
        random_items = random.sample(range(0, len(tem_index)), num_to_select)
        # print(random_items)
        sub_data = tem_data.iloc[random_items]
        sub_label = tem_label.iloc[random_items]
        # print((sub_data.index == sub_label.index).all())
        p_data.append(sub_data)
        p_label.append(sub_label)

    #' split data to training, test, valdiaton sets

    data_train = []
    data_test = []
    data_val = []
    label_train = []
    label_test = []
    label_val = []

    for i in range(0, len(p_data)):
        if test_on_train:
            temD_train, temL_train = p_data[i], p_label[i]
            temd_train, temd_val, teml_train, teml_val = train_test_split(
                temD_train, temL_train, test_size=0.1, random_state=1)
            temd_test, teml_test = p_data[i], p_label[i]
        else:
            temD_train, temd_test, temL_train, teml_test = train_test_split(
                p_data[i], p_label[i], test_size=0.1, random_state=1)
            temd_train, temd_val, teml_train, teml_val = train_test_split(
                temD_train, temL_train, test_size=0.1, random_state=1)
        print(temd_train.shape, teml_train.shape, temd_val.shape, teml_val.shape, temd_test.shape, teml_test.shape)
        print((temd_train.index == teml_train.index).all())
        print((temd_test.index == teml_test.index).all())
        print((temd_val.index == teml_val.index).all())
        data_train.append(temd_train)
        label_train.append(teml_train)
        data_test.append(temd_test)
        label_test.append(teml_test)
        data_val.append(temd_val)
        label_val.append(teml_val)

    data_train1 = pd.concat(data_train)
    data_test1 = pd.concat(data_test)
    data_val1 = pd.concat(data_val)
    label_train1 = pd.concat(label_train)
    label_test1 = pd.concat(label_test)
    label_val1 = pd.concat(label_val)


    # TODO this `test_on_train` is just a temperary setting, should make explicit interface for train and test dataset input!
    if test_on_train:
        # sort index to match the raw data for comparison
        data_test1.sort_index(inplace=True)
        label_test1.sort_index(inplace=True)
        # change index to prevent repeat with the train data, or would cause an extra match in utils.py:171.
        ttl_num_dat = data_test1.shape[0]
        new_id = data_test1.index + ttl_num_dat
        data_test1.index = new_id
        label_test1.index = new_id

    train2 = pd.concat([data_train1, lab_data2])
    lab_train2 = pd.concat([label_train1, lab_label2])

    #' save objects

    PIK = "{}/datasets.dat".format(DataDir)
    res = [
        data_train1, data_test1, data_val1, label_train1, label_test1,
        label_val1, lab_data2, lab_label2, types
    ]

    with open(PIK, "wb") as f:
        pkl.dump(res, f)

    print('load data succesfully....')
