
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import csv
import linecache
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Add, Lambda
from tensorflow.keras.layers import Dropout, concatenate

from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from collections import namedtuple, OrderedDict

# In[2]:

batch_size = 12
epochs=30
iterations = 100

outputfolder = "D:/genes57_model_file/"
tasks = ['ADAS13_bl','AV45_bl', 'CDRSB_bl', 'FAQ_bl', 'FDG_bl', 'MMSE_bl', 'MOCA_bl']

gene_list = ["COL11A1", "FCER1G", "GBP2", "HSD11B1", "PARP1", "POU2F1", "NGF", "LHCGR", "LRP2", "APOD", "SST", "ALB", "COL25A1", "ADRB2", "ARSB", "FGF1", "FGF10", "FGF10-AS1", "FGF18", "NDUFS4", "PPP2R2B-IT1", "HSPA1A", "MICA", "MICAL1", "CASC14", "TBP", "TBPL1", "CAV1", "PON3", "RELN", "ADAM9", "NAT1", "NRG1", "DFNB31", "HSPA5", "POMT1", "RXRA", "TLR4", "CACNB2", "MINPP1", "TET1", "APOC3", "HBG2", "ATF7", "ATF7IP", "SLC11A2", "KLF5", "HNRNPC", "MTHFD1", "PNP", "APOC1", "APOE", "APOC1P1", "APOC2", "APOC4", "TOMM40"]

gene_folder = "D:/softwarepk/myjupyterfile/gblup/qiegene_ld0.9/txt_file/"
phenotype_file = 'D:/desktop/file/chromosome_one/normalized_phenotype_data_StandardScaler_463.csv'

file_path = 'D:/gains/group2.txt'

# In[3]:

def prepare_data(random_seed=None):
    y = pd.read_csv(phenotype_file)
    phe_ids = y.iloc[:, 0].astype(str).tolist()
    train_ids, temp_ids = train_test_split(phe_ids, test_size=0.2, random_state=random_seed)
    valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=random_seed)

    # train
    y_train = y.loc[y['PTID'].isin(train_ids)].drop(y.columns[[0]], axis=1)

    # valid
    y_valid = y.loc[y['PTID'].isin(valid_ids)].drop(y.columns[[0]], axis=1)

    # test
    y_test = y.loc[y['PTID'].isin(test_ids)].drop(y.columns[[0]], axis=1)

    return train_ids, valid_ids, test_ids, y_train, y_valid, y_test

# In[4]:

def load_data_and_models(seed):

    includegene=np.zeros(len(gene_list),dtype=int);
    modelsconcat=np.ndarray(sum(includegene==0),dtype=object);
    modelsinput=np.ndarray(sum(includegene==0),dtype=object);
    xtrain_array=np.ndarray(sum(includegene==0),dtype=object);
    
    xvalid_array=np.ndarray(sum(includegene==0),dtype=object);
    xtest_array=np.ndarray(sum(includegene==0),dtype=object);
    
    for index, gene in enumerate(gene_list):
        seed_folder = os.path.join(outputfolder, "seed" + str(seed))
        model_path = os.path.join(seed_folder, "result_model_" + gene + ".h5")
        model1 = keras.models.load_model(model_path)
        
        modelsconcat[index] = model1.get_layer(model1.layers[-2].name).output
        modelsinput[index]=model1.input;

        gene_file_path = os.path.join(gene_folder, gene + ".txt")
        x = pd.read_table(gene_file_path, sep=' ', engine='python')
        x_train = x.loc[x['IID'].isin(train_ids)]
        tmpdatax = x_train.drop(x_train.columns[[0, 1, 2, 3, 4, 5]], axis=1)
        xalltmp=tmpdatax.to_numpy()
        xtrain_array[index]=xalltmp;
        
        #valid
        x_valid = x.loc[x['IID'].isin(valid_ids)]
        tmpdatav = x_valid.drop(x_valid.columns[[0, 1, 2, 3, 4, 5]], axis=1)
        xalltmp2=tmpdatav.to_numpy()
        xvalid_array[index]=xalltmp2;
        #test
        x_test = x.loc[x['IID'].isin(test_ids)]
        tmpdatat = x_test.drop(x_test.columns[[0, 1, 2, 3, 4, 5]], axis=1)
        xalltmp3=tmpdatat.to_numpy()
        xtest_array[index]=xalltmp3; 

    return modelsconcat, modelsinput, xtrain_array, xvalid_array, xtest_array

# In[5]:

def gene_model_Multi():
    result_list = eval(line.strip())

    flattened_list = [item for sublist in result_list for item in sublist]

    y_train_new = y_train[result_list[0] + result_list[1]].to_numpy()
    y_valid_new = y_valid[result_list[0] + result_list[1]].to_numpy()
    y_test_new = y_test[result_list[0] + result_list[1]].to_numpy()

    tmp = modelsconcat.tolist()
    model_concat = concatenate(tmp, axis=-1)

    x1 = Dropout(0.3)(model_concat)
    x1 = Dense(128, activation='relu')(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(32, activation='relu')(x1)

    x_layers = []
    for i in range(len(result_list)):
        x_layers.append(Dense(32, activation='relu', name=f'x2_{i + 1}')(x1))

    outputs_list = []
    index = 1

    for i, feature_group in enumerate(result_list):
        group_size = len(feature_group)
        for _ in range(group_size):
            output = Dense(1, name=f'output_{index}')(x_layers[i])
            outputs_list.append(output)
            index += 1

    concatenated_outputs = concatenate(outputs_list, name='concatenated_outputs')
    tmp = modelsinput.tolist()
    model2 = Model(inputs=tmp, outputs=concatenated_outputs, name='gene_model_Multi')

    return model2, y_train_new, y_valid_new, y_test_new, flattened_list


# In[6]:

start_time = time.time()
output_pearson = "pearson_2groups.csv"

for a in range(1, 101): 
    seed = a
    line = linecache.getline(file_path, seed)
    result_list = eval(line.strip())    
    print(f"{seed}th group")
    np.random.seed(seed)
    train_ids, valid_ids, test_ids, y_train, y_valid, y_test= prepare_data(random_seed=seed)
    modelsconcat, modelsinput, xtrain_array, xvalid_array, xtest_array = load_data_and_models(seed=seed)

    model2, y_train_new, y_valid_new, y_test_new, flattened_list = gene_model_Multi()

    xtrain_all = xtrain_array.tolist()
    xvalid_all = xvalid_array.tolist()
    xtest_all = xtest_array.tolist()

    model_filename = 'model_seed2.hdf5'
    model2.compile(loss='mse', optimizer="adam")
    checkpointer = ModelCheckpoint(filepath=model_filename,
                                   monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce = ReduceLROnPlateau(monitor='val_loss', patience=7,
                               verbose=1,
                               factor=0.5,
                               min_lr=1e-6)

    history = model2.fit(xtrain_all, y_train_new,
                         validation_data=(xvalid_all, y_valid_new),
                         batch_size=batch_size,
                         callbacks=[checkpointer, reduce],
                         epochs=epochs, verbose=0)

    model3 = load_model(model_filename)
    # testing
    predictions_test = model3.predict(xtest_all)
    pearson_per_test = []
    mse_per_test = []

    for ii in range(len(tasks)):
        y_true_test_flat = y_test_new[:, ii].flatten()
        y_pred_test_flat = predictions_test[:, ii]

        pearson_test, _ = pearsonr(y_true_test_flat, y_pred_test_flat)
        pearson_per_test.append(pearson_test)

        mse_task = mean_squared_error(y_test_new[:, ii], y_pred_test_flat)
        mse_per_test.append(mse_task)

    with open(output_pearson, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(flattened_list)
        writer.writerow(pearson_per_test)

    tf.keras.backend.clear_session()

end_time = time.time() 
elapsed_time = end_time - start_time
print("All lines completed.")
print(f"Total time{elapsed_time}")




