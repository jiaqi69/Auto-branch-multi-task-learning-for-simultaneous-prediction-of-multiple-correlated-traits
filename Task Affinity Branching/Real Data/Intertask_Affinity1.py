
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import csv
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

Params = namedtuple("Params", ['lr', 'alpha']) 
params = Params(lr=0.001, alpha=0.1)
order = 1


# In[3]:

def prepare_data(random_seed=None):
    y = pd.read_csv(phenotype_file)
    phe_ids = y.iloc[:, 0].astype(str).tolist()
    train_ids, temp_ids = train_test_split(phe_ids, test_size=0.2, random_state=random_seed)
    valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=random_seed)

    # train
    y_train = y.loc[y['PTID'].isin(train_ids)].drop(y.columns[[0]], axis=1).to_numpy()
    labels_train = {task: y_train[:, b] for b, task in enumerate(tasks)}

    # valid
    y_valid = y.loc[y['PTID'].isin(valid_ids)].drop(y.columns[[0]], axis=1).to_numpy()
    labels_valid = {task: y_valid[:, b] for b, task in enumerate(tasks)}

    # test
    y_test = y.loc[y['PTID'].isin(test_ids)].drop(y.columns[[0]], axis=1).to_numpy()
    labels_test = {task: y_test[:, b] for b, task in enumerate(tasks)}

    return labels_train, labels_valid, labels_test, train_ids, valid_ids, test_ids, y_train

# In[4]:
def genes_models(genes_epoch, seed):
    seed_folder = os.path.join(outputfolder, "seed" + str(seed))
    os.makedirs(seed_folder, exist_ok=True)

    for file_name in os.listdir(gene_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(gene_folder, file_name)

            x = pd.read_table(file_path, sep=' ', engine='python')
            x_train = x.loc[x['IID'].isin(train_ids)]
            tmpdatax = x_train.drop(x_train.columns[[0, 1, 2, 3, 4, 5]], axis=1)
            x_train = tmpdatax.to_numpy()

            geneindex = os.path.splitext(os.path.basename(file_path))[0]

            if x_train.shape[1] > 32:
                nunit1 = 16

            elif x_train.shape[1] < 11:
                nunit1 = x_train.shape[1]

            else:
                nunit1 = x_train.shape[1] // 2
            inputs = Input(shape=(x_train.shape[1],), name=geneindex + "_input")

            if x_train.shape[1] > 100:
                x = Dropout(0.5, name=geneindex + "l0drop")(inputs)
                x = Dense(units=nunit1, activation='relu', name=geneindex + "l0dense")(x)
            else:
                x = Dense(units=nunit1, activation='relu', name=geneindex + "l0dense")(inputs)

            outputs = Dense(7, name=geneindex + "l2")(x)

            model = Model(inputs=inputs, outputs=outputs)

            output_file_model = os.path.join(seed_folder, 'result_model_' + geneindex + '.h5')
            model.save(output_file_model)
            del model


def permute(losses):
    """Returns all combinations of losses in the loss dictionary."""
    losses = OrderedDict(sorted(losses.items()))
    rtn = {} 
    for task,loss in losses.items():
        tmp_dict = {task:loss}
        for saved_task, saved_loss in rtn.items():
            if order == 1: 
                continue 
            new_task = "{}_{}".format(saved_task, task)
            new_loss = loss + saved_loss 
            tmp_dict[new_task] = new_loss 
        rtn.update(tmp_dict)
  
    if order == 1:
        rtn["_".join(losses.keys())] = sum(losses.values())
    return rtn

def permute_list(lst):
    """Returns all combinations of tasks in the task list."""
    lst.sort() 
    rtn = []
    for task in lst:
        tmp_lst = [task]
        for saved_task in rtn:
            if order == 1:
                continue
            new_task = "{}_{}".format(saved_task, task)
            tmp_lst.append(new_task)
        rtn += tmp_lst
  
    if order == 1:
        rtn.append("_".join(lst))
    return rtn


# In[5]:

class AttributeDecoder(tf.keras.Model):
    def __init__(self, name=None):
        super(AttributeDecoder, self).__init__(name=name)
        self.fc = Dense(1)

    def call(self, inputs):
        x = self.fc(inputs)
        return x

    def get_config(self):
        return {'name': self.fc.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# In[6]:

def base_step(inputs, base_updated):

    updated_weights = [param.numpy() for param in base_updated]
    gene_layers = []
    
    for i, gene in enumerate(gene_list):
        gene_input = inputs[i]

        if gene_input.shape[1] > 32:
            nunit1 = 16
            
        elif gene_input.shape[1] < 11:    
            nunit1 = gene_input.shape[1]
            
        else:
            nunit1 = gene_input.shape[1] // 2  
            
        if gene_input.shape[1] > 100:
            gene_drop = Dropout(0.5)(gene_input)
            gene_dense = tf.keras.layers.Dense(nunit1, activation='relu', use_bias=True,
                                               kernel_initializer=tf.constant_initializer(updated_weights[2*i]),
                                               bias_initializer=tf.constant_initializer(updated_weights[2*i + 1]))(gene_drop)
        
        else:
            gene_dense = tf.keras.layers.Dense(nunit1, activation='relu', use_bias=True,
                                               kernel_initializer=tf.constant_initializer(updated_weights[2*i]),
                                               bias_initializer=tf.constant_initializer(updated_weights[2*i + 1]))(gene_input)

        gene_layers.append(gene_dense)

    concatenate = tf.keras.layers.Concatenate(axis=-1)(gene_layers)


    final_drop = tf.keras.layers.Dropout(0.3)(concatenate)
    final_layer1 = tf.keras.layers.Dense(128, activation='relu', use_bias=True,
                                          kernel_initializer=tf.constant_initializer(updated_weights[-4]),
                                          bias_initializer=tf.constant_initializer(updated_weights[-3]))(final_drop)
    final_drop2 = tf.keras.layers.Dropout(0.2)(final_layer1)

    final_layer2 = tf.keras.layers.Dense(32, activation='relu', use_bias=True,
                                        kernel_initializer=tf.constant_initializer(updated_weights[-2]),
                                        bias_initializer=tf.constant_initializer(updated_weights[-1]))(final_drop2)

    return final_layer2


# In[7]:

optimizer = tf.keras.optimizers.Adam()
global_step = tf.Variable(0, trainable=False)

def train_task_gains(input, labels, first_step=False):
    """This is TAG."""
    with tf.GradientTape(persistent=True) as tape:
        preds = model2(input, training=True)
        losses = {task: tf.reduce_mean(tf.square(tf.cast(preds[task], dtype=tf.float32) -
                                                 tf.cast(labels[task], dtype=tf.float32))) for task in labels}
        task_gains = {}
        task_permutations = permute(losses)
        combined_task_gradients = [
            (combined_task, tape.gradient(task_permutations[combined_task], ResBase.trainable_weights))
            for combined_task in task_permutations]

    for combined_task, task_gradient in combined_task_gradients:
        if first_step:
            base_update = [optimizer.lr * grad for grad in task_gradient]
            base_updated = [param - update for param, update in zip(ResBase.trainable_weights, base_update)]
        else:
            base_update = [(optimizer._get_hyper('beta_1') * optimizer.get_slot(param, 'm') - optimizer.lr * grad)
                           for param, grad in zip(ResBase.trainable_weights, task_gradient)]
            base_updated = [param + update for param, update in zip(ResBase.trainable_weights, base_update)]

        task_update_rep = base_step(input, base_updated)

        task_update_preds = {task: model(task_update_rep, training=True) for (task, model) in ResTowers.items()}
        task_update_losses = {task: tf.reduce_mean(tf.square(tf.cast(task_update_preds[task], dtype=tf.float32) -
                                                             tf.cast(labels[task], dtype=tf.float32))) for task in
                              labels}
        task_gain = {task: (1.0 - task_update_losses[task] / losses[task]) / optimizer.lr for task in labels}
        task_gains[combined_task] = task_gain

    global_step.assign_add(1)
    return task_gains


# In[8]:

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

# In[9]:

def build_final_model():
    
    tmp=modelsconcat.tolist()
    model_concat = concatenate(tmp, axis=-1)

    model_concat = Dropout(0.3, name="final_drop")(model_concat)
    model_concat = Dense(128, activation="relu", name="final_layer1")(model_concat)
    model_concat = Dropout(0.2, name="final_drop2")(model_concat)
    model_concat = Dense(32, activation="relu", name="final_layer2")(model_concat)

    tmp=modelsinput.tolist()
    ResBase = Model(inputs=tmp, outputs=model_concat, name="final")
    
    ResTowers = {task: AttributeDecoder(name=task) for task in tasks}
    preds = {task: tower(model_concat, training=True) for task, tower in ResTowers.items()}
    model2 = tf.keras.models.Model(inputs=tmp, outputs=preds)
    
    return model2, ResBase, ResTowers

# In[10]:

start_time = time.time()
save_path = 'D:/gains/'
pearson_all_val = []
best_pearson_all_val = []
pearson_all_test = []
mse_all_test = []

output_pearson_csv = "pearson.csv"
output_mse_csv = "mse.csv"

with open(output_pearson_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration'] + tasks)

with open(output_mse_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration'] + tasks)

for i in range(iterations):
    seed = i + 1
    np.random.seed(seed)
    
    labels_train, labels_valid, labels_test, train_ids, valid_ids, test_ids, y_train= prepare_data(i)
    
    genes_models(genes_epoch=100, seed=seed)
    
    modelsconcat, modelsinput, xtrain_array, xvalid_array, xtest_array = load_data_and_models(seed=seed)

    model2, ResBase, ResTowers = build_final_model()
    
    xtrain_all=xtrain_array.tolist()
    xvalid_all=xvalid_array.tolist()
    xtest_all=xtest_array.tolist()

    # Compile model
    model2.compile(loss='mean_squared_error', optimizer="adam")
    checkpointer = ModelCheckpoint(filepath='best_model.hdf5',
                            monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce = ReduceLROnPlateau(monitor='val_loss', patience=7,
                            verbose=1,
                            factor=0.5,
                            min_lr=1e-6)  

    gradient_metrics = {task:[] for task in permute_list(tasks)}
    
    best_val_loss = float('inf')
    best_pearson_per_val = None
    
    val_loss_history = []
    pearson_per_val_history = []

    for epoch in range(epochs):
        history = model2.fit(xtrain_all, labels_train, 
                             epochs=1,
                             batch_size=batch_size,
                             verbose=1)
        
        task_gains = train_task_gains(xtrain_all, labels_train, first_step=(len(optimizer.variables()) == 0)) 

        batch_grad_metrics = {combined_task:{task:0. for task in tasks} for combined_task in gradient_metrics}

        for combined_task,task_gain_map in task_gains.items():
            for task,gain in task_gain_map.items():
                batch_grad_metrics[combined_task][task] = gain.numpy()

        for combined_task2,task_gain_map2 in batch_grad_metrics.items():
            gradient_metrics[combined_task2].append(task_gain_map2)
            
        for task in gradient_metrics:
            gradient_metrics[task] = gradient_metrics[task][:epochs]
  

        val_metrics = model2.evaluate(xvalid_all, labels_valid,
                                      callbacks=[checkpointer, reduce],
                                      batch_size=batch_size)
        val_loss = val_metrics[0]
        
        # 预测验证集  
        predictions = model2.predict(xvalid_all, batch_size=batch_size) 
        task_predictions = {task: predictions[task][:, 0] for task in tasks}  
        pearson_per_val = {task: np.corrcoef(task_predictions[task], labels_valid[task])[0, 1] for task in tasks}
        print(pearson_per_val)
        
        val_loss_history.append(val_loss)
        pearson_per_val_history.append(pearson_per_val)

        checkpointer.on_epoch_end(epoch, logs={'val_loss': val_loss})
        reduce.on_epoch_end(epoch, logs={'val_loss': val_loss})

    min_val_loss_index = val_loss_history.index(min(val_loss_history))
    best_pearson_per_val = pearson_per_val_history[min_val_loss_index]
    
    print("Best Pearson per validation loss:", best_pearson_per_val)        
    best_pearson_all_val.append(best_pearson_per_val)    

    model3 = load_model('best_model.hdf5', custom_objects={'AttributeDecoder': AttributeDecoder})


    filename = f'{save_path}gains_{seed}.txt'    
    with open(filename, 'w') as file:
        for key, value in gradient_metrics.items():
            file.write(f'{key}:\n')
            for item in value:
                file.write(str(item) + '\n')
            file.write('\n')  

    pearson_all_val.append(pearson_per_val)

    #testing
    predictions_test = model3.predict(xtest_all)

    task_predictions_test = {task: predictions_test[task][:, 0] for task in tasks}

    pearson_per_test = {task: np.corrcoef(task_predictions_test[task], labels_test[task])[0, 1] for task in tasks}
    mse_per_test = {task: mean_squared_error(labels_test[task], task_predictions_test[task]) for task in tasks}

    print('Test Pearson Correlation:',pearson_per_test)
    pearson_all_test.append(pearson_per_test)
    print('Test MSE:', mse_per_test)
    mse_all_test.append(mse_per_test)

    # pearson
    with open(output_pearson_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i+1] + [pearson_per_test[task] for task in tasks])
    # mse
    with open(output_mse_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i+1] + [mse_per_test[task] for task in tasks])


    print(i)
    print(f"Iteration {seed} completed.")
    tf.keras.backend.clear_session()

    
end_time = time.time()
elapsed_time = end_time - start_time
print("All iterations completed.")
print(f"Total time{elapsed_time}")

