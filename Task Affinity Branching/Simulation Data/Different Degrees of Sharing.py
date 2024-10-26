
# coding: utf-8

import numpy as np
import pandas as pd
import random
import os
from sklearn.preprocessing import StandardScaler

seed = 35735
np.random.seed(seed)
random.seed(seed)
n = 463

folder_path = 'D:/softwarepk/myjupyterfile/gblup/qiegene_ld0.9/txt_file/'
all_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

phenotype_file = 'D:/desktop/file/chromosome_one/normalized_phenotype_data_StandardScaler_463.csv'

phenotype_df = pd.read_csv(phenotype_file, index_col=0)

phenotype_index = phenotype_df.index.tolist()

tasks = ['ADAS13_bl','AV45_bl', 'CDRSB_bl', 'FAQ_bl', 'FDG_bl', 'MMSE_bl', 'MOCA_bl']


for i in range(100):
    selected_files_list = []
    remaining_files = all_files.copy()
    
    for _ in range(8):
        if len(remaining_files) <= round(56 * 0.1):
            selected_files_list.append(remaining_files)
            break

        selected_files = random.sample(remaining_files, round(56 * 0.1))
        selected_files_list.append(selected_files)

        remaining_files = [file for file in remaining_files if file not in selected_files]
    
    x_list = []
    
    for selected_files in selected_files_list:
        x = None
        for file_name in selected_files:
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, delim_whitespace=True)
            df = df.iloc[:, 6:]
            x_file = df.values
            index = random.sample(range(0, x_file.shape[1]), round(x_file.shape[1] * 0.1))
            data = x_file[:, index]
            if x is None:
                x = data
            else:
                x = np.concatenate((x, data), axis=1)
        x_list.append(x)

    y_err_list = [np.random.normal(0, np.sqrt(490), n) for _ in range(7)]
    beta_list = [np.random.normal(0, np.sqrt(9), x.shape[1]) for x in x_list[:7]]
    beta8 = np.random.normal(0, np.sqrt(1), x_list[7].shape[1])

    
    x1, x2, x3, x4, x5, x6, x7, x8= x_list
    y_err1, y_err2, y_err3, y_err4, y_err5, y_err6, y_err7 = y_err_list
    beta1, beta2, beta3, beta4, beta5, beta6, beta7 = beta_list 

    y1 = np.dot(x1,beta1)+np.dot(x8,beta8)+y_err1
    y2 = np.dot(x2,beta2)+np.dot(x8,beta8)+y_err2
    y3 = np.dot(x3,beta3)+np.dot(x8,beta8)+y_err3    
    y4 = np.dot(x4,beta4)+np.dot(x8,beta8)+y_err4
    y5 = np.dot(x5,beta5)+np.dot(x8,beta8)+y_err5    
    y6 = np.dot(x6,beta6)+np.dot(x8,beta8)+y_err6    
    y7 = np.dot(x7,beta7)+np.dot(x8,beta8)+y_err7    

    merged_data = np.column_stack((y1, y2, y3, y4, y5, y6, y7))
    y = pd.DataFrame(merged_data)

    y.rename(columns=dict(zip(y.columns, tasks)), inplace=True)
    y1 = (y - y.mean()) / y.std()
    y1.index = phenotype_index    

    y1.insert(0, "PTID", y1.index)    

    new_file_path = f"D:/y_not9_share1_{i}.csv"
    y1.to_csv(new_file_path, index=False)
