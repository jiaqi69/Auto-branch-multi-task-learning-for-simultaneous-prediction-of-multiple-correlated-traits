
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import os
from sklearn.preprocessing import StandardScaler

seed = 9
np.random.seed(seed)
random.seed(seed)
n = 463

folder_path = 'D:/softwarepk/myjupyterfile/gblup/qiegene_ld0.9/txt_file/'
all_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

phenotype_file = 'D:/desktop/file/chromosome_one/normalized_phenotype_data_StandardScaler_463.csv'

phenotype_df = pd.read_csv(phenotype_file, index_col=0)

phenotype_index = phenotype_df.index.tolist()

tasks = ['ADAS13_bl','AV45_bl', 'CDRSB_bl', 'FAQ_bl', 'FDG_bl', 'MMSE_bl', 'MOCA_bl']


# In[2]:


for i in range(100):
    selected_files_list = []

    remaining_files = all_files.copy()
    
    for _ in range(7):
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
    
    y_err_list = [np.random.normal(0, 7, n) for _ in range(7)]
    beta_list = [np.random.normal(0, 1, x.shape[1]) for x in x_list]
    y_list = []
    
    
    for a, x in enumerate(x_list):
        y = np.dot(x, beta_list[a]) + y_err_list[a]
        y_list.append(y)

    merged_data = np.column_stack(y_list)
    
    y = pd.DataFrame(merged_data)

    y.rename(columns=dict(zip(y.columns, tasks)), inplace=True)

    y1 = (y - y.mean()) / y.std()

    y1.index = phenotype_index    

    y1.insert(0, "PTID", y1.index)

    new_file_path = f"D:\\y_notshare_{i}.csv"
    y1.to_csv(new_file_path, index=False)



