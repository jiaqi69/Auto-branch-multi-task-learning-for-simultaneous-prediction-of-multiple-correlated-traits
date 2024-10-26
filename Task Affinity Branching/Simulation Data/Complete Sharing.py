
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


for i in range(100):

    selected_files = random.sample(all_files, round(56*0.1))
    x1 = None
    for file_name in selected_files:
        file_path = os.path.join(folder_path, file_name)
        
        df = pd.read_csv(file_path, delim_whitespace=True)

        df = df.iloc[:, 6:]
        

        x = df.values

        index1 = random.sample(range(0,x.shape[1]),round(x.shape[1] * 0.1))
        data = x[:,index1]
        if x1 is None:
            x1 = data
        else:
            x1 = np.concatenate((x1, data), axis=1)


    y_err1=np.random.normal(0,7,(n))
    y_err2=np.random.normal(0,7,(n))
    y_err3=np.random.normal(0,7,(n))    
    y_err4=np.random.normal(0,7,(n))
    y_err5=np.random.normal(0,7,(n))
    y_err6=np.random.normal(0,7,(n))
    y_err7=np.random.normal(0,7,(n))

    beta1=np.random.normal(0,1,(x1.shape[1]))

    y1 = np.dot(x1,beta1)+y_err1
    y2 = np.dot(x1,beta1)+y_err2
    y3 = np.dot(x1,beta1)+y_err3    
    y4 = np.dot(x1,beta1)+y_err4    
    y5 = np.dot(x1,beta1)+y_err5    
    y6 = np.dot(x1,beta1)+y_err6
    y7 = np.dot(x1,beta1)+y_err7    
    

    merged_data = np.column_stack((y1, y2, y3, y4, y5, y6, y7))
    y = pd.DataFrame(merged_data)

    
    y.rename(columns=dict(zip(y.columns, tasks)), inplace=True)

    y1 = (y - y.mean()) / y.std()
    

    y1.index = phenotype_index
    
    y1.insert(0, "PTID", y1.index)    
    

    new_file_path = f"D:\\y_{i}.csv"
    y1.to_csv(new_file_path, index=False)
    



