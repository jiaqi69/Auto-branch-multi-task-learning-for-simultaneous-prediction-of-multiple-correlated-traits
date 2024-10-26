
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


# Two-group Sharing
for i in range(100):

    selected_files1 = random.sample(all_files, round(56*0.1))
    selected_files2 = random.sample([file for file in all_files if file not in selected_files1],round(56*0.1))
    
    x1 = None
    x2 = None
          
    for file_name1 in selected_files1:
        file_path1 = os.path.join(folder_path, file_name1)
        
        df1 = pd.read_csv(file_path1, delim_whitespace=True)
        df1 = df1.iloc[:, 6:]

        x = df1.values

        index1 = random.sample(range(0,x.shape[1]),round(x.shape[1] * 0.1))
        data1 = x[:,index1]
        if x1 is None:
            x1 = data1
        else:
            x1 = np.concatenate((x1, data1), axis=1)
            
    for file_name2 in selected_files2:
        file_path2 = os.path.join(folder_path, file_name2)
        
        df2 = pd.read_csv(file_path2, delim_whitespace=True)
        df2 = df2.iloc[:, 6:]

        xx = df2.values

        index2 = random.sample(range(0,xx.shape[1]),round(xx.shape[1] * 0.1))
        data2 = xx[:,index2]
        if x2 is None:
            x2 = data2
        else:
            x2 = np.concatenate((x2, data2), axis=1)

    y_err1=np.random.normal(0,7,(n))
    y_err2=np.random.normal(0,7,(n))
    y_err3=np.random.normal(0,7,(n))    
    y_err4=np.random.normal(0,7,(n))
    y_err5=np.random.normal(0,7,(n))
    y_err6=np.random.normal(0,7,(n))
    y_err7=np.random.normal(0,7,(n))

    beta1=np.random.normal(0,1,(x1.shape[1]))
    beta2=np.random.normal(0,1,(x2.shape[1]))    

    y1 = np.dot(x1,beta1)+y_err1
    y2 = np.dot(x1,beta1)+y_err2
    y3 = np.dot(x1,beta1)+y_err3
    
    y4 = np.dot(x2,beta2)+y_err4 
    y5 = np.dot(x2,beta2)+y_err5    
    y6 = np.dot(x2,beta2)+y_err6
    y7 = np.dot(x2,beta2)+y_err7    

    merged_data = np.column_stack((y1, y2, y3, y4, y5, y6, y7))
    y = pd.DataFrame(merged_data)
    
    y.rename(columns=dict(zip(y.columns, tasks)), inplace=True)
    y1 = (y - y.mean()) / y.std()
    y1.index = phenotype_index
    
    y1.insert(0, "PTID", y1.index)    

    new_file_path = f"D:/y_2group_{i}.csv"
    y1.to_csv(new_file_path, index=False)

# Three-group Sharing
for i in range(100):
    
    selected_files1 = random.sample(all_files, round(56*0.1))
    selected_files2 = random.sample([file for file in all_files if file not in selected_files1],round(56*0.1))
    selected_files3 = random.sample([file for file in all_files if file not in selected_files1 and file not in selected_files2],round(56*0.1))
    x1 = None
    x2 = None
    x3 = None
          
    for file_name1 in selected_files1:
        file_path1 = os.path.join(folder_path, file_name1)
        
        df1 = pd.read_csv(file_path1, delim_whitespace=True)
        df1 = df1.iloc[:, 6:]

        x = df1.values

        index1 = random.sample(range(0,x.shape[1]),round(x.shape[1] * 0.1))
        data1 = x[:,index1]
        if x1 is None:
            x1 = data1
        else:
            x1 = np.concatenate((x1, data1), axis=1)
            
    for file_name2 in selected_files2:
        file_path2 = os.path.join(folder_path, file_name2)
        
        df2 = pd.read_csv(file_path2, delim_whitespace=True)
        df2 = df2.iloc[:, 6:]

        xx = df2.values

        index2 = random.sample(range(0,xx.shape[1]),round(xx.shape[1] * 0.1))
        data2 = xx[:,index2]
        if x2 is None:
            x2 = data2
        else:
            x2 = np.concatenate((x2, data2), axis=1)
            
    for file_name3 in selected_files3:
        file_path3 = os.path.join(folder_path, file_name3)
        
        df3 = pd.read_csv(file_path3, delim_whitespace=True)
        df3 = df3.iloc[:, 6:]

        xxx = df3.values

        index3 = random.sample(range(0,xxx.shape[1]),round(xxx.shape[1] * 0.1))
        data3 = xxx[:,index3]
        if x3 is None:
            x3 = data3
        else:
            x3 = np.concatenate((x3, data3), axis=1)

    y_err1=np.random.normal(0,7,(n))
    y_err2=np.random.normal(0,7,(n))
    y_err3=np.random.normal(0,7,(n))    
    y_err4=np.random.normal(0,7,(n))
    y_err5=np.random.normal(0,7,(n))
    y_err6=np.random.normal(0,7,(n))
    y_err7=np.random.normal(0,7,(n))

    beta1=np.random.normal(0,1,(x1.shape[1]))
    beta2=np.random.normal(0,1,(x2.shape[1]))
    beta3=np.random.normal(0,1,(x3.shape[1]))    

    y1 = np.dot(x1,beta1)+y_err1
    y2 = np.dot(x1,beta1)+y_err2
    y3 = np.dot(x1,beta1)+y_err3 
    
    y4 = np.dot(x2,beta2)+y_err4 
    y5 = np.dot(x2,beta2)+y_err5
    
    y6 = np.dot(x3,beta3)+y_err6
    y7 = np.dot(x3,beta3)+y_err7    

    merged_data = np.column_stack((y1, y2, y3, y4, y5, y6, y7))
    y = pd.DataFrame(merged_data)
    
    y.rename(columns=dict(zip(y.columns, tasks)), inplace=True)
    y1 = (y - y.mean()) / y.std()

    y1.index = phenotype_index
    
    y1.insert(0, "PTID", y1.index)    

    new_file_path = f"D:/y_3group_{i}.csv"
    y1.to_csv(new_file_path, index=False)

# Four-group Sharing
for i in range(100):
    selected_files1 = random.sample(all_files, round(56*0.1))
    remaining_files = [file for file in all_files if file not in selected_files1]
    selected_files2 = random.sample(remaining_files, round(56*0.1))
    remaining_files = [file for file in remaining_files if file not in selected_files2]
    selected_files3 = random.sample(remaining_files, round(56*0.1))
    remaining_files = [file for file in remaining_files if file not in selected_files3]
    selected_files4 = random.sample(remaining_files, round(56*0.1))
    
    x1, x2, x3, x4 = None, None, None, None
    
    for file_name1 in selected_files1:
        file_path1 = os.path.join(folder_path, file_name1)
        df1 = pd.read_csv(file_path1, delim_whitespace=True)
        df1 = df1.iloc[:, 6:]
        x = df1.values
        index1 = random.sample(range(0, x.shape[1]), round(x.shape[1] * 0.1))
        data1 = x[:, index1]
        if x1 is None:
            x1 = data1
        else:
            x1 = np.concatenate((x1, data1), axis=1)
            
    for file_name2 in selected_files2:
        file_path2 = os.path.join(folder_path, file_name2)
        df2 = pd.read_csv(file_path2, delim_whitespace=True)
        df2 = df2.iloc[:, 6:]
        xx = df2.values
        index2 = random.sample(range(0, xx.shape[1]), round(xx.shape[1] * 0.1))
        data2 = xx[:, index2]
        if x2 is None:
            x2 = data2
        else:
            x2 = np.concatenate((x2, data2), axis=1)
            
    for file_name3 in selected_files3:
        file_path3 = os.path.join(folder_path, file_name3)
        df3 = pd.read_csv(file_path3, delim_whitespace=True)
        df3 = df3.iloc[:, 6:]
        xxx = df3.values
        index3 = random.sample(range(0, xxx.shape[1]), round(xxx.shape[1] * 0.1))
        data3 = xxx[:, index3]
        if x3 is None:
            x3 = data3
        else:
            x3 = np.concatenate((x3, data3), axis=1)
            
    for file_name4 in selected_files4:
        file_path4 = os.path.join(folder_path, file_name4)
        df4 = pd.read_csv(file_path4, delim_whitespace=True)
        df4 = df4.iloc[:, 6:]
        xxxx = df4.values
        index4 = random.sample(range(0, xxxx.shape[1]), round(xxxx.shape[1] * 0.1))
        data4 = xxxx[:, index4]
        if x4 is None:
            x4 = data4
        else:
            x4 = np.concatenate((x4, data4), axis=1)
    
    y_err1 = np.random.normal(0, 7, (n))
    y_err2 = np.random.normal(0, 7, (n))
    y_err3 = np.random.normal(0, 7, (n))
    y_err4 = np.random.normal(0, 7, (n))
    y_err5 = np.random.normal(0, 7, (n))
    y_err6 = np.random.normal(0, 7, (n))
    y_err7 = np.random.normal(0, 7, (n))

    beta1 = np.random.normal(0, 1, (x1.shape[1]))
    beta2 = np.random.normal(0, 1, (x2.shape[1]))
    beta3 = np.random.normal(0, 1, (x3.shape[1]))
    beta4 = np.random.normal(0, 1, (x4.shape[1]))

    y1 = np.dot(x1, beta1) + y_err1
    y2 = np.dot(x1, beta1) + y_err2
    
    y3 = np.dot(x2, beta2) + y_err3
    y4 = np.dot(x2, beta2) + y_err4
    
    y5 = np.dot(x3, beta3) + y_err5
    y6 = np.dot(x3, beta3) + y_err6
    
    y7 = np.dot(x4, beta4) + y_err7

    merged_data = np.column_stack((y1, y2, y3, y4, y5, y6, y7))
    y = pd.DataFrame(merged_data)

    y.rename(columns=dict(zip(y.columns, tasks)), inplace=True)
    y1 = (y - y.mean()) / y.std()
    y1.index = phenotype_index
    y1.insert(0, "PTID", y1.index)
    
    new_file_path = f"D:/y_4group_{i}.csv"
    y1.to_csv(new_file_path, index=False)
    




