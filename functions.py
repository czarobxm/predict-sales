import re
import pandas as pd

def delete_letters(df, column_name = 'size'):
    # This function deletes letters from each cell in given column and changes its type from string into integer
    tab = []
    for i in range(len(df[column_name].value_counts().index)):
        tab.append(int(re.sub("[^0-9]","",df[column_name].value_counts().index[i])))

    indices = df[column_name].value_counts().index

    dict_size = {}
    for i in range(len(tab)):
        dict_size[indices[i]] = tab[i]
        
    if column_name == 'size':
        dict_size['2x68g'] = 136

    df[column_name] = df[column_name].replace(dict_size)

def cat_into_num(df):
    # changing categorical columns into numerical columns
    index = []

    # save indexes of columns with dtype == object
    for i in range(len(df.dtypes)):
        if df.dtypes[i] == 'O':
            index.append(i)

    # if every object column change its categorical values to numerical values
    for i in index:
        df.iloc[:,i] = pd.factorize(df.iloc[:,i])[0]


import math
import random

def split(df , train_size= 0.8, validation_size =0.2, test_size =0):
    # This function devides given DataFrame into tr_set, val_set and te_set with given sizes of these sets

    # sanity check
    if train_size + validation_size + test_size != 1:
        raise ValueError("Sum of train_size, validation_size and trst_size must be equal to 1")
    
    df_size = df.shape[0]
    
    # shuffling list of sorted indices
    indices = list(range(df_size))
    random.shuffle(indices)

    # setting split points of DataFrame
    split_tr_val = int(math.floor(df_size * train_size))
    split_val_te = int(math.floor(split_tr_val + df_size * validation_size))

    if test_size != 0:

        # putting indeces of tr_data, val_data and te_data into lists
        tr_data_indices  = indices[0:split_tr_val]
        val_data_indices = indices[split_tr_val+1: split_val_te]
        te_data_indices  = indices[split_val_te+1:df_size-1]

        # dividing given DataFrame into 3 parts
        tr_data  = df.iloc[tr_data_indices,:]

        val_data = df.iloc[val_data_indices,:]

        te_data  = df.iloc[te_data_indices,1:]

        return tr_data, val_data, te_data;
    
    else:
        # putting indeces of tr_data, val_data and te_data into lists
        tr_data_indices  = indices[0:split_tr_val]
        val_data_indices = indices[split_tr_val+1: df_size-1]
        
        # dividing given DataFrame into 2 parts
        tr_data  = df.iloc[tr_data_indices,:]

        val_data = df.iloc[val_data_indices,:]

        return tr_data, val_data;

def corr_above_cutoff(df, cutoff=0.6):

    tr_data_corr = df.corr()>cutoff
    indices_list = []
    for i in range(df.shape[1]):
        for j in range(i):
            if tr_data_corr.iloc[j][i] == True:
                if i not in indices_list:
                    indices_list.append(i)
        
    return indices_list


def delete_corr(df, list_of_indices):

    columns_to_delete = list(df.columns[list_of_indices])
    for column in columns_to_delete: df =  df.drop(column,axis=1)

    return df

