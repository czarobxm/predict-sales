import re
import math
import random
import zipfile

import pandas as pd


def unzip_data(zip_path):
    """
    Unzip data from given path
    """

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall()


def delete_letters(df, column_name="size"):
    """
    Delete letters from each cell in given column and changes its type from string into integer
    """
    tab = [
        int(re.sub("[^0-9]", "", idx)) for idx in df[column_name].value_counts().index
    ]
    indices = df[column_name].value_counts().index
    dict_size = {indices[i]: val for i, val in enumerate(tab)}
    if column_name == "size":
        dict_size["2x68g"] = 136
    df[column_name] = df[column_name].map(dict_size)
    return df


def cat_into_num(df):
    """
    Change categorical columns into numerical columns
    """
    index = []
    # save indexes of columns with dtype == object
    for i, dtype in enumerate(df.dtypes):
        if dtype == "O":
            index.append(i)
    # if every object column change its categorical values to numerical values
    for i in index:
        df.iloc[:, i] = pd.factorize(df.iloc[:, i])[0]
    return df


def split(df, train_size=0.6, validation_size=0.2, test_size=0.2, shuffle=True):
    """
    Devide given DataFrame into tr_set, val_set and te_set with given sizes of these sets
    """

    # sanity check
    if train_size + validation_size + test_size != 1:
        raise ValueError(
            "Sum of train_size, validation_size and test_size must be equal to 1"
        )

    df_size = df.shape[0]

    # shuffling list of sorted indices
    indices = list(range(df_size))
    if shuffle:
        random.shuffle(indices)

    # setting split points of DataFrame
    split_tr_val = int(math.floor(df_size * train_size))
    split_val_te = int(math.floor(split_tr_val + df_size * validation_size))

    if test_size != 0:
        # putting indeces of tr_data, val_data and te_data into lists
        tr_data_indices = indices[0:split_tr_val]
        val_data_indices = indices[split_tr_val + 1 : split_val_te]
        te_data_indices = indices[split_val_te + 1 : df_size - 1]

        # dividing given DataFrame into 3 parts
        tr_data = df.iloc[tr_data_indices, :]

        val_data = df.iloc[val_data_indices, :]

        te_data = df.iloc[te_data_indices, 1:]

        return tr_data, val_data, te_data

    else:
        # putting indeces of tr_data, val_data and te_data into lists
        tr_data_indices = indices[0:split_tr_val]
        val_data_indices = indices[split_tr_val + 1 : df_size - 1]

        # dividing given DataFrame into 2 parts
        tr_data = df.iloc[tr_data_indices, :]

        val_data = df.iloc[val_data_indices, :]

        return tr_data, val_data, te_data


def corr_above_cutoff(df, base_col="target_sales", cutoff=0.3, delete=True):
    tr_data_corr = df.corr().abs() < cutoff
    indices_list = tr_data_corr.columns[tr_data_corr[base_col]]
    if delete:
        return df.drop(indices_list, axis=1)
    return indices_list
