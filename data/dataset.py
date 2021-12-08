import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from category_encoders import LeaveOneOutEncoder

def remove_unused_column(data):
    unused_list = []
    for col in data.columns:
        uni = len(data[col].unique())
        if uni <= 1:
            unused_list.append(col)
    data.drop(columns=unused_list, inplace=True)
    return data

def split_data(data, target):
    label = data[target]
    data = data.drop([target], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=123, shuffle=True)
    return X_train, y_train.values, X_test, y_test.values


def quantile_transform(X_train, X_test):
    quantile_train = np.copy(X_train)
    qt = QuantileTransformer(random_state=1330, output_distribution='normal').fit(quantile_train)
    X_train = qt.transform(X_train)
    X_test = qt.transform(X_test)

    return X_train, X_test

def forest_cover():
    target = "Covertype"

    bool_columns = [
        "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
        "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
        "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
        "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
        "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
        "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
        "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
        "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
        "Soil_Type40"
    ]

    int_columns = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
        ]
    feature = int_columns + bool_columns + [target]
    data = pd.read_csv('./data/forest_cover_type/forest-cover-type.csv', header=None, names=feature)
    train_idx = pd.read_csv('./data/forest_cover_type/train_idx.csv', header=None)[0].values
    train = data.iloc[train_idx, :]
    test_idx = pd.read_csv('./data/forest_cover_type/test_idx.csv', header=None)[0].values
    test = data.iloc[test_idx, :]


    y_train = train[target].values
    X_train = train.drop([target], axis=1).values
    y_test = test[target].values
    X_test = test.drop([target], axis=1).values

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test

def MSLR():
    target = 0
    train = pd.read_pickle('./data/MSLR-WEB10K/train.pkl')
    valid = pd.read_pickle('./data/MSLR-WEB10K/valid.pkl')
    test = pd.read_pickle('./data/MSLR-WEB10K/test.pkl')
    train = pd.concat([train, valid], axis=0)

    y_train = train[target].values
    y_test = test[target].values
    train.drop([target], axis=1, inplace=True)
    test.drop([target], axis=1, inplace=True)
    X_train, X_test = quantile_transform(train, test)

    return X_train, y_train, X_test, y_test

def yahoo():
    target = 0
    train = pd.read_pickle('./data/yahoo/train.pkl')
    valid = pd.read_pickle('./data/yahoo/valid.pkl')
    test = pd.read_pickle('./data/yahoo/test.pkl')
    train = remove_unused_column(train)
    valid = remove_unused_column(valid)
    test = remove_unused_column(test)
    train = pd.concat([train, valid], axis=0)

    y_train = train[target].values
    y_test = test[target].values
    train.drop([target], axis=1, inplace=True)
    test.drop([target], axis=1, inplace=True)
    X_train, X_test = quantile_transform(train, test)
    return X_train, y_train, X_test, y_test

def yearpred():
    target = 0
    data = pd.read_pickle('./data/YearPrediction.pkl')
    label = data[target]
    data = data.drop([target], axis=1)
    y_test = label[-51630:].copy().values
    X_test = data[-51630:].copy()
    y_train = label[:-51630].copy().values
    X_train = data[:-51630].copy()

    X_train, X_test = quantile_transform(X_train, X_test)
    return X_train, y_train, X_test, y_test

def epsilon():
    target = 0
    train = pd.read_pickle('./data/epsilon/train.pkl')
    test = pd.read_pickle('./data/epsilon/test.pkl')

    y_train = train[target].values
    y_test = test[target].values
    X_train = train.drop([target], axis=1).values
    X_test = test.drop([target], axis=1).values

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test

def click():
    target = 'target'
    data = pd.read_pickle('./data/click.pkl')
    label = data[target]
    data.drop([target], axis=1, inplace=True)
    X_train, X_test = data[:-100_000].copy(), data[-100_000:].copy()
    y_train, y_test = label[:-100_000].copy().values, label[-100_000:].copy().values
    cat_features = ['url_hash', 'ad_id', 'advertiser_id', 'query_id',
                    'keyword_id', 'title_id', 'description_id', 'user_id']

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    X_train, X_test = quantile_transform(X_train.astype(np.float32), X_test.astype(np.float32))
    return X_train, y_train, X_test, y_test

def cardio():
    target = 'cardio'
    data = pd.read_csv('./data/cardio/cardiovascular-disease.csv', delimiter=';').drop(['id'], axis=1)
    train_idx = pd.read_csv('./data/cardio/train_idx.csv')['0'].values
    train = data.iloc[train_idx, :]
    test_idx = pd.read_csv('./data/cardio/test_idx.csv')['0'].values
    test = data.iloc[test_idx, :]

    y_train = train[target].values
    train.drop([target], axis=1, inplace=True)
    y_test = test[target].values
    test.drop([target], axis=1, inplace=True)

    X_train, X_test = quantile_transform(train, test)
    return X_train, y_train, X_test, y_test


def get_data(datasetname):
    if datasetname == 'forest':
        return forest_cover()
    elif datasetname == 'MSLR':
        return MSLR()
    elif datasetname == 'year':
        return yearpred()
    elif datasetname == 'cardio':
        return cardio()
    elif datasetname == 'yahoo':
        return yahoo()
    elif datasetname == 'epsilon':
        return epsilon()
    elif datasetname == 'click':
        return click()

if __name__ == '__main__':
    forest_cover()