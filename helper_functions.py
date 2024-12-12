import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import os
import pickle

def prepare_data(df, features, targets):
    df['nr_boolean'] = df['nrtask'].apply(lambda x: 0 if x == 0 else 1)
    df['jic_start_datetime'] = pd.to_datetime(df['jic_start_datetime'])
    df_sorted = df.sort_values(by=['registration', 'jic_start_datetime'])
    registration_groups = df_sorted.groupby('registration')

    for i in range(1, 6):
        # Create a new column 'nrlabour-i' with shifted values
        df_sorted[f'nrlabour-{i}'] = registration_groups['nrlabour'].shift(i).fillna(0)
        df_sorted[f'nrtask-{i}'] = registration_groups['nrtask'].shift(i).fillna(0)
    nrlabour_n = [f'nrlabour-{i}' for i in range(1, 6)]
    nrtask_n = [f'nrtask-{i}' for i in range(1, 6)]
    features += nrlabour_n
    features += nrtask_n

    if 'registration' in features:
        registration_mapping = {label: idx for idx, label in enumerate(df_sorted['registration'].unique(), 1)}
        df_sorted['registration'] = df_sorted['registration'].map(registration_mapping)
        df_sorted= df_sorted[features+targets]
        df_encoded = pd.get_dummies(df_sorted, columns=['registration', 'type'], drop_first=True)
        df_encoded = df_encoded[[col for col in df_encoded.columns if col not in targets] + targets]
        return df_encoded, registration_mapping
    else:
        df_sorted = df_sorted[features+targets]
        df_encoded = pd.get_dummies(df_sorted, columns=['type'], drop_first=True)
        df_encoded = df_encoded[[col for col in df_encoded.columns if col not in targets] + targets]
        return df_encoded

def prepare_data_ordinal(df, features, targets):
    df['nr_boolean'] = df['nrtask'].apply(lambda x: 0 if x == 0 else 1)
    df['jic_start_datetime'] = pd.to_datetime(df['jic_start_datetime'])
    df_sorted = df.sort_values(by=['registration', 'jic_start_datetime'])
    registration_groups = df_sorted.groupby('registration')

    n_backward = 5
    for i in range(1, n_backward+1):
        # Create new columns 'nrlabour-i' and 'nrtask-i' with shifted values
        df_sorted[f'nrlabour-{i}'] = registration_groups['nrlabour'].shift(i).fillna(0)
        df_sorted[f'nrtask-{i}'] = registration_groups['nrtask'].shift(i).fillna(0)
    nrlabour_n = [f'nrlabour-{i}' for i in range(1, n_backward+1)]
    nrtask_n = [f'nrtask-{i}' for i in range(1, n_backward+1)]
    features += nrlabour_n
    features += nrtask_n

    features = list(set(features))
    df_sorted = df_sorted[features + targets]

    # Determine categorical columns to be ordinally encoded
    categorical_cols = df_sorted.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder()
        # Apply ordinal encoding to each categorical column
        for col in categorical_cols:
            # Ensure no NaN values which can't be handled by OrdinalEncoder directly
            df_sorted[col] = df_sorted[col].fillna('Missing')
            df_sorted[col] = encoder.fit_transform(df_sorted[[col]])

    df_encoded = df_sorted[[col for col in df_sorted.columns if col not in targets] + targets]
    return df_encoded

def prepare_data_lgbm(df, features, targets):
    df['nr_boolean'] = df['nrtask'].apply(lambda x: 0 if x == 0 else 1)
    df['jic_start_datetime'] = pd.to_datetime(df['jic_start_datetime'])
    df_sorted = df.sort_values(by=['registration', 'jic_start_datetime'])
    registration_groups = df_sorted.groupby('registration')

    n_backward = 5
    for i in range(1, n_backward+1):
        # Create new columns 'nrlabour-i' and 'nrtask-i' with shifted values
        df_sorted[f'nrlabour-{i}'] = registration_groups['nrlabour'].shift(i).fillna(0)
        df_sorted[f'nrtask-{i}'] = registration_groups['nrtask'].shift(i).fillna(0)
    nrlabour_n = [f'nrlabour-{i}' for i in range(1, n_backward+1)]
    nrtask_n = [f'nrtask-{i}' for i in range(1, n_backward+1)]
    features += nrlabour_n
    features += nrtask_n

    features = list(set(features))
    df_sorted = df_sorted[features + targets]

    # Determine categorical columns to be ordinally encoded
    categorical_cols = df_sorted.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder()
        # Apply ordinal encoding to each categorical column
        for col in categorical_cols:
            # Ensure no NaN values which can't be handled by OrdinalEncoder directly
            df_sorted[col] = df_sorted[col].fillna('Missing')
            df_sorted[col] = df_sorted[col].astype('category')

    df_encoded = df_sorted[[col for col in df_sorted.columns if col not in targets] + targets]
    return df_encoded

def split_df(df_encoded):
    df_labour = df_encoded.pop('nrlabour')
    df_boolean = df_encoded.pop('nr_boolean')
    df_task = df_encoded.pop('nrtask')

    X_train, X_test= train_test_split(df_encoded, test_size=0.2, random_state=42)
    y_train_h, y_test_h = train_test_split(df_labour, test_size=0.2, random_state=42)
    y_train_b, y_test_b = train_test_split(df_boolean, test_size=0.2, random_state=42)
    X_train_h = pd.concat([X_train, y_train_b], axis=1)

    return X_train, X_train_h, X_test, y_train_b, y_train_h, y_test_b, y_test_h

def generate_sets():
    folder_path = 'RT_datasets'
    df_list = []  # Initialize an empty list to store the DataFrames
    y_train_h_list = []
    y_train_b_list = []
    y_test_h_list = []
    y_test_b_list = []
    X_test_list = []
    X_train_list = []
    X_train_h_list = []

    gooon = True
    dict_jic_lengths = {}

    for i, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if filename != '.DS_Store':
            dfRT = pd.read_csv(file_path)
        #if filename.split('-')[0] == 'AMPGV':
        #if filename == 'AMPGV-200-4001.csv':
        if dfRT.shape[0] > 20 and 'nrtask' in dfRT.columns:
            df_list.append(dfRT)  # Append the DataFrame to the list
            features = ['scheduled_labor_hours', 'actual_labor_hours', 'type', 'flycycle', 'flyhour', 'jic_code']
            targets = ['nrtask', 'nrlabour', 'nr_boolean']
            df_encoded = prepare_data_ordinal(dfRT, features, targets)
            df_encoded['jic_code'] = i
            dict_jic_lengths[i] = df_encoded.shape[0]
            df_encoded.loc[df_encoded['nrlabour'] == 0, 'nr_boolean'] = 0
            X_train, X_train_h, X_test, y_train_b, y_train_h, y_test_b, y_test_h = split_df(df_encoded)
            y_train_h_list.append(pd.Series(y_train_h))
            y_train_b_list.append(pd.Series(y_train_b))
            y_test_h_list.append(pd.Series(y_test_h))
            y_test_b_list.append(pd.Series(y_test_b))
            X_test_list.append(X_test)
            X_train_list.append(X_train)
            X_train_h_list.append(X_train_h)

    # Concatenate all DataFrames and Series
    df_tot = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    y_train_h = pd.concat(y_train_h_list, ignore_index=True) if y_train_h_list else pd.Series(dtype=float)
    y_train_b = pd.concat(y_train_b_list, ignore_index=True) if y_train_b_list else pd.Series(dtype=float)
    y_test_h = pd.concat(y_test_h_list, ignore_index=True) if y_test_h_list else pd.Series(dtype=float)
    y_test_b = pd.concat(y_test_b_list, ignore_index=True) if y_test_b_list else pd.Series(dtype=float)
    X_test = pd.concat(X_test_list, ignore_index=True) if X_test_list else pd.DataFrame()
    X_train = pd.concat(X_train_list, ignore_index=True) if X_train_list else pd.DataFrame()
    X_train_h = pd.concat(X_train_h_list, ignore_index=True) if X_train_h_list else pd.DataFrame()
    return df_tot, y_train_h, y_train_b, y_test_h, y_test_b, X_test, X_train, X_train_h, dict_jic_lengths


def generate_sets2(folder_path='RT_datasets'):
    df_list = []  # List to store each DataFrame
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename != '.DS_Store' and filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            if df.shape[0] > 20 and 'nrtask' in df.columns:
                df_list.append(df)

    # Combine all data into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Assume prepare_data_ordinal() is defined elsewhere
    features = ['scheduled_labor_hours', 'actual_labor_hours', 'type', 'flycycle', 'flyhour', 'jic_code']
    targets = ['nrtask', 'nrlabour', 'nr_boolean']

    combined_df_encoded = prepare_data_lgbm(combined_df, features, targets)

    categorical_features = ['type', 'flycycle', 'jic_code']  # Specify your categorical columns
    encoder = OrdinalEncoder()
    combined_df_encoded[categorical_features] = encoder.fit_transform(combined_df_encoded[categorical_features])

    # Splitting data by 'jic_code'
    X_train_list = []
    X_train_h_list = []
    X_test_list = []
    y_train_h_list = []
    y_train_b_list = []
    y_test_h_list = []
    y_test_b_list = []
    dict_jic_lengths = {}

    print(combined_df_encoded.shape)
    
    for jic_code, group in combined_df_encoded.groupby('jic_code'):
        print(group.shape[0])
        dict_jic_lengths[jic_code] = group.shape[0]
        X_train, X_train_h, X_test, y_train_b, y_train_h, y_test_b, y_test_h = split_df(group)
        X_train_h_list.append(X_train_h)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_h_list.append(y_train_h)
        y_train_b_list.append(y_train_b)
        y_test_h_list.append(y_test_h)
        y_test_b_list.append(y_test_b)

    # Concatenate all splits
    X_train = pd.concat(X_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_train_h = pd.concat(y_train_h_list, ignore_index=True)
    y_train_b = pd.concat(y_train_b_list, ignore_index=True)
    y_test_h = pd.concat(y_test_h_list, ignore_index=True)
    y_test_b = pd.concat(y_test_b_list, ignore_index=True)
    
    return X_train, y_train_h, y_train_b, X_test, y_test_h, y_test_b, dict_jic_lengths

# Assuming prepare_data_ordinal and split_df are properly defined elsewhere in your code.

# Function to load an object from a pickle file
def load_from_pickle(file_name, load_directory):
    file_path = os.path.join(load_directory, file_name)
    with open(file_path, 'rb') as file:
        return pickle.load(file)
