from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import *

def train_svm_regress_model(X_train, y_train):
    model = SVR()
    
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = SVR(**best_params)
    best_model.fit(X_train, y_train)  # Fit the best model
    
    return best_model, best_params


def train_xgb_regress_model(X_train, y_train):
    # Define the regressor with other parameters
    model = XGBRegressor(
        objective='reg:squarederror',  # Objective suitable for regression
        random_state=42,
        eval_metric='rmse'  # Using RMSE as the evaluation metric for regression
    )

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize the GridSearchCV object with minimal verbosity
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Extract the best parameters from the grid search
    best_params = grid_search.best_params_

    # Reinitialize the regressor with the best parameters found
    best_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        eval_metric='rmse',
        **best_params  # Unpack the best parameters
    )

    # Fit the model with the best parameters on the training data
    best_model.fit(X_train, y_train)

    # Return the best model and its parameters
    return best_model, best_params

def train_lgbm_regress_model(X_train, y_train):

    best_params =  param_grid = {
        'n_estimators': 50,
        'max_depth': 12,
        'learning_rate': 0.1,
        'subsample': 1,  # Subsample ratio of the training instances.
        'colsample_bytree': 1,  # Subsample ratio of columns when constructing each tree.
        'num_leaves': 64  # Number of leaves in full tree
    }

    # Reinitialize the LightGBM regressor with the best parameters found
    best_model = LGBMRegressor(
        objective='regression',
        random_state=42,
        metric='rmse',
        **best_params  # Unpack the best parameters
    )

    # Fit the model with the best parameters on the training data
    best_model.fit(X_train, y_train)

    # Return the best model and its parameters
    return best_model, best_params


def train_xgb_binary_class_model(X_train, y_train):
    # Define the classifier with initial parameters
    model = XGBClassifier(
        objective='binary:logistic',  # Objective suitable for binary classification
        random_state=42,
        eval_metric='logloss'  # Using log loss as the evaluation metric for classification
    )

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize the GridSearchCV object with minimal verbosity
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Extract the best parameters from the grid search
    best_params = grid_search.best_params_

    best_params = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 18, 'n_estimators': 100, 'subsample': 1.0}

    # Reinitialize the classifier with the best parameters found
    best_model = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        **best_params  # Unpack the best parameters
    )

    # Fit the model with the best parameters on the training data
    best_model.fit(X_train, y_train)

    # Return the best model and its parameters
    return best_model, best_params

def train_lgbm_binary_class_model(X_train, y_train, class_weight=None):
    # Reinitialize the LightGBM classifier with the best parameters found
    best_model = LGBMClassifier(
        objective='binary',
        random_state=42,
        metric='binary_logloss',
        n_estimators = 200,
        subsample = 0.8,
        num_leaves = 124,
        max_depth = 18,
        lr = 0.01,
        class_weight=class_weight,
        verbose = -1
    )

    # Fit the model with the best parameters on the training data
    best_model.fit(X_train, y_train)

    # Return the best model and its parameters
    return best_model, 'hi'

def train_rf_binary_class_model(X_train, y_train):
    # Initialize the Random Forest classifier with the best parameters found
    best_model = RandomForestClassifier(
        n_estimators=100,  # Number of trees in the forest
        max_depth=18,      # Maximum depth of the tree
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
        max_features=None,  # Number of features to consider when looking for the best split
        bootstrap=True,       # Whether bootstrap samples are used when building trees
        random_state=42,      # Controls both the randomness of the bootstrapping and the features considered
        verbose=1             # Controls the verbosity when fitting and predicting
    )

    # Fit the model with the best parameters on the training data
    best_model.fit(X_train, y_train)

    # Gather the parameters used in the model for reference or further analysis
    model_params = {
        'n_estimators': best_model.n_estimators,
        'max_depth': best_model.max_depth,
        'min_samples_split': best_model.min_samples_split,
        'min_samples_leaf': best_model.min_samples_leaf,
        'max_features': best_model.max_features,
        'bootstrap': best_model.bootstrap
    }

    # Return the best model and its parameters
    return best_model, model_params


#{'learning_rate': 0.01, 'max_depth': 12, 'n_estimators': 200, 'num_leaves': 31, 'subsample': 0.6}

# def train_lgbm_binary_class_model(X_train, y_train):
#     # Define the classifier with initial parameters
#     model = LGBMClassifier(
#         objective='binary',
#         random_state=42,
#         metric='binary_logloss',
#         verbose=-1
#     )

#     # Define the parameter grid
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'num_leaves': [31, 62, 124],
#         'max_depth': [6, 12, 18],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'subsample': [0.6, 0.8, 1.0]
#     }

#     # Initialize the GridSearchCV object with minimal verbosity
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

#     # Perform the grid search
#     grid_search.fit(X_train, y_train)

#     # Extract the best parameters from the grid search
#     best_params = grid_search.best_params_

#     # Reinitialize the classifier with the best parameters found
#     best_model = LGBMClassifier(
#         objective='binary',
#         random_state=42,
#         metric='binary_logloss',
#         verbose=-1,
#         **best_params  # Unpack the best parameters
#     )

#     # Fit the model with the best parameters on the training data
#     best_model.fit(X_train, y_train)

#     # Return the best model and its parameters
#     return best_model, best_params


def train_adaboost_binary_class_model(X_train, y_train):
    # Define the base classifier, here we use a Decision Tree
    base_classifier = DecisionTreeClassifier(max_depth=1)  # Decision stumps

    # Define the AdaBoost classifier with initial parameters
    model = AdaBoostClassifier(
        estimator=base_classifier,
        random_state=42
    )

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 1.0],
        # 'base_estimator__max_depth': [1, 2, 3]  # Uncomment this if you want to vary depth
    }

    # Initialize the GridSearchCV object with minimal verbosity and cross-validation setup
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Extract the best parameters from the grid search
    best_params = grid_search.best_params_

    # Reinitialize the AdaBoost classifier with the best parameters found
    best_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),  # Reuse decision stumps
        random_state=42,
        **best_params  # Unpack the best parameters
    )

    # Fit the model with the best parameters on the training data
    best_model.fit(X_train, y_train)

    # Return the best model and its parameters
    return best_model, best_params



def train_xgb_multiclass_model(X_train, y_train):
    """
    Train a multiclass classifier using XGBoost with a grid search to find optimal parameters.

    Args:
    X_train (array-like): Feature matrix for training data.
    y_train (array-like): Labels for training data.

    Returns:
    tuple: A tuple containing the trained model and the best parameters dictionary.
    """
    # Define the classifier with initial parameters
    model = XGBClassifier(
        objective='multi:softprob',  # Objective for multiclass probability output
        num_class=3,  # Number of classes
        random_state=42,
        eval_metric='mlogloss'  # Evaluation metric for multiclass problems
    )

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1, verbose=1)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Extract the best parameters from the grid search
    best_params = grid_search.best_params_

    # Reinitialize the classifier with the best parameters found
    best_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
        **best_params  # Unpack the best parameters
    )

    # Fit the model with the best parameters on the training data
    best_model.fit(X_train, y_train)

    # Return the best model and its parameters
    return best_model, best_params

def train_rf_regress_model(X_train, y_train):
    # # Define the regressor with initial parameters
    # model = RandomForestRegressor(
    #     random_state=42,
    #     n_jobs=-1  # Use all available cores
    # )

    # # Define the parameter grid
    # param_grid = {
    #     'n_estimators': [50, 100, 200],  # Number of trees in the forest
    #     'max_depth': [3, 6, 9, None],    # Maximum depth of each tree
    #     'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
    #     'min_samples_leaf': [1, 2, 4],   # Minimum number of samples required at each leaf node
    #     'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
    # }

    # # Initialize the GridSearchCV object with minimal verbosity
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

    # # Perform the grid search
    # grid_search.fit(X_train, y_train)

    # # Extract the best parameters from the grid search
    # best_params = grid_search.best_params_
    
    best_params = {
        'n_estimators': 2000,  # Number of trees in the forest
        'max_depth': None,    # Maximum depth of each tree
        'min_samples_split': 10, # Minimum number of samples required to split an internal node
        'min_samples_leaf': 2,   # Minimum number of samples required at each leaf node
        'max_features': 'log2'  # Number of features to consider when looking for the best split
    }

    # Reinitialize the regressor with the best parameters found
    best_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,  # Use all available cores
        **best_params  # Unpack the best parameters
    )

    # Fit the model with the best parameters on the training data
    best_model.fit(X_train, y_train)

    # Return the best model and its parameters
    return best_model, best_params

def full_LGBM(df_encoded):
    df_labour = df_encoded.pop('nrlabour')
    df_boolean = df_encoded.pop('nr_boolean')
    df_task = df_encoded.pop('nrtask')

    X_train, X_test= train_test_split(df_encoded, test_size=0.2, random_state=42)
    y_train_h, y_test_h = train_test_split(df_labour, test_size=0.2, random_state=42)
    y_train_b, y_test_b = train_test_split(df_boolean, test_size=0.2, random_state=42)
    X_train_h = pd.concat([X_train, y_train_b], axis=1)

    drop = ['nrlabour-5', 'nrlabour-3', 'nrlabour-4', 'nrtask-4', 'nrtask-5', 'nrtask-5']
    X_test = X_test.drop(columns=drop)
    X_train = X_train.drop(columns=drop)

    lgbm_model, binary_params = train_lgbm_binary_class_model(X_train, y_train_b)
    y_pred = lgbm_model.predict(X_test)

    # Assuming y_pred is a NumPy array or a list of predictions
    y_pred_series = pd.Series(y_pred, name='Predictions')
    X_test_h = X_test.copy()
    #X_test_h['Predictions'] = y_pred_series.values

    hours_model, hours_params = train_lgbm_regress_model(X_train, y_train_h)
    y_pred_h = hours_model.predict(X_test_h)
    y_pred_h[y_pred_series == 0] = 0
    y_pred_h[y_pred_h < 0] = 0

    return y_pred_h, y_pred, y_test_h, y_test_b, X_test_h, X_test