"""
Summary: Predict donation amount
TODO:
"""
import numpy as np
import pandas as pd
from ml_tools import ModelTrainer

from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

from utils import get_feature_importances, plot_feature_importances

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

SEED = 1337

# Read in the data and split training from the test set
df_train = pd.read_csv('input_data/model_data/df_train_reg.csv', low_memory=False)
# Get the test set with the predicted probas
df_test_clf = pd.read_csv('input_data/classifier_output/df_validation.csv', low_memory=False)
# Get the test set from the regression feature selection
df_test_reg = pd.read_csv('input_data/model_data/df_validation_reg.csv', low_memory=False)

# Join the predicted probas to the regression test set
df_test_reg.loc[:, 'calibrated_proba_donation'] = df_test_clf['calibrated_proba_donation']
df_test = df_test_reg.copy()

# Get lists of the feature columns and the additional columns
ls_features = [col for col in df_train.columns if col not in ['TARGET_B', 'TARGET_D', 'CONTROLN']]
ls_extra_cols = [col for col in df_train.columns if col not in ls_features]

# Define the hyperparameters for the lgb classifier
hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'is_training_metric': True,
    'is_enable_sparse': True,
    'use_two_round_loading': False,
    'num_machines': 1,
    'verbose': 0,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'min_split_gain': 0.0,
    'colsample_bytree': 1.0,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'n_jobs': -1,
    'random_state': SEED
}

# Define possible values for a full grid search
grid_params = {
    'min_data_in_leaf': [20, 50, 80, 100, 200],
    'learning_rate': [0.1, 0.01, 0.005],
    'num_leaves': [5, 10, 15, 40],
    'max_depth': [3, 5, 10, 15],
    'objective': ['regression'],
    # 'max_bin': [100, 500, 750, 1000, 2000],
    'bagging_fraction': [0.8, 1.],
    'feature_fraction': [0.8, 1.],
    # 'num_iterations': [200, 1000, 10000],
    # 'early_stopping_rounds': [350]
}
# Define possible values for a random grid search
grid_params_randomized = {
    'min_data_in_leaf': list(np.arange(20, 200, 10)),
    'learning_rate': list(np.arange(0.05, 0.3, 0.05)),
    'num_leaves': list(np.arange(5, 50, 5)),
    'max_depth': list(np.arange(3, 16, 1)),
    'objective': ['regression'],
    'bagging_fraction': list(np.arange(0.4, 1., .1)),
    'feature_fraction': list(np.arange(0.4, 1., .1)),
}

# Define scoring for the classifier
mt = ModelTrainer(
         problem_type='regression',
         df_train=df_train,
         df_test=df_test,
         ls_features=ls_features,
         col_target='TARGET_D',
         n_folds=5,
         grid=grid_params_randomized,
         refit='neg_mean_squared_error',
         n_jobs=-1,
         random_state=SEED
)
# Run the grid and get the parameters of the best model and the fitted grid object
# best_params, grid_object = mt.grid_search(randomized=False)
best_params_rand, grid_object_rand = mt.grid_search(randomized=True, metric='neg_root_mean_squared_error', n_iter=25)

# Predict on the test set using the best model from the grid
preds_test = grid_object_rand.best_estimator_.predict(df_test[ls_features])
mse_test = mean_squared_error(df_test['TARGET_D'], preds_test)

# Check feature importances based on the best model
df_feature_importances = get_feature_importances(grid_object_rand.best_estimator_)
df_feature_importances.to_csv('tables/df_feature_importances_regression.csv', index=False)
plot_feature_importances(df_importances=df_feature_importances, top_n=20, name='reg')

# Assign the donation predictions to the test set
df_test.loc[:, 'predicted_donation'] = preds_test

# Save the test set with predicted donations
df_test.to_csv('input_data/regressor_output/df_validation.csv', index=False)
