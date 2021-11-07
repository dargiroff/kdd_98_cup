"""
Summary: Predict probability of donation
TODO: add confusion matrix?
"""
import numpy as np
import pandas as pd
from ml_tools import ModelTrainer

from matplotlib import pyplot as plt

from utils import get_auc, calibrate_probas, get_feature_importances, plot_feature_importances

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

SEED = 1337

# Read in the data and split training from the test set
df_train = pd.read_csv('input_data/model_data/df_train.csv', low_memory=False)
df_test = pd.read_csv('input_data/model_data/df_validation.csv', low_memory=False)

# Get lists of the feature columns and the additional columns
ls_features = [col for col in df_train.columns if col not in ['TARGET_B', 'TARGET_D', 'CONTROLN']]
ls_extra_cols = [col for col in df_train.columns if col not in ls_features]

# Define the hyperparameters for the lgb classifier
hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'metric_freq': 1,
    'is_training_metric': True,
    'learning_rate': 0.1,
    'tree_learner': 'serial',
    'bagging_freq': 5,
    'min_sum_hessian_in_leaf': 5,
    'is_enable_sparse': True,
    'use_two_round_loading': False,
    'num_machines': 1,
    'verbose': 0,
    'subsample_for_bin': 200000,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'min_split_gain': 0.0,
    'colsample_bytree': 1.0,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'is_unbalanced': True,  # same as scale pos weight, computes the weights auto
    'n_jobs': -1,
    'random_state': SEED
}

# Define possible values for a full grid search
grid_params = {
    'min_data_in_leaf': [20, 50, 80, 100, 200],
    'learning_rate': [0.1, 0.01, 0.005],
    'num_leaves': [5, 10, 15, 40],
    'max_depth': [3, 5, 10, 15],
    'objective': ['binary'],
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
    'objective': ['binary'],
    'bagging_fraction': list(np.arange(0.4, 1., .1)),
    'feature_fraction': list(np.arange(0.4, 1., .1)),
}
# Define scoring for the classifier
mt = ModelTrainer(
         problem_type='classification',
         df_train=df_train,
         df_test=df_test,
         ls_features=ls_features,
         col_target='TARGET_B',
         n_folds=5,
         grid=grid_params_randomized,
         refit='AUC',
         n_jobs=-1,
         random_state=SEED
)
# Run the grid and get the parameters of the best model and the fitted grid object
# best_params, grid_object = mt.grid_search(randomized=False)
best_params_rand, grid_object_rand = mt.grid_search(randomized=True, metric='roc_auc', n_iter=25)

# Get the train features and train target to use when calibrating the predictions
X_train, y_train, ls_indeces = mt._split_data()

# Predict on the test set using the best model from the grid
preds_test = grid_object_rand.best_estimator_.predict_proba(df_test[ls_features])[:, 1]
# Calculate AUC on the test set and make an ROC curve plot
auc, fpr, tpr = get_auc(target=df_test['TARGET_B'], preds=preds_test)
# Check the CV train score for overfit
print(f'Absolute percentage difference between the train and test scores:'
      f' {np.round(np.abs(auc - grid_object_rand.best_score_) / auc * 100, 4)} %')

# Check feature importances based on the best model
df_feature_importances = get_feature_importances(grid_object_rand.best_estimator_)
df_feature_importances.to_csv('tables/df_feature_importances.csv', index=False)
plot_feature_importances(df_importances=df_feature_importances, top_n=20, name='clf')

# Calibrate the model
best_calibrated_probas, dict_calibrations = calibrate_probas(clf=grid_object_rand.best_estimator_,
                                                             X_train=X_train,
                                                             y_train=y_train['TARGET_B'].values,
                                                             X_test=df_test[ls_features],
                                                             y_test=df_test['TARGET_B'].values,
                                                             ls_features=ls_features,
                                                             cv='prefit',
                                                             save_plot=True)


# Append the best calibrated probas to the test set
df_test.loc[:, 'calibrated_proba_donation'] = best_calibrated_probas

# Save the test set with the predicted probabilities
df_test.to_csv('input_data/classifier_output/df_validation.csv', index=False)
