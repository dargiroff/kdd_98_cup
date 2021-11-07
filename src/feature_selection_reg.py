"""
Summary: Feature selection for the regression problem
TODO:
"""
import pandas as pd
from ml_tools import Preprocessor, FeatureSelector

import lightgbm as lgb

# Read in the data and split training from the test set
df_train = pd.read_csv('input_data/preprocessed_data/df_train_pre.csv', low_memory=False)
df_test = pd.read_csv('input_data/preprocessed_data/df_validation_pre.csv', low_memory=False)

# Get lists of the feature columns and the additional columns
ls_features = [col for col in df_train.columns if col not in ['TARGET_B', 'TARGET_D', 'CONTROLN']]
ls_extra_cols = [col for col in df_train.columns if col not in ls_features]

# Select only the training samples that have donated
df_train = df_train[df_train['TARGET_B'] == 1]

pre_train = Preprocessor(df=df_train)
fs_train = FeatureSelector(df_train=df_train, ls_features=ls_features, ls_extra_cols=ls_extra_cols)

# Drop the first feature out of pairs that have correlation of above .9
# fs_train.drop_correlated_features(corr_thresh=.9)

# Label encode categorical features and then cast them as int
# the int cast is recommended for LGBM - see https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
ls_categorical = list(fs_train.df_train.select_dtypes(include=['category', 'object']).columns)
map_encodings = fs_train.label_encode_categories(ls_features_to_encode=ls_categorical, cast_to_int=True)
# Apply the encodings to the test dataset
df_test = fs_train.apply_label_encoding(df=df_test,
                                        ls_categories=ls_categorical,
                                        map_encodings=map_encodings,
                                        cast_to_int=False)

ls_cols_to_keep = fs_train.recursive_feature_elimination(
    target='TARGET_D',
    model=lgb.LGBMRegressor(),
    model_params={},
    rfe_metric='neg_mean_squared_error',
    step=1,
    min_features=1,
    verbose=20,
    n_jobs=-1,
)

# Apply the selection to the test dataset
df_test = df_test[fs_train.df_train.columns]

# Save the datasets for modelling
df_test.to_csv('input_data/model_data/df_validation_reg.csv', index=False)
fs_train.df_train.to_csv('input_data/model_data/df_train_reg.csv', index=False)
