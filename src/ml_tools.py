"""
Summary: ML tools to be used in modelling and data preprocessing
TODO: Make the label encoder more abstract - maybe a separate class
"""
from typing import Union, Tuple

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFECV

import lightgbm as lgb


class Preprocessor:
    def __init__(self,
                 df: pd.DataFrame):
        self.df = df

    def convert_dates(self,
                      ls_col_dates: list
                      ) -> None:
        """
        Convert date strings in YYMM format to dates
        :param ls_col_dates: list of the column names of the date columns
        :return: None
        """
        self.df[ls_col_dates] = self.df[ls_col_dates].astype('str')
        for col in ls_col_dates:
            self.df.loc[:, col] = self.df[col].astype('str')
            year = self.df[col].apply(lambda x: '19' + x[:2])
            month = self.df[col].apply(lambda x: x[2:4])
            date = year.astype(str) + '-' + month.astype(str)
            date = date.apply(lambda x: x if len(x) == 7 else '-')

            self.df[col] = pd.to_datetime(date, errors='coerce').dt.date

    def calc_day_diff(self,
                      ls_cols_diff: list,
                      present_date=pd.to_datetime('9807', format='%y%m').date(),
                      drop_col_dates=True
                      ) -> None:
        """
        Calculate difference in days between a present date and a column of dates
        :param ls_cols_diff: list of the column names for the date columns we want to take difference with
        :param present_date: the date we want to take difference with
        :param drop_col_dates: if True, we drop the differenced columns
        :return: None
        """
        for col in ls_cols_diff:
            self.df.loc[:, col + '_diff_days'] = (present_date - self.df[col]).dt.days

        if drop_col_dates:
            self.df = self.df.drop(columns=ls_cols_diff)

    @staticmethod
    def _select_categorical_features(df: pd.DataFrame,
                                     ls_features: list,
                                     ls_types=None
                                     ) -> list:
        """
        Select features based on column type
        :param df: the dataframe containing the features
        :param ls_features: the features possible to select from
        :param ls_types: list of the types to select
        :return: a list of the selected features based on type
        """
        if ls_types is None:
            ls_types = ['object', 'category']
        ls_cats = list(df[ls_features].select_dtypes(include=ls_types).columns)

        return ls_cats

    def cast_categorical(self,
                         ls_features: list,
                         ls_types=None
                         ) -> None:
        """
        Cast a selection of variables to category type
        :param ls_features: list of features to select categories from
        :param ls_types: list of types to cast to category
        :return: None
        """
        if ls_types is None:
            ls_types = ['object', 'category']
        ls_categorical_vars = self._select_categorical_features(self.df, ls_features, ls_types=ls_types)
        self.df[ls_categorical_vars] = self.df[ls_categorical_vars].astype('category')

    def replace_strings(self,
                        ls_columns: list,
                        str_to_replace: str,
                        replacement: Union[int, str],
                        regex: bool,
                        astype: str
                        ) -> None:
        """
        Replace strings with a category
        :param ls_columns: list of columns to replace in
        :param str_to_replace: what to replace
        :param replacement: what to replace the string with
        :param regex: if True, use regex
        :param astype: the type of the column in which the strings were replaced
        :return: None
        """
        if regex:
            self.df[ls_columns] = self.df[ls_columns].replace(fr'{str_to_replace}',
                                                              replacement,
                                                              regex=True
                                                              ).astype(astype)
        else:
            self.df[ls_columns] = self.df[ls_columns].replace(f'{str_to_replace}',
                                                              replacement,
                                                              ).astype(astype)

    def replace_str_with_bool(self,
                              col: str,
                              str_to_replace: str,
                              bool_order='01',
                              astype='category'
                              ) -> None:
        """
        Replace a string with True/False
        :param col: the name of the column in which to replace
        :param str_to_replace: what to replace
        :param bool_order: 01 to replace the string with 0 and everything else with 1; 10 for the opposite
        :param astype: the type of the column in which the strings were replaced
        :return: None
        """
        if bool_order == '01':
            self.df[f'cat_{col}'] = np.where(self.df[col] == f'{str_to_replace}', 0, 1).astype(astype)
        elif bool_order == '10':
            self.df[f'cat_{col}'] = np.where(self.df[col] == f'{str_to_replace}', 1, 0).astype(astype)
        else:
            raise ValueError('01 and 10 are the only supported boolean orders')

    def extract_str_bits(self,
                         col: str,
                         n_bits: int,
                         astype='category'
                         ) -> None:
        """
        Extract the symbols of a string column per row and make new columns
        :param col: the name of the column
        :param n_bits: number of symbols to extract
        :param astype: the type of the new column(s)
        :return: None
        """
        for bit in range(0, n_bits + 1):
            self.df[f'{bit + 1}_bit_{col}'] = self.df[col].apply(lambda x: x[0] if len(x) >= 1 else np.nan
                                                                 ).astype(astype)

    def group_categories(self,
                         col: str,
                         size_thresh: float,
                         new_category: Union[int, str],
                         astype='category'
                         ) -> None:
        """
        Group categories with few samples into 'other' type of category
        :param col: the name of the categorical column
        :param size_thresh: percentage of the dataframe len, if <= size_thresh, the category will be grouped
        :param new_category: the name of the new category
        :param astype: the type of the new category
        :return: None
        """
        cat_counts = self.df[col].value_counts()
        for key in cat_counts.keys():
            if cat_counts[key] < size_thresh * len(self.df):
                self.df[col].replace(key, new_category, inplace=True)
        self.df[col] = self.df[col].astype(astype)

    def suggest_gender_from_title(self) -> None:
        """
        Extract uknown genders from the person's title via a mapping
        :return: None
        """
        for idx, data in self.df[(self.df['GENDER'] == ' ') | (self.df['GENDER'] == 'U')].iterrows():
            # MR. title
            if (self.df.loc[idx, 'TCODE'] in [1]) & (self.df.loc[idx, 'GENDER'] in ['U', ' ']):
                self.df.loc[idx, 'GENDER'] = 'M'
            # MRS., MISS, and MS. titles
            elif (self.df.loc[idx, 'TCODE'] in [2, 3, 28]) & (self.df.loc[idx, 'GENDER'] in ['U', ' ']):
                self.df.loc[idx, 'GENDER'] = 'F'


class FeatureSelector:
    def __init__(self,
                 df_train: pd.DataFrame,
                 ls_features: list,
                 ls_extra_cols: list
                 ):
        self.df_train = df_train
        self.ls_features = ls_features
        self.ls_extra_cols = ls_extra_cols

    def label_encode_categories(self,
                                ls_features_to_encode: list,
                                cast_to_int=True
                                ) -> dict:
        """
        Label encode categories
        :param ls_features_to_encode: list of features to encode
        :param cast_to_int: if True, the column is cast to int after the encoding
        This is recommended for LGBM; see: https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
        :return: dict of the encodings, to be applied on a out-of-sample dataset
        """
        label_encoder = LabelEncoder()
        map_encodings = dict()
        for col in ls_features_to_encode:
            self.df_train[col] = label_encoder.fit_transform(self.df_train[col])
            map_encodings[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            if cast_to_int:
                self.df_train[col] = self.df_train[col].astype('int')

        return map_encodings

    @staticmethod
    def apply_label_encoding(df: pd.DataFrame,
                             ls_categories: list,
                             map_encodings: dict,
                             cast_to_int=True
                             ) -> pd.DataFrame:
        """
        Apply label encoding to a set of columns based on mapping derived in label_encode_categories()
        :param df: the dataframe the encodings will be applied to
        :param ls_categories: list of categories to be encoded
        :param map_encodings: a map of the encodings
        :param cast_to_int: cast_to_int: if True, the column is cast to int after the encoding
        Not needed to be True if the map has the encodings mapped to ints
        :return: the encoded dataframe
        """
        for col in ls_categories:
            df[col] = df[col].map(map_encodings[col])
            if cast_to_int:
                df[col] = df[col].astype('int')

        return df

    def drop_correlated_features(self,
                                 corr_thresh=0.9
                                 ) -> None:
        """
        Drop the first feature between pairs correlated more than an absolute threshold
        :param corr_thresh: the correlation threshold
        :return: None
        """
        corr_matrix = self.df_train[self.ls_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        ls_correlated_features = [col for col in upper.columns if any(upper[col] >= corr_thresh)]
        self.df_train = self.df_train.drop(columns=ls_correlated_features)
        self.ls_features = [col for col in self.df_train.columns if
                            col not in self.ls_extra_cols]

    def recursive_feature_elimination(self,
                                      target: str,
                                      model: Union[lgb.LGBMClassifier, lgb.LGBMRegressor],
                                      model_params: dict,
                                      rfe_metric: str,
                                      step=1,
                                      min_features=1,
                                      folds=3,
                                      verbose=10,
                                      n_jobs=-1
                                      ) -> list:
        """
        Recursively eliminate features
        :param target: the name of the target column
        :param model: the model object
        :param model_params: a dict with the model params
        :param rfe_metric: the evaluation metric during the RFE
        :param step: how many features to drop at a time
        :param min_features: the minimal number of features left before RFE evaluation is over
        :param folds: number of folds for the RFE
        :param verbose: how much information is shown during RFE
        :param n_jobs: how many CPU cores are used
        :return: list of features to keep based on the RFE results
        """
        X_train, y_train = self.df_train[self.ls_features], self.df_train[target]

        model = model
        model.set_params(**model_params)
        rfecv = RFECV(
            estimator=model,
            step=step,
            cv=folds,
            scoring=rfe_metric,
            min_features_to_select=min_features,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        rfecv.fit(X_train, y_train)

        ls_cols_to_keep = [col for col in rfecv.get_feature_names_out()]
        ls_cols_to_keep += self.ls_extra_cols
        self.df_train = self.df_train[ls_cols_to_keep]

        return ls_cols_to_keep


class ModelTrainer:
    def __init__(self,
                 problem_type: str,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 ls_features: list,
                 col_target: str,
                 n_folds: int,
                 grid: dict,
                 refit: Union[str, bool],
                 n_jobs=-1,
                 random_state=1337
                 ):
        self.problem_type = problem_type
        self.df_train = df_train
        self.df_test = df_test
        self.ls_features = ls_features
        self.col_target = col_target
        self.n_folds = n_folds
        self.grid = grid
        self.refit = refit
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
        """
        Split the data into features and target; split folds for CV
        :return: the feature df, the target df, list of tuples containing indeces for CV (in order train, val)
        """

        self.df_train = self.df_train.sample(frac=1).reset_index(drop=True)
        X_train, y_train = self.df_train[self.ls_features], self.df_train[self.col_target]
        y_train = pd.DataFrame(y_train)

        if self.problem_type == 'classification':
            ls_indeces = list()
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=X_train, y=y_train)):
                X_train.loc[val_idx, 'fold'] = fold
                y_train.loc[val_idx, 'fold'] = fold

                ls_indeces.append((train_idx, val_idx))

            return X_train, y_train, ls_indeces

        elif self.problem_type == 'regression':
            num_bins = int(np.floor(1 + np.log2(len(X_train))))
            if num_bins > 10:
                num_bins = 10

            ls_indeces = list()

            X_train.loc[:, 'target'] = y_train[self.col_target].values
            X_train['target_bins'] = pd.cut(X_train.loc[:, 'target'].ravel(), bins=num_bins, labels=False)
            X_train.drop(columns=['target'], inplace=True)

            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=X_train, y=X_train['target_bins'].values)):
                X_train.loc[val_idx, 'fold'] = fold
                y_train.loc[val_idx, 'fold'] = fold

                ls_indeces.append((train_idx, val_idx))

            X_train = X_train.drop("target_bins", axis=1)

            return X_train, y_train, ls_indeces
        else:
            raise Exception(f'Unrecognized problem type {self.problem_type}')

    def grid_search(self,
                    randomized=True,
                    metric=None,
                    n_iter=10
                    ) -> Tuple[dict, Union[RandomizedSearchCV, GridSearchCV]]:
        """
        Do a full or randomized grid search
        :param randomized: if True do randomized grid search
        :param metric: the scoring metric for the grid search
        :param n_iter: number of iterations for the randomized grid search; not used for the full grid search
        :return: the best parameters based on the search and the fitted grid object
        """
        if self.problem_type == 'classification':
            if metric is None:
                metric = 'roc_auc'
            model = lgb.LGBMClassifier()
        elif self.problem_type == 'regression':
            if metric is None:
                metric = 'neg_root_mean_squared_error'
            model = lgb.LGBMRegressor()
        else:
            raise Exception(f'Unrecognized problem type {self.problem_type}')

        X_train, y_train, ls_indeces = self._split_data()

        if randomized:
            grid_object = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.grid,
                cv=ls_indeces,
                n_jobs=self.n_jobs,
                refit=self.refit,
                scoring=metric,
                n_iter=n_iter,
                verbose=10,
                random_state=self.random_state
            )
        else:
            grid_object = GridSearchCV(
                estimator=model,
                param_grid=self.grid,
                cv=ls_indeces,
                n_jobs=self.n_jobs,
                refit=self.refit,
                scoring=metric,
                verbose=10,
            )

        grid_object.fit(X_train[self.ls_features], y_train[self.col_target].values)

        print(f'Best parameters found by grid search are: {grid_object.best_params_}')
        best_params = grid_object.best_estimator_.get_params()

        return best_params, grid_object
