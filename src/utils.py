"""
Summary: Utility functions to be used in data exploration
TODO:
"""
from typing import Union, Tuple
from sklearn.base import ClassifierMixin
from lightgbm import LGBMClassifier, LGBMRegressor

import pandas as pd
import numpy as np

from scipy.stats import normaltest, shapiro, kstest, zscore, pearsonr
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from statsmodels.graphics.gofplots import qqplot

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

rng = np.random.RandomState(1337)


def check_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes the number and the percentage of missing observations per variable
    :param data: pandas DataFrame
        The dataframe for which missings will be summarized
    :return: df_missings: pandas Dataframe
        The summary of the missings
    """
    df_missings = pd.DataFrame({"variable": data.columns,
                                "number of missings": data.isnull().sum().reset_index().iloc[:, 1],
                                "percentage of missings": 100 * data.isnull().sum().reset_index().iloc[:, 1] / len(
                                    data)})

    return df_missings


def check_normality(data: pd.DataFrame,
                    list_vars: list,
                    alpha: float,
                    qq_plot=False
                    ) -> pd.DataFrame:
    """
    Checks normality of a array-like dataset via 3 stastical tests and a Quantile-Quantile plot
    :param list_vars: list
        The list of vars to be checked.
    :param data: pandas Series or Series like
    :param alpha: float in [0; 1]
        The level of confidence for the p-values
    :param qq_plot: bool
        If True a QQ plot is displayed
    :return: df_normality: pandas DataFrame
        A summary of the statistical normality tests
    """
    data = data[list_vars]
    rows = []
    for var in data.columns:
        rows.append('statistic_' + var)
        rows.append('p-value_' + var)
        rows.append('Gaussian?_' + var)
    df_normality = pd.DataFrame(index=rows, columns=["D\'Agostino", 'Shapiro-Wilk', 'Kolmogorov-Smirnov'])
    for var in data.columns:
        df_normality.loc['statistic_' + var, "D\'Agostino"], df_normality.loc[
            'p-value_' + var, "D\'Agostino"] = normaltest(data[var])
        df_normality.loc['statistic_' + var, 'Shapiro-Wilk'], df_normality.loc[
            'p-value_' + var, 'Shapiro-Wilk'] = shapiro(data[var])
        df_normality.loc['statistic_' + var, 'Kolmogorov-Smirnov'], df_normality.loc[
            'p-value_' + var, 'Kolmogorov-Smirnov'] = kstest(data[var], 'norm')
        g = np.nan
        for p in df_normality.loc['p-value_' + var, :]:
            if p > alpha:
                g = True
            else:
                g = False
        df_normality.loc['Gaussian?_' + var, :] = g

        if qq_plot:
            qqplot(data, line='45')
            plt.title('QQ-plot')
            plt.show()

    return df_normality


def get_stats(data: pd.DataFrame,
              list_vars=None
              ) -> pd.DataFrame:
    """
    Summarizes count, mean, std, min, 25%, 50% (mean), 75%, max for numeric variables.
    Makes additional summaries for string variables.
    :param data: pandas DataFrame
        The dataframe for the values of which a statistical summary would be made
    :param list_vars: list
        A list to subset the dataframe on. Statistical summary will be provided only for the variables in that list
        If None, a statistical summary is provided for all variables in the dataframe
    :return: df_stats: pandas DataFrame
        The dataframe with the summary stats
    """
    dict_stats = dict()

    data = data[list_vars]

    for var in data.columns:
        dict_stats[var] = data[var].describe()

    return (pd.DataFrame(dict_stats)).T


def plot_distro(data: pd.DataFrame,
                ls_vars: list,
                path_save: str
                ) -> None:
    """
    Plots a distribution of variables from a dataframe
    :param data: pandas DataFrame
    :param ls_vars: list of strings
    :param path_save: string
    :return: None
    """
    if len(ls_vars):
        data = pd.DataFrame(data[ls_vars])

    for col in data.columns:
        plt.hist(data[col], bins=100)
        plt.title(str(col) + ' Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        plt.tight_layout()
        if path_save:
            plt.savefig(path_save + col + '_distro.png', dpi=300)
        plt.show()


def plot_feature_target_scatter(*features: Union[np.array, list],
                                target: str,
                                data: pd.DataFrame
                                ) -> None:
    """
    Plots a scatter plot of the features with respect to the target
    :param features: array-like of strings
    :param target: string
    :param data: pandas DataFrame
    :return: None
    """
    fig, axs = plt.subplots(len(features), figsize=(8, 2 * len(features)))
    for idx in range(len(features)):
        axs[idx].scatter(data[features[idx]], data[target], s=10)
        axs[idx].set_title(str(target) + ' vs ' + str(features[idx]))
        axs[idx].set_facecolor('white')

    for idx, ax in enumerate(axs.flat):
        ax.set(xlabel=features[idx], ylabel=target)

    plt.tight_layout()
    plt.show()


def outliers(data: pd.DataFrame,
             ls_vars: list,
             z_thresh=3.
             ) -> dict:
    """
    Checks for outliers based on z-score
    :param data: pandas DataFrame
        The dataframe containing the variables we want to check for outliers
    :param ls_vars: list of strings
        List of the column names we want to check
    :param z_thresh: float
        The treshold for defining an outlier based on z-score
    :return: dict_pct_outliers: dict
        A dictionary with keys the variable names and values the percentage of outliers for this variable
    """
    dict_pct_outliers = dict()
    for var in ls_vars:
        col = data[var]

        df_out = pd.DataFrame({'values': col.values, 'z-score': zscore(col.values)})
        df_out['is_outlier'] = np.where(abs(zscore(col.values)) > z_thresh, 1, 0)

        dict_pct_outliers[var] = sum(df_out['is_outlier']) / len(df_out['is_outlier'])

    return dict_pct_outliers


def calc_correlation(df: pd.DataFrame,
                     path_save: str,
                     plot=False
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Checks the correlation between the variables of interest
    :param path_save: string
    :param df: pandas DataFrame
    :param plot: bool
    :return: corr, pvals: pandas DataFrames
    """
    corr = df.corr()
    pvals = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(df.columns))

    if plot:
        sns.heatmap(corr, annot=False, cmap='Blues')
        if path_save:
            plt.savefig(path_save, dpi=300)
        plt.show()

    for i, row in enumerate(corr.columns):
        for j, column in enumerate(corr.columns):
            if abs(pvals.iloc[i][column]) < 0.01:
                corr.loc[row, column] = str(np.round(corr.loc[row, column], 4)) + '***'
            elif abs(pvals.iloc[i][column]) < 0.05:
                corr.loc[row, column] = str(np.round(corr.loc[row, column], 4)) + '**'
            elif abs(pvals.iloc[i][column]) < 0.1:
                corr.loc[row, column] = str(np.round(corr.loc[row, column], 4)) + '*'
            else:
                corr.loc[row, column] = str(np.round(corr.loc[row, column], 4))

    return corr, pvals


def get_auc(target: Union[np.array, list],
            preds: Union[np.array, list],
            make_plot=True,
            save_plot=True
            ) -> Tuple[float, float, float]:
    """
    Calculate area under the ROC curve, make a plot of the ROC curve
    :param target: the classification target
    :param preds: the predicted probabilities
    :param make_plot: if True plots the ROC curve
    :param save_plot: if True saves the ROC plot
    :return: auc, fpr, tpr
    """
    auc = roc_auc_score(target, preds)
    fpr, tpr, _ = roc_curve(target, preds)

    if make_plot:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'LGBClassifier, AUC test={np.round(auc, 4)}')
        ax.plot([0, 1], [0, 1], ls='--', color='red')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        plt.title('ROC curve')
        plt.legend()
        plt.show()
        if save_plot:
            plt.savefig('plots/auc_plot.png', dpi=300)

    return auc, fpr, tpr


def calibrate_probas(clf: ClassifierMixin,
                     X_train: pd.DataFrame,
                     y_train: Union[np.array, list],
                     X_test: pd.DataFrame,
                     y_test: Union[np.array, list],
                     ls_features: list,
                     cv='prefit',
                     save_plot=True
                     ) -> Tuple[list, dict]:
    """
    Calibrate a classifier via isotonic and sigmoid calibration, plot a reliability curve.
    Select the best calibration based on Brier score.
    :param clf: the classifier object
    :param X_train: The training features
    :param y_train: The training target
    :param X_test: The test features
    :param y_test: The test target
    :param ls_features: list of features
    :param cv: what kind of cross validation to use for the calibrators
    :param save_plot: if True saves the reliability plot
    :return: the probabilities of the best calibrated classifier; a dictionary with all classifiers
    """
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    dict_calibrations = dict()
    for calibrator, name in [(clf, 'uncalibrated'),
                             (CalibratedClassifierCV(clf, cv=cv, method='isotonic'),
                              'Isotonic Calibration'),
                             (CalibratedClassifierCV(clf, cv=cv, method='sigmoid'),
                              'Sigmoid Calibration')
                             ]:
        calibrator.fit(X_train[ls_features], y_train)
        if hasattr(calibrator, 'predict_proba'):
            prob_pos = calibrator.predict_proba(X_test[ls_features])[:, 1]
        else:  # use decision function
            prob_pos = calibrator.decision_function(X_test[ls_features])
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        # Check Brier scores
        brier = brier_score_loss(y_test, prob_pos)
        dict_calibrations[name] = (brier, prob_pos)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label=f'{name}, Brier score: {np.round(brier, 6)}')

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

    if save_plot:
        plt.savefig('plots/calibration_plot.png')

    print(dict_calibrations)
    # Sort according to brier score, the lower the better
    sorted_dict_calibrations = dict(sorted(dict_calibrations.items(), key=lambda item: item[1][0], reverse=False))
    print(f'\nThe best calibrated classifier is \'{list(sorted_dict_calibrations.keys())[0]}\'')

    return sorted_dict_calibrations[list(sorted_dict_calibrations.keys())[0]][1].tolist(), sorted_dict_calibrations


def get_feature_importances(model: Tuple[LGBMClassifier, LGBMRegressor]) -> pd.DataFrame:
    """
    Extract feature importances of a tree based model.
    Assumes sklearn style API.
    :param model: the model object
    :return: a dataframe with the feature importances as defined in the model object
    """
    importance_type = f'importace ({model.importance_type})'
    df_feature_importances = pd.DataFrame(
        {
            'feature': model.feature_name_,
            f'{importance_type}': model.feature_importances_
        }
    )
    df_feature_importances = df_feature_importances.sort_values(f'{importance_type}', ascending=False)

    return df_feature_importances


def plot_feature_importances(df_importances: pd.DataFrame,
                             top_n=20,
                             save_plot=True,
                             name='clf'
                             ) -> None:
    """
    Plots feature importances extracted by get_feature_importances()
    :param df_importances: the output dataframe of get_feature_importances()
    :param top_n: how many features to be included in the plot
    :param save_plot: if True saves the feature importance plot
    :param name: the name of the feature importance plot
    :return: None
    """
    df_importances.reset_index(inplace=True, drop=True)

    df_importances = df_importances[:20].sort_values(df_importances.columns[1], ascending=True)

    fig, ax = plt.subplots()
    ax.barh(df_importances[df_importances.columns[0]],
            df_importances[df_importances.columns[1]])

    ax.set_xlabel(df_importances.columns[1])

    plt.title(f'Feature Importances (top {top_n} features)')
    plt.tight_layout()
    plt.show()
    if save_plot:
        plt.savefig(f'plots/feature_importances_{name}.png', dpi=300)


def get_expected_value(df: pd.DataFrame,
                       col_proba: str,
                       col_pred_amount: str,
                       col_true_amount: str,
                       ls_costs: list
                       ) -> pd.DataFrame:
    """
    Calculate expected value based on predicted probability of the action and expected outcome
    :param df: input data containing the predicted probas and outcomes
    :param col_proba: the name of the predicted probabilities column
    :param col_pred_amount: the name of the predicted outcome column
    :param col_true_amount: the name of the actual outcome column
    :param ls_costs: list of cost (floats) constraints to calculate the expected value
    :return: a dataframe with expected values for each cost constraint and some EV stats
    """
    df.loc[:, 'expected_donation'] = df[col_proba] * df[col_pred_amount]

    df_evs = pd.DataFrame()
    for idx, cost in enumerate(ls_costs):
        df_positive_ev = df[df['expected_donation'] > cost]
        df_evs.loc[idx, 'Mail Cost'] = cost
        df_evs.loc[idx, 'Total EV'] = sum(df_positive_ev[col_true_amount] - cost)
        df_evs.loc[idx, 'Max. Donation'] = np.max(df_positive_ev[col_true_amount] - cost)
        df_evs.loc[idx, 'Min. Donation'] = np.min(df_positive_ev[col_true_amount] - cost)
        df_evs.loc[idx, 'Std. Donation'] = np.std(df_positive_ev[col_true_amount] - cost)
        df_evs.loc[idx, '# Engagements'] = len(df_positive_ev)
        df_evs.loc[idx, '% Engagements'] = len(df_positive_ev) / len(df) * 100.

    return df_evs


def _decile_table(y_true: Union[np.array, list],
                  y_prob: Union[np.array, list],
                  change_deciles=10,
                  round_decimal=3
                  ) -> pd.DataFrame:
    """
     Generates the Decile Table from labels and probabilities
    The Decile Table is creared by first sorting the samples by their predicted
    probabilities, in decreasing order from highest (closest to one) to
    lowest (closest to zero). Splitting the samples into equally sized segments,
    we create groups containing the same numbers of customers, for example, 10 decile
    groups each containing 10% of the samples.
    :param y_true: the target labels
    :param y_prob: the predicted probabilities
    :param change_deciles: the number of deciles to bin the samples into
    :param round_decimal: how many decimals to use when rounding calculated numbers
    :return: a decile table dataframe
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_prob'] = y_prob

    df.sort_values('y_prob', ascending=False, inplace=True)
    df['decile'] = np.linspace(1, change_deciles + 1, len(df), False, dtype=int)

    dt = df.groupby('decile').apply(lambda x: pd.Series([
        np.min(x['y_prob']),
        np.max(x['y_prob']),
        np.mean(x['y_prob']),
        np.size(x['y_prob']),
        np.sum(x['y_true']),
        np.size(x['y_true'][x['y_true'] == 0]),
    ],
        index=(["prob_min", "prob_max", "prob_avg",
                "cnt_cust", "cnt_resp", "cnt_non_resp"])
    )).reset_index()

    dt['prob_min'] = dt['prob_min'].round(round_decimal)
    dt['prob_max'] = dt['prob_max'].round(round_decimal)
    dt['prob_avg'] = round(dt['prob_avg'], round_decimal)

    tmp = df[['y_true']].sort_values('y_true', ascending=False)
    tmp['decile'] = np.linspace(1, change_deciles + 1, len(tmp), False, dtype=int)

    dt['cnt_resp_rndm'] = np.sum(df['y_true']) / change_deciles
    dt['cnt_resp_wiz'] = tmp.groupby('decile', as_index=False)['y_true'].sum()['y_true']

    dt['resp_rate'] = round(dt['cnt_resp'] * 100 / dt['cnt_cust'], round_decimal)
    dt['cum_cust'] = np.cumsum(dt['cnt_cust'])
    dt['cum_resp'] = np.cumsum(dt['cnt_resp'])
    dt['cum_resp_wiz'] = np.cumsum(dt['cnt_resp_wiz'])
    dt['cum_non_resp'] = np.cumsum(dt['cnt_non_resp'])
    dt['cum_cust_pct'] = round(dt['cum_cust'] * 100 / np.sum(dt['cnt_cust']), round_decimal)
    dt['cum_resp_pct'] = round(dt['cum_resp'] * 100 / np.sum(dt['cnt_resp']), round_decimal)
    dt['cum_resp_pct_wiz'] = round(dt['cum_resp_wiz'] * 100 / np.sum(dt['cnt_resp_wiz']), round_decimal)
    dt['cum_non_resp_pct'] = round(
        dt['cum_non_resp'] * 100 / np.sum(dt['cnt_non_resp']), round_decimal)
    dt['KS'] = round(dt['cum_resp_pct'] - dt['cum_non_resp_pct'], round_decimal)
    dt['lift'] = round(dt['cum_resp_pct'] / dt['cum_cust_pct'], round_decimal)

    return dt


def plot_cumulative_gain(y_true: Union[np.array, list],
                         y_prob: Union[np.array, list],
                         save_plot=True
                         ) -> None:
    """
    Generates the cumulative gain plot from labels and probabilities
    :param y_true: the target labels
    :param y_prob: the predicted probabilities
    :param save_plot: if True, save the cumulative gain plot
    :return: None
    """
    dt = _decile_table(y_true, y_prob)
    fig, ax = plt.subplots()

    ax.plot(np.append(0, dt.decile.values), np.append(0, dt.cum_resp_pct.values), marker='o', label='Model')
    ax.plot(np.append(0, dt.decile.values), np.append(0, dt.cum_resp_pct_wiz.values), '--', label='Perfect Model')
    ax.plot([0, 10], [0, 100], '--', marker='o', label='Random Guess')

    ax.set_xlabel('Deciles')
    ax.set_ylabel('% Responders')

    plt.title('Cumulative Gain Plot')
    plt.legend()
    plt.show()
    if save_plot:
        plt.savefig('plots/cumulative_plot.png')


def plot_lift(y_true: Union[np.array, list],
              y_prob: Union[np.array, list],
              save_plot=True
              ) -> None:
    """
    Generates the lift plot from labels and probabilities
    :param y_true: the target labels
    :param y_prob: the predicted probabilities
    :param save_plot: if True, save the lift plot
    :return: None
    """
    dt = _decile_table(y_true, y_prob)

    fig, ax = plt.subplots()

    ax.plot(dt.decile.values, dt.lift.values, marker='o', label='Model')
    ax.plot([1, 10], [1, 1], '--', marker='o', label='Random Guess')

    ax.set_xlabel('Deciles')
    ax.set_ylabel('Lift')

    plt.title('Lift Plot')
    plt.legend()
    plt.show()
    if save_plot:
        plt.savefig('plots/lift_plot.png')
