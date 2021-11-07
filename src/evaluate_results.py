"""
Summary: Evaluate the results of the predictions; draw final conclusions
TODO:
"""
import pandas as pd

from matplotlib import pyplot as plt

from utils import get_expected_value, plot_cumulative_gain, plot_lift

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

SEED = 1337

# Read in the output datasets
df_test_reg = pd.read_csv('input_data/regressor_output/df_validation.csv')
df_test_clf = pd.read_csv('input_data/classifier_output/df_validation.csv')

# Get lists of the feature columns and the additional columns
ls_features_reg = [col for col in df_test_reg.columns if col not in ['TARGET_B', 'TARGET_D', 'CONTROLN']]
ls_extra_cols_reg = [col for col in df_test_reg.columns if col not in ls_features_reg]

ls_features_clf = [col for col in df_test_clf.columns if col not in ['TARGET_B', 'TARGET_D', 'CONTROLN']]
ls_extra_cols_clf = [col for col in df_test_clf.columns if col not in ls_features_clf]

# Set cost for the mailing
ls_costs = [1, 5, .68]
# Check the Expected Value given the probas of donating and the predicted donation amounts
df_evs = get_expected_value(df=df_test_reg,
                            col_proba='calibrated_proba_donation',
                            col_pred_amount='predicted_donation',
                            col_true_amount='TARGET_D',
                            ls_costs=ls_costs)
# Save the resulting EV table
df_evs.to_csv('tables/df_expected_values.csv', index=False)

# Check cumulative gain and lift for the binary classifier
plot_cumulative_gain(y_true=df_test_clf['TARGET_B'], y_prob=df_test_clf['calibrated_proba_donation'])
plot_lift(y_true=df_test_clf['TARGET_B'], y_prob=df_test_clf['calibrated_proba_donation'])
