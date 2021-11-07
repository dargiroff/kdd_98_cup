"""
Summary: Look into the dataset
TODO:
"""
import pandas as pd
from matplotlib import pyplot as plt

from utils import check_missing_values

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

# Read in the data
df_train = pd.read_csv('input_data/raw_data/train.txt')
df_val_features = pd.read_csv('input_data/raw_data/validation.txt')
df_val_targets = pd.read_csv('input_data/raw_data/val_targets.csv', header=0, delimiter=',')

# Check data shape
print(f'shape train: {df_train.shape}')
print(f'shape validation: {df_val_features.shape}')

# Get lists of the feature columns and the additional columns
ls_features = [col for col in df_train.columns if col not in ['TARGET_B', 'TARGET_D', 'CONTROLN']]
assert len(ls_features) == df_train.shape[1] - 3, 'Some features were dropped'
ls_extra_cols = [col for col in df_train.columns if col not in ls_features]

# Check the target(s) distribution
df_train['TARGET_B'].value_counts(normalize=True)

fig, ax = plt.subplots()
ax.hist(df_train.loc[df_train['TARGET_D'] != 0, 'TARGET_D'], bins=50)
ax.set_ylabel('Frequency')
ax.set_xlabel('Donation Amount')
plt.title('Donation amounts frequency (excl. 0)')
plt.show()
plt.savefig('plots/donation_amount_freq.png', dpi=300)

# Check missing values in the feature frame
df_missings_train = check_missing_values(data=df_train[ls_features])
df_missings_train = df_missings_train.loc[df_missings_train['number of missings'] != 0, :]
df_missings_train.to_csv('tables/df_missing_values.csv', index=False)
# 92 variables have some missing values
# For some variables like NUMCHLD we can assume that missing values mean 0 kids and we can fill with 0
# However, many variables have a high amount of missings
# In general, it would be a good idea to use some algo that deals with missings and does not require filling

# Check the distribution of the age variable and how it correlates with the target(s)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].hist(df_train['AGE'], bins=50)
axs[0].set_ylabel('Frequency')
axs[0].set_xlabel('Age')
axs[0].set_title('Age Distribution')

axs[1].scatter(df_train['AGE'], df_train['TARGET_B'])
axs[1].set_xlabel('Age')
axs[1].set_ylabel('Binary Target')
axs[1].set_title('Age vs Binary Target')

axs[2].scatter(df_train['AGE'], df_train['TARGET_D'])
axs[2].set_xlabel('Age')
axs[2].set_ylabel('Donation Amount')
axs[2].set_title('Age vs Donation Amount')

plt.show()
plt.savefig('plots/age_feature.png', dpi=300)

# Check correlation between the features and the target(s)
corr_target_b = df_train.corr()['TARGET_B']
corr_target_b = corr_target_b.sort_values(ascending=False)
# Most of the variables are not strongly correlated with the target B based on OLS
# Therefore a non linear model might be prefered

corr_target_d = df_train.corr()['TARGET_D']
corr_target_d = corr_target_d.sort_values(ascending=False)
# Most of the variables are not strongly correlated with the target D based on OLS
# Therefore a non linear model might be prefered
