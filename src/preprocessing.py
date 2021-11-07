"""
Summary: Preprocess data for feature selection or modelling
TODO: Some methods used here could be made more efficient
"""
import pandas as pd
import numpy as np

from ml_tools import Preprocessor

# Read in the data
df_train = pd.read_csv('input_data/raw_data/train.txt', low_memory=False)
df_val_features = pd.read_csv('input_data/raw_data/validation.txt', low_memory=False)
df_val_targets = pd.read_csv('input_data/raw_data/val_targets.csv', header=0, delimiter=',', low_memory=False)
df_census = pd.read_fwf('input_data/raw_data/cols_census.txt', sep=' ', header=None)
# Combine the datasets before modifying some variables
df_validation = df_val_features.merge(df_val_targets, on=['CONTROLN'])

# Define preprocessor objects for training and validation dataset
pre_train = Preprocessor(df=df_train)
pre_val = Preprocessor(df=df_validation)

# Convert date columns to date type
ls_cols_date = [col for col in df_train if 'DATE' in col]
PRESENT_DATE = pd.to_datetime('9807', format='%y%m').date()  # Date of the KDD 1998 cup
pre_train.convert_dates(ls_col_dates=ls_cols_date)
pre_val.convert_dates(ls_col_dates=ls_cols_date)
# Calculate day differences with the present for some features
ls_cols_day_diff = [
    'ODATEDW',
    'MAXADATE',
    'MINRDATE',
    'MAXRDATE',
    'LASTDATE',
    'FISTDATE'
]
pre_train.calc_day_diff(ls_cols_diff=ls_cols_day_diff,
                        present_date=PRESENT_DATE,
                        drop_col_dates=True)
pre_val.calc_day_diff(ls_cols_diff=ls_cols_day_diff,
                      present_date=PRESENT_DATE,
                      drop_col_dates=True)

# Get the features and extra columns after dropping the date columns
ls_features = [col for col in pre_train.df.columns if col not in ['TARGET_B', 'TARGET_D', 'CONTROLN']]
ls_extra_cols = [col for col in pre_train.df.columns if col not in ls_features]

# Cast object and category variables to category
pre_train.cast_categorical(ls_features=ls_features, ls_types=['category', 'object'])
pre_val.cast_categorical(ls_features=ls_features, ls_types=['category', 'object'])

# Recode some variables according to the data dictionary https://kdd.org/cupfiles/KDDCupData/1998/cup98dic.txt
# Replace '' with 0 and X with 1 for some variables
ls_fill_binary = ['NOEXCH', 'RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP', 'MAJOR', 'PEPSTRFL']
pre_train.replace_strings(ls_columns=ls_fill_binary,
                          str_to_replace=' ',
                          replacement='0',
                          regex=False,
                          astype='category')
pre_train.replace_strings(ls_columns=ls_fill_binary,
                          str_to_replace='X',
                          replacement='1',
                          regex=False,
                          astype='category')

pre_val.replace_strings(ls_columns=ls_fill_binary,
                        str_to_replace=' ',
                        replacement='0',
                        regex=False,
                        astype='category')
pre_val.replace_strings(ls_columns=ls_fill_binary,
                        str_to_replace='X',
                        replacement='1',
                        regex=False,
                        astype='category')

# Replace B with 0 (bad mailcode) and '' with 1 (normal mailcode) for MAILCODE
pre_train.replace_strings(ls_columns=['MAILCODE'],
                          str_to_replace='B',
                          replacement='0',
                          regex=False,
                          astype='category')
pre_train.replace_strings(ls_columns=['MAILCODE'],
                          str_to_replace=' ',
                          replacement='1',
                          regex=False,
                          astype='category')

pre_val.replace_strings(ls_columns=['MAILCODE'],
                        str_to_replace='B',
                        replacement='0',
                        regex=False,
                        astype='category')
pre_val.replace_strings(ls_columns=['MAILCODE'],
                        str_to_replace=' ',
                        replacement='1',
                        regex=False,
                        astype='category')
# Drop samples with bad mailcode, because these people cannot be contacted
pre_train.df = pre_train.df[pre_train.df['MAILCODE'] == '1']
pre_val.df = pre_val.df[pre_val.df['MAILCODE'] == '1']

# SOLP3 - create do not solicit or mail indicator
pre_train.df['cat_solicit'] = np.where(pre_train.df['SOLP3'] == '00', 0, 1).astype(bool)
pre_val.df['cat_solicit'] = np.where(pre_val.df['SOLP3'] == '00', 0, 1).astype(bool)

# GENDER - replace the joint accounts gender with unknown; replace some random categories with uknown for a few samples
pre_train.replace_strings(ls_columns=['GENDER'],
                          str_to_replace='J',
                          replacement='U',
                          regex=False,
                          astype='category')
pre_train.replace_strings(ls_columns=['GENDER'],
                          str_to_replace='C',
                          replacement='U',
                          regex=False,
                          astype='category')
pre_train.replace_strings(ls_columns=['GENDER'],
                          str_to_replace='A',
                          replacement='U',
                          regex=False,
                          astype='category')
pre_train.replace_strings(ls_columns=['GENDER'],
                          str_to_replace=' ',
                          replacement='U',
                          regex=False,
                          astype='category')

pre_val.replace_strings(ls_columns=['GENDER'],
                        str_to_replace='J',
                        replacement='U',
                        regex=False,
                        astype='category')
pre_val.replace_strings(ls_columns=['GENDER'],
                        str_to_replace='C',
                        replacement='U',
                        regex=False,
                        astype='category')
pre_val.replace_strings(ls_columns=['GENDER'],
                        str_to_replace='A',
                        replacement='U',
                        regex=False,
                        astype='category')
pre_val.replace_strings(ls_columns=['GENDER'],
                        str_to_replace=' ',
                        replacement='U',
                        regex=False,
                        astype='category')

# Create features based on information already existing in the bits of other features
# MDMAUD
pre_train.df['is_major_donor'] = np.where(pre_train.df['MDMAUD'] == 'XXXX', 0, 1).astype(bool)
pre_val.df['is_major_donor'] = np.where(pre_val.df['MDMAUD'] == 'XXXX', 0, 1).astype(bool)

pre_train.extract_str_bits(col='MDMAUD', n_bits=3, astype='category')
pre_val.extract_str_bits(col='MDMAUD', n_bits=3, astype='category')
# Drop the column after we extract the needed info
pre_train.df.drop(columns=['MDMAUD'], inplace=True)
pre_val.df.drop(columns=['MDMAUD'], inplace=True)

# DOMAIN
pre_train.replace_strings(ls_columns=['DOMAIN'],
                          str_to_replace=' ',
                          replacement=np.nan,
                          regex=False,
                          astype='str')

pre_val.replace_strings(ls_columns=['DOMAIN'],
                        str_to_replace=' ',
                        replacement=np.nan,
                        regex=False,
                        astype='str')
pre_train.extract_str_bits(col='DOMAIN', n_bits=2, astype='category')
pre_val.extract_str_bits(col='DOMAIN', n_bits=2, astype='category')
# Drop the column after we extract the needed info
pre_train.df.drop(columns=['DOMAIN'], inplace=True)
pre_val.df.drop(columns=['DOMAIN'], inplace=True)

# RFA
ls_cols_rfa = [col for col in ls_features if col.startswith('RFA')]
ls_cols_rfa = [elem for elem in ls_cols_rfa if elem not in ('RFA_2R', 'RFA_2F', 'RFA_2A')]
for col in ls_cols_rfa:
    pre_train.df[col] = pre_train.df[col].astype('str')
    pre_val.df[col] = pre_val.df[col].astype('str')

    pre_train.extract_str_bits(col=col, n_bits=3, astype='category')
    pre_val.extract_str_bits(col=col, n_bits=3, astype='category')
    # Drop the column after we extract the needed info
    pre_train.df.drop(columns=[col], inplace=True)
    pre_val.df.drop(columns=[col], inplace=True)

# Drop 'RFA_2R', 'RFA_2F', 'RFA_2A', as we have already constructed the same variables above
pre_train.df.drop(columns=['RFA_2R', 'RFA_2F', 'RFA_2A'], inplace=True)
pre_val.df.drop(columns=['RFA_2R', 'RFA_2F', 'RFA_2A'], inplace=True)

# Drop all variables characterising the donors neighborhood - if having a different neighbourhood
# predictive the ZIP and MAILCODE variables should contain it
# Also the relationship between the donor and the neighborhood is unclear - they might have not lived there
# at the time of the census
ls_census_cols = list(df_census.iloc[:, 0].values)
pre_train.df.drop(columns=[col for col in ls_census_cols], inplace=True)
pre_val.df.drop(columns=[col for col in ls_census_cols], inplace=True)
# Drop the AGEFLAG variable
pre_train.df.drop(columns=['AGEFLAG'], inplace=True)
pre_val.df.drop(columns=['AGEFLAG'], inplace=True)

# Convert DOB to date
pre_train.convert_dates(ls_col_dates=['DOB'])
pre_val.convert_dates(ls_col_dates=['DOB'])
# Fill AGE with the age calculated from DOB where possible
age_new_train = np.floor((PRESENT_DATE - pre_train.df['DOB']).dt.days / 365)
pre_train.df['AGE'] = pre_train.df['AGE'].fillna(age_new_train)

age_new_val = np.floor((PRESENT_DATE - pre_val.df['DOB']).dt.days / 365)
pre_val.df['AGE'] = pre_val.df['AGE'].fillna(age_new_val)

# Drop the DOB variable
pre_train.df.drop(columns=['DOB'], inplace=True)
pre_val.df.drop(columns=['DOB'], inplace=True)

# Fill in unknown gender via the title
pre_train.suggest_gender_from_title()
pre_val.suggest_gender_from_title()

# Replace the '-' in ZIPCODE
pre_train.replace_strings(ls_columns=['ZIP'],
                          str_to_replace='-',
                          replacement='',
                          regex=False,
                          astype='category')
pre_val.replace_strings(ls_columns=['ZIP'],
                        str_to_replace='-',
                        replacement='',
                        regex=False,
                        astype='category')

# Combine the less common titles into one category
pre_train.group_categories(col='TCODE', size_thresh=0.001, new_category=211, astype='category')
pre_val.group_categories(col='TCODE', size_thresh=0.001, new_category=211, astype='category')

# Save the preprocessed dataframe
pre_train.df.to_csv('input_data/preprocessed_data/df_train_pre.csv', index=False)
pre_val.df.to_csv('input_data/preprocessed_data/df_validation_pre.csv', index=False)
