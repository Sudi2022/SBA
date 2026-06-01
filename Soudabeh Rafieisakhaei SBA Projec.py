#!/usr/bin/env python
# coding: utf-8
                          
                           +-------------------------------------------------------- +
                           |          “Should This Loan be Approved or Denied?”      |
                           |        Predictive Modeling Using the SBA National Data  |
                           |                    Soudabeh Rafieisakhaei               |
                           |                                                         |
                           +---------------------------------------------------------+
                                                  
                                                 Project Architecture               
                           +--------------------------------------------------------+
                                                       Data                          
                                         +-----------------------------+            
                                         |        SBA National Data    |            
                                         +-----------------------------+            
                                                        |                            
                                                        v                            
                            +---------------------------+------------------------+ 
                            |    Data Preprocessing     | Feature Engineering     |
                            |   - Handle Missing Data   |   - Variable Selection  | 
                            |   - Feature Scaling       |   - Interaction Terms   | 
                            |   - Categorical Encoding  |                         |
                            +---------------------------+------------------------+ 
                                                        |                            
                                                        v                            
                                          +-----------------------------+                      
                                          |     Train-Test Split        |                       
                                          +-----------------------------+                       
                                                        |                            
                                                        v                            
                                          +-----------------------------+                      
                                          | Feature Selection/          |                       
                                          | Engineering                 |                       
                                          +-----------------------------+                       
                                                        |                            
                                                        v                            
                                          +-----------------------------+                       
                                          | Model Development           |                       
                                          |   - kNN                     |                       
                                          |      - Fit Model            |                       
                                          |      - Evaluate             |                       
                                          |   - Classification Trees    |                       
                                          |      - Fit Model            |                       
                                          |      - Evaluate             |                       
                                          |   - Logit Model             |                       
                                          |      - Fit Model (Lasso,    |                      
                                          |        Ridge, ElasticNet)   |                       
                                          |      - Evaluate             |                       
                                          |   - Neural Networks         |                       
                                          |      - Build Model          |                       
                                          |      - Evaluate             |                       
                                          |   - Discriminant Analysis   |                      
                                          |      - Fit Model            |                       
                                          |      - Evaluate             |                       
                                          +-----------------------------+                       
                                                         |                            
                                                         v                            
                                          +-----------------------------+                       
                                          | Model Evaluation and        |                       
                                          | Selection                   |                       
                                          |   - Confusion Matrix        |                       
                                          |   - Cost/Gain Matrix        |                       
                                          +-----------------------------+                       
                                                         |                           
                                                         v                            
                                          +-----------------------------+                       
                                          | Hyperparameter Tuning       |                      
                                          |   - Grid Search/Random      |                       
                                          |     Search                  |                      
                                          +-----------------------------+                      
                                                         |                            
                                                         v                           
                                          +-----------------------------+                      
                                          | Final Model Evaluation      |                       
                                          |   - Net Profit Calculation  |                       
                                          +-----------------------------+                      
                                                         |                           
                                                         v                           
                                          +-----------------------------+                      
                                          | Interpretation and          |                       
                                          | Documentation               |                      
                                          +-----------------------------+                      
                                                         |                           
                                                         v                          
                                          +-----------------------------+                       
                                          | Gains and Lift Charts       |                       
                                          |   - Calculate Gains         |                       
                                          +-----------------------------+                       
                                                         |                            
                                                         v                            
                                          +-----------------------------+                       
                                          | Comparision of the models   |                       
                                          |   - sorting by accuracy     |
                                          |     in ascending order      |
                                          +-----------------------------+                       
                                                         |                            
                                                         v                            
                                          +-----------------------------+                       
                                          | Report Generation           |                       
                                          +-----------------------------+                       


# In[30]:


# Load Libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
#matplotlib.use('Agg')
import pandas as pd
from dmba import liftChart, gainsChart, classificationSummary, regressionSummary, plotDecisionTree
from dmba.metric import AIC_score
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.feature_selection import RFE
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import seaborn as sns
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mord import LogisticIT
from scipy.stats import randint, uniform
import cProfile
import pstats
import math
import warnings
import matplotlib.pyplot as plt


# ## 1. Data Exploration and Preprocessing
# How is the outcome variable MIS_Status distributed? Identify predictors that may help predict MIS_Status using descriptive statistics and visualization.

# In[31]:


Loan_df = pd.read_csv('/Users/soudabeh/desktop/SBA case Study /SBAnational.csv', low_memory=False)

Loan_df.head(3)


# In[32]:


Loan_df['MIS_Status'].value_counts()


# In[33]:


Loan_df.info()


# ## Handle Missing Data

# In[34]:


Loan_df.isna().sum()


# In[35]:


Loan_df = Loan_df

# Columns with missing values
columns_with_missing_values = ['LoanNr_ChkDgt', 'Name', 'City', 'State', 'Zip', 'Bank', 'BankState', 'NAICS',
                                'ApprovalDate', 'ApprovalFY', 'FranchiseCode']

# Drop rows with missing values in specified columns
Loan_df.dropna(subset=columns_with_missing_values, inplace=True)


# In[36]:


# Binary columns suitable for filling with mode
binary_columns_to_fill = ['NewExist', 'RevLineCr', 'LowDoc', 'UrbanRural', 'MIS_Status']

# Fill missing values with mode for binary columns
Loan_df[binary_columns_to_fill] = Loan_df[binary_columns_to_fill].apply(lambda x: x.fillna(x.mode()[0]))


# In[37]:


Loan_df.isna().sum()


# In[38]:


# Forward fill missing values in 'ChgOffDate'
# If a loan is declared in default, it's likely that this event occurred at a specific point in time,
#and forward filling would make more sense. 

# Forward fill missing values in 'ChgOffDate'
Loan_df['ChgOffDate'].ffill(inplace=True)

# Backward fill missing values in 'DisbursementDate'
Loan_df['DisbursementDate'].bfill(inplace=True)


# In[39]:


Loan_df.isna().sum()


# In[40]:


# Drop rows with missing values in 'ChgOffDate'
Loan_df.dropna(subset=['ChgOffDate'], inplace=True)

# Verify that there are no more missing values
print(Loan_df.isna().sum())


# ## Remove currency symbols
# 
# Currency symbols and commas removed and then convert to float.
# The affected variable include:
# 
# SBA_Appv, GrAppv, BalanceGross, ChgOffPrinGr, DisbursementGross
# 

# In[41]:


# Convert 'SBA_Appv' and 'GrAppv' to numeric
# Remove currency symbols and commas, and then convert to float

Loan_df['DisbursementGross'] = Loan_df['DisbursementGross'].replace('[\$,]', '', regex=True).astype(float)
Loan_df['BalanceGross'] = Loan_df['BalanceGross'].replace('[\$,]', '', regex=True).astype(float)
Loan_df['ChgOffPrinGr'] = Loan_df['ChgOffPrinGr'].replace('[\$,]', '', regex=True).astype(float)
Loan_df['GrAppv'] = Loan_df['GrAppv'].replace('[\$,]', '', regex=True).astype(float)
Loan_df['SBA_Appv'] = Loan_df['SBA_Appv'].replace('[\$,]', '', regex=True).astype(float)


# ## Categorical Encoding and Adding New variable  
# 
# New, Portion, RealState, Recession, Default

# ### New
# 
# New versus Established Businesses

# In[42]:


# New variable based on 'NewExist'
Loan_df = Loan_df
Loan_df['New'] = Loan_df['NewExist'].map({2: 1, 1: 0})

# Check the newly created variable
Loan_df[['New', 'NewExist']].head(6)


# ### Portion
# 
# SBA’s Guaranteed Portion of Approved Loan

# In[43]:


# Create a new variable 'Portion'
Loan_df['Portion'] = round(Loan_df['SBA_Appv'] / Loan_df['GrAppv'],2)

# Check the newly created variable
Loan_df[['SBA_Appv', 'GrAppv', 'Portion']].head(5)


# ### RealEstate
# 
# Loans Backed by Real Estate

# In[44]:


# Create a new variable 'RealEstate' based on the condition
# Since the term of the loan is a function of the expected life- time of the assets,
# loans backed by real estate will have terms 20 years or greater (!240 months)

Loan_df['RealEstate'] = Loan_df['Term'].apply(lambda x: 1 if x >= 240 else 0)

# Check the newly created variable
Loan_df[['Term', 'RealEstate']].head()


# ### Recession 
# 
# Economic Recession
# 
# “Recession” = 1 if the loans were active6 during the Great Recession (December 2007 to June 2009), and “Recession” = 0 for all other times.
# 
# 

# In[45]:


# Convert 'DisbursementDate' to datetime format
Loan_df['DisbursementDate'] = pd.to_datetime(Loan_df['DisbursementDate'], format='%d-%b-%y')

# Extract year, month, and day into separate columns
Loan_df['Year'] = Loan_df['DisbursementDate'].dt.year
Loan_df['Month'] = Loan_df['DisbursementDate'].dt.month
Loan_df['Day'] = Loan_df['DisbursementDate'].dt.day

# Check conditions and create a new column based on the specified criteria
Loan_df['Recession'] = 0  # Initialize with 0
Loan_df.loc[(Loan_df['Year'].isin([2007, 2008, 2009])) & 
            ((Loan_df['Year'] == 2007) & (Loan_df['Month'] == 12) |
             (Loan_df['Year'] == 2008) |
             (Loan_df['Year'] == 2009) & Loan_df['Month'].isin(range(1, 7))),
            'Recession'] = 1

# Display the result
Loan_df[['Term', 'Year', 'Month', 'Day', 'Recession']].head (5)


# In[46]:


Loan_df[Loan_df['Recession'] == 1][['Term', 'Year', 'Month', 'Day', 'Recession']].head(5)


# ### Default
# 
# The value for “Default” = 1 if MIS_Status = CHGOFF, and “Default” = 0 if MIS_Status = PIF.

# In[47]:


Loan_df['Default'] = Loan_df['MIS_Status'].map({'P I F': 0, 'CHGOFF': 1})

Loan_df['Default'].value_counts()


# In[48]:


print(Loan_df.iloc[6, :])


# In[49]:


columns = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'DisbursementGross', 'BalanceGross', 'ChgOffPrinGr',
           'GrAppv', 'SBA_Appv', 'Portion']

df = Loan_df[columns]

df.describe().round(2)


# ## State default rates 

# In[50]:


# Group by 'State' and calculate the total number of loans and defaulted loans
state_loan_counts = Loan_df.groupby('State')['Default'].agg(['count', 'sum'])

# Calculate the default rate as the ratio of defaulted loans to total loans
default_rate = (state_loan_counts['sum'] / state_loan_counts['count']) * 100

# Round the default rate to the nearest integer
state_loan_counts['Default_Rate'] = default_rate.round(0).astype(int)

# Format the 'Default_Rate' column as a percentage with the sign
state_loan_counts['Default_Rate'] = state_loan_counts['Default_Rate'].apply(lambda x: f"{x:.2f}%" if x % 1 != 0 else f"{x:.0f}%")

# Print the pivot table showing the default rates by state
state_loan_counts

# Create a new DataFrame with the results
data_for_map = pd.DataFrame({
    'State': state_loan_counts.index,
    'Default_Rate': state_loan_counts['Default_Rate'].values
})

state_loan_counts


# In[51]:


# Create a new DataFrame with the results
data_for_map = pd.DataFrame({
    'State': state_loan_counts.index,
    'Default_Rate': state_loan_counts['Default_Rate'].values
})

# Convert the 'Default_Rate' column to numeric
data_for_map['Default_Rate'] = data_for_map['Default_Rate'].replace({'%': ''}, regex=True).astype(float)

# Sort the DataFrame in ascending order by 'Default_Rate'
data_for_map = data_for_map.sort_values(by='Default_Rate', ascending=True)


# In[52]:


import geopandas as gpd 
import os 

df = gpd.read_file('/Users/soudabeh/desktop/SBA case Study /tl_2020_us_state/tl_2020_us_state.shp')

# Filtering out non-continental states
non_continental = ['HI','VI','MP','GU','AK','AS','PR']
us49 = df
for n in non_continental:
    us49 = us49[us49.STUSPS != n]
#us49.columns


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Set the variable that will call whatever column you want to visualize on the map
variable = 'Default_Rate'

# Set the range for the choropleth
vmin, vmax = 7, 27

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(20, 14))  # Main subplot for map

# Assuming the index is used for joining
merged = us49.merge(data_for_map, left_on='STUSPS', right_on='State')

# Plot the specified column as a choropleth map using the 'cool' colormap
merged.plot(column='Default_Rate', cmap='bwr', linewidth=0.8, ax=ax, edgecolor='0.8', vmin=vmin, vmax=vmax, legend=True,legend_kwds={'shrink': 0.6, 'ticks': range(7, 28)})

# Customize the plot
ax.set_title(f'States Shaded by Default Rate of Loans Disbursed before 2011', fontdict={'fontsize': '20', 'fontweight': '6', 'color': 'darkblue'})
ax.set_axis_off()

# Add labels of state names on the map
for _, row in merged.iterrows():
    state_name = row['State']  
    plt.annotate(text=state_name, xy=(row.geometry.centroid.x, row.geometry.centroid.y - 0.5), color='darkblue', fontsize=10, ha='center')

# Display the plot
plt.show()


# In[54]:


# Create a new DataFrame with the results
data_for_map2 = pd.DataFrame({
    'State': state_loan_counts.index,
    'Default_Rate': state_loan_counts['Default_Rate'].values
})

# Set the variable that will call whatever column you want to visualize on the map
variable = 'Default_Rate'

# Set the range for the choropleth
vmin, vmax = 1, 28

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(20, 16))  # Main subplot for map

# Assuming the index is used for joining
merged = us49.merge(data_for_map2, left_on='STUSPS', right_on='State')

# Sort the DataFrame in descending order based on the variable column
#merged = merged.sort_values(by=variable, ascending=False)

# Plot the specified column as a choropleth map
merged.plot(column=variable, cmap='tab10', linewidth=0.8, ax=ax, edgecolor='0.8', vmin=vmin, vmax=vmax, legend=False)

# Customize the plot
ax.set_title(f'States labeled by Default Rate of Loans Disbursed before 2011', fontdict={'fontsize': '20', 'fontweight': 'bold'})
ax.set_axis_off()

# Add labels of percentages on the states
for idx, row in merged.iterrows():
    plt.annotate(text=f"{row[variable]}", xy=(row.geometry.centroid.x, row.geometry.centroid.y), color='darkblue', fontsize=10 ,fontweight='bold', ha='center')

# Display the plot
plt.show()


# In[55]:


# Create a copy of state_loan_counts
state_loan_counts2 = state_loan_counts.copy()

# Convert the 'Default_Rate' column to numeric
state_loan_counts2['Default_Rate'] = state_loan_counts2['Default_Rate'].replace({'%': ''}, regex=True).astype(float)

# Sort the DataFrame in ascending order by 'Default_Rate'
state_loan_counts_sorted = state_loan_counts2.sort_values(by='Default_Rate', ascending=True)

# Define a sequential color map
colors = sns.color_palette("bwr", n_colors=len(state_loan_counts_sorted))

plt.figure(figsize=(14, 6))  # Adjust figure size

# Create a bar plot for the default rate per state
ax = sns.barplot(x=state_loan_counts_sorted.index, y='Default_Rate', data=state_loan_counts_sorted, palette=colors)


ax.set_title(f'Default Rates per State, including non_continental', fontdict={'fontsize': '17'})
ax.set_xlabel(f'State',fontdict={'fontsize': '15'})
ax.set_ylabel('Default Rate',fontdict={'fontsize': '15'})

plt.tight_layout()
plt.show()



# ## Industry default rates. a
# 
# six digit NAICS codes

# In[56]:


# Group by 'State' and calculate the total number of loans and defaulted loans
industry_loan_counts = Loan_df.groupby('NAICS')['Default'].agg(['count', 'sum'])

# Calculate the default rate as the ratio of defaulted loans to total loans
default_rate = (industry_loan_counts['sum'] / industry_loan_counts['count']) * 100

# Round the default rate to the nearest integer
industry_loan_counts['Default_Rate'] = default_rate.round(0).astype(int)

# Format the 'Default_Rate' column as a percentage with the sign
industry_loan_counts['Default_Rate'] = industry_loan_counts['Default_Rate'].apply(lambda x: f"{x:.2f}%" if x % 1 != 0 else f"{x:.0f}%")

# Print the pivot table showing the default rates by state
industry_loan_counts


# ## Industry default rates.b
# 
# Extracting the first two digits of the NAICS column, and adding the description

# In[57]:


#import pandas as pd

# Extract the first two digits of the NAICS codes
Loan_df['NAICS_first_two_digits'] = Loan_df['NAICS'].astype(str).str[:2]

# Define the industry descriptions
industry_descriptions = {
    '21': 'Mining, quarrying, and oil and gas extraction',
    '11': 'Agriculture, forestry, fishing and hunting',
    '55': 'Management of companies and enterprises',
    '62': 'Health care and social assistance',
    '22': 'Utilities',
    '92': 'Public administration',
    '54': 'Professional, scientific, and technical services',
    '42': 'Wholesale trade',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '81': 'Other services (except public administration)',
    '71': 'Arts, entertainment, and recreation',
    '72': 'Accommodation and food services',
    '44': 'Retail trade',
    '45': 'Retail trade',
    '23': 'Construction',
    '56': 'Administrative/support & waste management/remediation Service',
    '61': 'Educational services',
    '51': 'Information',
    '48': 'Transportation and warehousing',
    '49': 'Transportation and warehousing',
    '52': 'Finance and insurance',
    '53': 'Real estate and rental and leasing'
}

# Group by 'NAICS_first_two_digits' and calculate the total number of loans and defaulted loans
industry_loan_counts = Loan_df.groupby('NAICS_first_two_digits')['Default'].agg(['count', 'sum'])

# Calculate the default rate as the ratio of defaulted loans to total loans
industry_loan_counts['Default_Rate'] = (industry_loan_counts['sum'] / industry_loan_counts['count']) * 100

# Round the default rate to the nearest integer
industry_loan_counts['Default_Rate'] = industry_loan_counts['Default_Rate'].round(0).fillna(0).astype(int)

# Reset index to make 'NAICS_first_two_digits' a column
industry_loan_counts.reset_index(inplace=True)

# Create a DataFrame with default rates and industry descriptions
industry_default_rates = industry_loan_counts.copy()

industry_default_rates['Description'] = industry_default_rates['NAICS_first_two_digits'].map(industry_descriptions)

# Format the 'Default_Rate' column as a percentage with the sign for non-NaN values
industry_default_rates['Default_Rate'] = industry_default_rates['Default_Rate'].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else 'NaN')

# Reorder the columns
industry_default_rates = industry_default_rates[['NAICS_first_two_digits', 'Description', 'Default_Rate']]

# Rename columns
industry_default_rates.columns = ['NAICS_first_two_digits', 'Description', 'Default rate (%)']

# Print the table
industry_default_rates


# ### Gross SBA and Average SBA Loan Disbursements by Industry from 1984-2010

# In[58]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Extract the first two digits of the NAICS codes
Loan_df['2_digit_NAICS'] = Loan_df['NAICS'].astype(str).str[:2]

# Group by Industry and calculate sum and mean of DisbursementGross
df_industrySum = Loan_df.groupby(['2_digit_NAICS'])['DisbursementGross'].sum().sort_values(ascending=False) / 1e9
df_industryAve = Loan_df.groupby(['2_digit_NAICS'])['DisbursementGross'].mean().sort_values(ascending=False)

# Create a figure and axes for subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Plot the first subplot
ax1.bar(df_industrySum.index, df_industrySum,color='blue')
ax1.set_xticks(range(len(df_industrySum.index)))
ax1.set_xticklabels([industry_descriptions.get(industry_code, 'Unknown') for industry_code in df_industrySum.index], rotation=30, ha='right', fontsize=10,color='blue')
ax1.set_title('Gross SBA Loan Disbursement by Industry from 1984-2010', fontsize=14)
ax1.set_xlabel('Industry', fontsize=10)
ax1.set_ylabel('Gross Loan Disbursement (Billions)',fontsize=12, fontweight = 'bold')

# Plot the second subplot
ax2.bar(df_industryAve.index, df_industryAve, color='blue')
ax2.set_xticks(range(len(df_industryAve.index)))
ax2.set_xticklabels([industry_descriptions.get(industry_code, 'Unknown') for industry_code in df_industryAve.index], rotation=30, ha='right', fontsize=10,color='blue')
ax2.set_title('Average SBA Loan Disbursement by Industry from 1984-2010', fontsize=14)
ax2.set_xlabel('Industry', fontsize=10)
ax2.set_ylabel('Average Loan Disbursement', fontsize=12, fontweight = 'bold')

plt.tight_layout()
plt.show()


# ### Loans backed by real estate

# In[59]:


# Create a DataFrame with the relevant data
data = {
    'Default': [
        len(Loan_df[(Loan_df['MIS_Status'] == 'CHGOFF') & (Loan_df['RealEstate'] == 1) & (Loan_df['Term'] >= 240)]),
        len(Loan_df[(Loan_df['MIS_Status'] == 'CHGOFF') & (Loan_df['RealEstate'] == 0) & (Loan_df['Term'] < 240)])
    ],
    'Paid in full': [
        len(Loan_df[(Loan_df['MIS_Status'] == 'P I F') & (Loan_df['RealEstate'] == 1) & (Loan_df['Term'] >= 240)]),
        len(Loan_df[(Loan_df['MIS_Status'] == 'P I F') & (Loan_df['RealEstate'] == 0) & (Loan_df['Term'] < 240)])
    ]
}

# Create a DataFrame with the calculated values
df = pd.DataFrame(data, index=['Loans Back by Real Estate(Term ≥240 months)', 'Loans Not Backed by Real Estate (Term <240 months)'])

# Calculate percentages and format the values
total = len(Loan_df)
df['Default'] = df['Default'].apply(lambda x: f"{x} ({(x / total * 100):.2f}%)")
df['Paid in full'] = df['Paid in full'].apply(lambda x: f"{x} ({(x / total * 100):.2f}%)")

# Print the DataFrame
df


# ## Quartiles of gross disbursement

# In[60]:


# Calculate the quartiles for 'ChgOffPrinGr' and 'GrAppv'
chgoff_quartiles = Loan_df['GrAppv'].quantile([0, 0.25, 0.5, 0.75, 1])
grappv_quartiles = Loan_df['DisbursementGross'].quantile([0, 0.25, 0.5, 0.75, 1])

# Create a DataFrame to store the quartiles data
data = {
    'Quartiles': ['Minimum', '25% quartile', '50% median', '75% quartile', '100% maximum'],
    'CHGOFF': chgoff_quartiles.values,
    'PIF': grappv_quartiles.values
}

quartiles_df = pd.DataFrame(data)

# Format the currency columns to show as dollars with commas
quartiles_df['CHGOFF'] = quartiles_df['CHGOFF'].map('${:,.2f}'.format)
quartiles_df['PIF'] = quartiles_df['PIF'].map('${:,.2f}'.format)

# Print the DataFrame
quartiles_df


# ### Status of the loans active or not active during the Great Recession.

# In[61]:


import matplotlib.pyplot as plt

# Calculate the total number of loans for each recession status
total_loans = Loan_df.groupby('Recession').size()

# Count the number of loans for each recession status and MIS_Status
loan_status_counts = Loan_df.groupby(['Recession', 'MIS_Status']).size().unstack()

# Calculate the percentage of loans for each loan status and recession status
loan_status_percentages = loan_status_counts.div(total_loans, axis=0) * 100

# Plot the horizontal bar chart with red and dark blue colors
ax = loan_status_percentages.plot(kind='barh', stacked=True, figsize=(10, 3), color=['red', 'blue'])
ax.set_ylabel('Loan Status')
ax.set_xlabel('Percentage of Loans')
ax.set_title('Status of Loans Active or Not Active during the Great Recession')
ax.legend(title='Recession Status', labels=['Default', 'Paid in Full'], loc='upper right')

# Set y-axis ticks and labels in reverse order
ax.set_yticks([0, 1])
ax.set_yticklabels(['Not Active', 'Active'])

# Add percentage labels to the bars
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f'{width:.2f}%', xy=(width, p.get_y() + p.get_height() / 2), xytext=(3, 0), textcoords="offset points", ha='left', va='center',color='white')

plt.tight_layout()
plt.show()


# ### Bar charts of categorical predictors 

# In[62]:


import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore specific deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Categorical predictors and the outcome variable
categorical_predictors = ['NewExist', 'UrbanRural', 'RevLineCr', 'LowDoc']
y_variable = 'MIS_Status'

# Plotting bar charts for each categorical predictor
plt.figure(figsize=(14, 10))

for i, predictor in enumerate(categorical_predictors, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=predictor, hue=y_variable, data=Loan_df, palette=['red', 'blue'])
    plt.title(f'{predictor} vs. {y_variable}')
    plt.xlabel(predictor)
    plt.ylabel('Count')
    plt.legend(title=y_variable)

plt.tight_layout()
plt.show()


# ### Chi-Square Test of Independence for a set of categorical predictors against the target variable 'Default

# In[63]:


from scipy.stats import chi2_contingency

# Selected predictors
predictors = ['New', 'RealEstate', 'RevLineCr', 'LowDoc', 'UrbanRural', 'Recession']

# Create an empty list to store the results
results = []

# Perform Chi-Square Test for each selected predictor
for predictor in predictors:
    contingency_table = pd.crosstab(Loan_df['Default'], Loan_df[predictor])
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    
    # Calculate standard error
    se = np.sqrt(chi2 / (contingency_table.sum().sum() * min(contingency_table.shape) * dof))
    
    # Append rounded values to the results list
    results.append((predictor, contingency_table.shape[1] - 1, round(chi2, 4), round(p, 4), round(se, 4)))

# Create the results DataFrame
results_df = pd.DataFrame(results, columns=['Source', 'DF', 'Chi-Square', 'Pr > ChiSq','Standard Error'])

# Print the results DataFrame
results_df



# ### Reature ranking using Recursive Feature Elimination

# In[64]:


# Define target variable and numerical features
y = Loan_df['MIS_Status']
numerical_features = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob','ChgOffPrinGr', 'DisbursementGross', 'BalanceGross', 'GrAppv', 'SBA_Appv']

# Create feature matrix
X = Loan_df[numerical_features]

# Logistic Regression model with increased max_iter
logreg_model = LogisticRegression(max_iter=1000)

# Recursive Feature Elimination (RFE) for feature selection
rfe = RFE(logreg_model, n_features_to_select=1).fit(X, y)

# Get the ranking of features
feature_ranking = pd.DataFrame({'Feature': numerical_features, 'Ranking': rfe.ranking_}).sort_values(by='Ranking')

# Display the ranked features
feature_ranking


# ### Random Forest Classifier Implementation

# In[43]:


# Convert the 'Default' variable into a categorical variable
Loan_df['Default'] = pd.Categorical(Loan_df['Default'])

predictors = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'NewExist',
              'UrbanRural', 'DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv',
              'RealEstate', 'Portion', 'Recession']

# Select predictors and target variable
X = Loan_df[predictors]
y = Loan_df['Default']

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

rf = RandomForestClassifier(n_estimators=500, random_state=1)
rf.fit(train_X, train_y)

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

df = pd.DataFrame({'feature': train_X.columns, 'importance': importances, 'std': std})
df['importance'] = df['importance'].round(4)  # Round importance to four digits
df['std'] = df['std'].round(4)  # Round standard deviation to four digits
df = df.sort_values('importance', ascending=False)

classificationSummary(valid_y, rf.predict(valid_X))

df


# In[44]:


df2 = df.head(20)
df2_sorted = df2.sort_values(by='importance', ascending=True)
ax = df2_sorted.plot(kind='barh', xerr='std', x='feature', color='blue', legend=False)
ax.set_ylabel('')

plt.tight_layout()
plt.show()


# ## KNN

# In[65]:


# Create a new DataFrame (df2) containing only the selected predictors and the target variable ('MIS_Status').
df_knn = Loan_df[['Term','NoEmp','RetainedJob','New',
               'CreateJob','UrbanRural','DisbursementGross',
               'ChgOffPrinGr','GrAppv','SBA_Appv',
               'RealEstate','Portion','Recession','Default']]
# Drop rows with null values from the dataset to ensure a complete dataset for modeling.
#df2.dropna(inplace=True)

# Separate the features (X) and the target variable (y) from the cleaned dataset.
X = df_knn[['Term','NoEmp','RetainedJob','New',
               'CreateJob','UrbanRural','DisbursementGross',
               'ChgOffPrinGr','GrAppv','SBA_Appv',
               'RealEstate','Portion','Recession']]
y = df_knn['Default']


# In[66]:


# Specify categorical columns
categorical_columns = ['New','UrbanRural','RealEstate','Recession']

# Convert specified columns to 'category' data type and apply get_dummies
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y, test_size=0.2, random_state=1)
print("Shape of X_train:", X_train.shape)
print("Shape of X_valid:", X_valid.shape)


# In[67]:


# Standardize the features using StandardScaler:
# - Fit the scaler on the training set (X_train) and transform it.
# - Transform the validation set (X_valid) using the scaler.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Check the shape of the scaled validation set to ensure consistency.
scaled_validation_shape = X_valid_scaled.shape


# In[69]:


# Sample a subset of the data
X_sample, _, y_sample, _ = train_test_split(X_train_scaled, y_train, test_size=0.8, random_state=1)

# Determine the number of components for 95% variance
pca = PCA(n_components=0.95)
pca.fit(X_sample)
n_components = pca.n_components_

# Apply Incremental PCA on the sample
ipca_sample = IncrementalPCA(n_components=n_components)
n_batches_sample = min(50, len(X_sample))  # Adjust based on your subset size and memory constraints

# Fit IPCA on the sampled dataset in batches
for X_batch in np.array_split(X_sample, n_batches_sample):
    ipca_sample.partial_fit(X_batch)

# Transform the sampled data using the fitted IPCA
X_train_scaled_reduced_sample = ipca_sample.transform(X_sample)

# Randomized Search on the Sample
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}

random_search_sample = RandomizedSearchCV(knn, param_grid, n_iter=5, cv=5, n_jobs=-1, verbose=1, random_state=1)
random_search_sample.fit(X_train_scaled_reduced_sample, y_sample)

# Get the best hyperparameters
best_params = random_search_sample.best_params_
print(best_params)


# In[70]:


# Extract best n_neighbors value
best_n_neighbors = best_params['n_neighbors']

# Initialize KNeighborsClassifier with the best n_neighbors
knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, n_jobs=-1)

# Fit the model on the full training set
knn_model.fit(X_train_scaled, y_train)

# Predict on the validation set
knn_predictions = knn_model.predict(X_valid_scaled)

# Evaluate the model
print(classificationSummary(y_valid, knn_predictions))


# In[72]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Define costs
cost_fp = -5 * 0.05
cost_tp = 0.05


# Fit your KNN model (assuming knn_model is already defined)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)

# Compute probabilities once
probabilities = knn_model.predict_proba(X_valid_scaled)[:, 1]

# Define cutoffs and initialize arrays
cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = np.zeros_like(cutoffs)
specificity_scores = np.zeros_like(cutoffs)
youden_scores = np.zeros_like(cutoffs)
net_profit = np.zeros_like(cutoffs)
accuracy_scores = np.zeros_like(cutoffs)


# Vectorized computation for each cutoff
for i, cutoff in enumerate(cutoffs):
    
    y_pred = (probabilities > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    sensitivity_scores[i] = sensitivity
    specificity_scores[i] = specificity
    youden_scores[i] = sensitivity + specificity - 1
    net_profit[i] = cost_fp * conf_matrix[0, 1] + cost_tp * conf_matrix[0, 0]

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    accuracy_scores[i] = accuracy

    # Print results
    print(f"Cutoff: {cutoff:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Accuracy: {accuracy:.2%}, Net Profit: {net_profit[i]:.2f}")

# Filter for total cost > 0 using NumPy arrays
mask = net_profit > 0
filtered_cutoffs = cutoffs[mask]
filtered_sensitivity_scores = sensitivity_scores[mask]
filtered_specificity_scores = specificity_scores[mask]
filtered_net_profit = net_profit[mask]
filtered_accuracy_scores = accuracy_scores[mask]

# Plot the results
plt.plot(cutoffs, sensitivity_scores, label='Sensitivity', marker='o')
plt.plot(cutoffs, specificity_scores, label='Specificity', marker='o')
plt.plot(cutoffs, accuracy_scores, label='Accuracy', marker='o')
plt.title('Sensitivity, Specificity, and Accuracy vs. Cutoff Value')
plt.xlabel('Cutoff Value')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot the results for net profit (where net profit is greater than 0)
plt.plot(cutoffs, net_profit, label="Net Profit", marker='o', color='red')
plt.title("Net Profit vs. Cutoff Value")
plt.xlabel('Cutoff Value')
plt.ylabel("Net Profit")
plt.legend()
plt.show()

# Print the best cutoff and corresponding net profit
best_index = np.argmax(filtered_net_profit)
best_cutoff = filtered_cutoffs[best_index]
best_net_profit = filtered_net_profit[best_index]
print(f"Best Cutoff: {best_cutoff}")
print(f"Corresponding Net Profit: {best_net_profit:.2f}")


# In[73]:


from sklearn.preprocessing import LabelEncoder

# Fit your KNN model 
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)

# Predict probabilities on the validation set
probabilities = knn_model.predict_proba(X_valid_scaled)[:, 1]

# Convert string labels to integers using LabelEncoder
label_encoder = LabelEncoder()
y_valid_encoded = label_encoder.fit_transform(y_valid)

# Predict using probabilities and best_cutoff
knn_predictions = (probabilities > best_cutoff).astype(int)

# Compute confusion matrix
conf_matrix_knn = confusion_matrix(y_valid_encoded, knn_predictions)

# Calculate net profit
cost_fp = -5 * 0.05  # Cost of false positive
cost_tp = 0.05  # Cost of true positive

net_profit_knn = (conf_matrix_knn[0, 1] * cost_fp) + (conf_matrix_knn[0, 0] * cost_tp)

# Calculate accuracy
accuracy_knn = accuracy_score(y_valid_encoded, knn_predictions)

# Display confusion matrix and net profit for kNN model
print(f"Accuracy (kNN): {accuracy_knn:.2%}")
print("Confusion Matrix (kNN):\n", conf_matrix_knn)
print(f"Net Profit (kNN): ${net_profit_knn}")


# In[74]:


y_pred = knn_model.predict_proba(X_valid_scaled)[:, 1] > best_cutoff

# Create a DataFrame with actual and predicted probabilities
results = pd.DataFrame({'Actual': y_valid,
                        'Predicted_Prob': y_pred})

# Sort the DataFrame by predicted probabilities
results = results.sort_values(by=['Predicted_Prob'], ascending=False)


# In[75]:


# Plot Cumulative Gains Chart and Lift Chart
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ax1 = gainsChart(results.Actual, ax=axes[0])
ax1.set_ylabel('  ')
ax1.set_title('Cumulative Gains Chart')

ax2 = liftChart(results.Actual, ax=axes[1], labelBars=False)
ax2.set_ylabel('Lift')

plt.tight_layout()
plt.show()


# ### Random Forest Classifier

# In[76]:


# Create a new DataFrame (df2) containing only the selected predictors and the target variable ('MIS_Status').
df2 = Loan_df[['Term','NoEmp','RetainedJob','New',
               'CreateJob','UrbanRural','DisbursementGross',
               'ChgOffPrinGr','GrAppv','SBA_Appv',
               'RealEstate','Portion','Recession','Default']]
# Drop rows with null values from the dataset to ensure a complete dataset for modeling.
#df2.dropna(inplace=True)

# Separate the features (X) and the target variable (y) from the cleaned dataset.
X = df2[['Term','NoEmp','RetainedJob','New',
               'CreateJob','UrbanRural','DisbursementGross',
               'ChgOffPrinGr','GrAppv','SBA_Appv',
               'RealEstate','Portion','Recession']]
y = df2['Default']


# In[77]:


# Specify categorical columns
categorical_columns = ['New','UrbanRural','RealEstate','Recession']

# Convert specified columns to 'category' data type and apply get_dummies
X_mapped = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_mapped, y, test_size=0.2, random_state=1)


# In[78]:


# Define and train the random forest classifier with fewer estimators
rf = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)

# Calculate feature importances in parallel
importances = np.mean([tree.feature_importances_ for tree in rf.estimators_], axis=0)
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
df = pd.DataFrame({'feature': X_train.columns, 'importance': importances, 'std': std})
df = df.sort_values('importance', ascending=True)
print(df)

# Plot the variable (feature) importance
ax = df.plot(kind='barh', xerr='std', x='feature',color='blue', legend=False)
ax.set_ylabel('')
plt.show()

# Evaluate the model and print the classification report
y_pred = rf.predict(X_valid)
print(classification_report(y_valid, y_pred))


# #### Random forest classifier and confusion summary

# In[79]:


rf = RandomForestClassifier(n_estimators=500, random_state=1, n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)
# variable (feature) importance plot
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
df = pd.DataFrame({'feature': X_train.columns, 'importance': importances, 'std': std})
df = df.sort_values('importance', ascending=True)
print(df)
ax = df.plot(kind='barh', xerr='std', x='feature',color='blue',legend=False)
ax.set_ylabel('')
plt.show()
# confusion matrix for validation set
classificationSummary(y_valid , rf.predict(X_valid))


# ### Best Cutoff for random forest

# In[81]:


# Define costs
cost_fp = -5 * 0.05
cost_tp = 0.05

# Compute probabilities once
probabilities = rf.predict_proba(X_valid)[:, 1]

# Define cutoffs and initialize arrays
cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = np.zeros_like(cutoffs)
specificity_scores = np.zeros_like(cutoffs)
youden_scores = np.zeros_like(cutoffs)
net_profit = np.zeros_like(cutoffs)
accuracy_scores = np.zeros_like(cutoffs)

# Vectorized computation for each cutoff
for i, cutoff in enumerate(cutoffs):
    y_pred = (probabilities > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    sensitivity_scores[i] = sensitivity
    specificity_scores[i] = specificity
    youden_scores[i] = sensitivity + specificity - 1
    net_profit[i] = cost_fp * conf_matrix[0, 1] + cost_tp * conf_matrix[0, 0]

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    accuracy_scores[i] = accuracy

    # Print results
    print(f"Cutoff: {cutoff:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Accuracy: {accuracy:.2%}, Net Profit: {net_profit[i]:.2f}")

    # Filter for total cost > 0 using NumPy arrays
mask = net_profit > 0
filtered_cutoffs = cutoffs[mask]
filtered_sensitivity_scores = sensitivity_scores[mask]
filtered_specificity_scores = specificity_scores[mask]
filtered_net_profit = net_profit[mask]
filtered_accuracy_scores = accuracy_scores[mask]

# Plot the results
plt.plot(cutoffs, sensitivity_scores, label='Sensitivity', marker='o')
plt.plot(cutoffs, specificity_scores, label='Specificity', marker='o')
plt.plot(cutoffs, accuracy_scores, label='Accuracy', marker='o')
plt.title('Sensitivity, Specificity, and Accuracy vs. Cutoff Value')
plt.xlabel('Cutoff Value')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot the results for net profit (where net profit is greater than 0)
plt.plot(cutoffs, net_profit, label="Net Profit", marker='o', color='red')
plt.title("Net Profit vs. Cutoff Value")
plt.xlabel('Cutoff Value')
plt.ylabel("Net Profit")
plt.legend()
plt.show()

# Print the best cutoff and corresponding net profit
best_index = np.argmax(net_profit)
best_cutoff = cutoffs[best_index]
best_net_profit = net_profit[best_index]
print(f"Best Cutoff: {best_cutoff}")
print(f"Corresponding Net Profit: {best_net_profit:.2f}")


# In[82]:


# Use the best cutoff for predictions on the validation set
rf_predictions = (rf.predict_proba(X_valid)[:, 1] > best_cutoff).astype(int)

# Define the cost matrix
cost_matrix = np.array([[0.05, -5*0.05], [0, 0]])

def calculate_net_profit(conf_matrix, cost_matrix):
    return np.sum(conf_matrix * cost_matrix)

# Calculate net profit for RF model
net_profit_rf = calculate_net_profit(confusion_matrix(y_valid, rf_predictions), cost_matrix)

# Calculate accuracy
accuracy_rf = accuracy_score(y_valid, rf_predictions)

# Display results
print(f"Accuracy (RF): {accuracy_rf:.2%}")
print("Confusion Matrix (RF):\n", confusion_matrix(y_valid, rf_predictions))
print(f"Net Profit (RF): ${net_profit_rf:.2f}")


# In[83]:


y_pred = rf.predict_proba(X_valid)[:, 1] > best_cutoff

# Create a DataFrame with actual and predicted probabilities
results = pd.DataFrame({'Actual': y_valid,
                        'Predicted_Prob': y_pred})

# Sort the DataFrame by predicted probabilities
results = results.sort_values(by=['Predicted_Prob'], ascending=False)


# In[84]:


# Plot Cumulative Gains Chart and Lift Chart
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ax1 = gainsChart(results.Actual, ax=axes[0])
ax1.set_ylabel('  ')
ax1.set_title('Cumulative Gains Chart')

ax2 = liftChart(results.Actual, ax=axes[1], labelBars=False)
ax2.set_ylabel('Lift')

plt.tight_layout()
plt.show()


# ### Decision Tree

# In[85]:


smallClassTree = DecisionTreeClassifier(max_depth=7, min_samples_split=50, min_impurity_decrease=0.01)
smallClassTree.fit(X_train, y_train)

# Model Evaluation
y_pred = smallClassTree.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)

plotDecisionTree(smallClassTree, feature_names=X_train.columns, class_names=smallClassTree.classes_)


# In[86]:


# Define costs
cost_fp = -5 * 0.05
cost_tp = 0.05

# Compute probabilities once
probabilities = smallClassTree.predict_proba(X_valid)[:, 1]

# Define cutoffs and initialize arrays
cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = np.zeros_like(cutoffs)
specificity_scores = np.zeros_like(cutoffs)
youden_scores = np.zeros_like(cutoffs)
net_profit = np.zeros_like(cutoffs)
accuracy_scores = np.zeros_like(cutoffs)

# Vectorized computation for each cutoff
for i, cutoff in enumerate(cutoffs):
    y_pred = (probabilities > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    sensitivity_scores[i] = sensitivity
    specificity_scores[i] = specificity
    youden_scores[i] = sensitivity + specificity - 1
    net_profit[i] = cost_fp * conf_matrix[0, 1] + cost_tp * conf_matrix[0, 0]

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    accuracy_scores[i] = accuracy

    # Print results
    print(f"Cutoff: {cutoff:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Accuracy: {accuracy:.2%}, Net Profit: {net_profit[i]:.2f}")

# Filter for total cost > 0 using NumPy arrays
mask = net_profit > 0
filtered_cutoffs = cutoffs[mask]
filtered_sensitivity_scores = sensitivity_scores[mask]
filtered_specificity_scores = specificity_scores[mask]
filtered_net_profit = net_profit[mask]
filtered_accuracy_scores = accuracy_scores[mask]

# Plot the results
plt.plot(cutoffs, sensitivity_scores, label='Sensitivity', marker='o')
plt.plot(cutoffs, specificity_scores, label='Specificity', marker='o')
plt.plot(cutoffs, accuracy_scores, label='Accuracy', marker='o')
plt.title('Sensitivity, Specificity, and Accuracy vs. Cutoff Value')
plt.xlabel('Cutoff Value')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot the results for net profit (where net profit is greater than 0)
plt.plot(cutoffs, net_profit, label="Net Profit", marker='o', color='red')
plt.title("Net Profit vs. Cutoff Value")
plt.xlabel('Cutoff Value')
plt.ylabel("Net Profit")
plt.legend()
plt.show()

    
# Print the best cutoff and corresponding net profit
best_index = np.argmax(net_profit)
best_cutoff = cutoffs[best_index]
best_net_profit = net_profit[best_index]

print(f"Best Cutoff: {best_cutoff}")
print(f"Corresponding Net Profit: {best_net_profit:.2f}")


# In[87]:


# Use the best cutoff for predictions on the validation set
smallClassTree_predictions = (smallClassTree.predict_proba(X_valid)[:, 1] > best_cutoff).astype(int)

# Define the cost matrix
cost_matrix = np.array([[0.05, -5*0.05], [0, 0]])

# Calculate net profit for RF model
net_profit_rf = calculate_net_profit(confusion_matrix(y_valid, smallClassTree_predictions), cost_matrix)

# Calculate accuracy
accuracy_rf = accuracy_score(y_valid, smallClassTree_predictions)

# Display results
print(f"Accuracy : {accuracy_rf:.2%}")
print("Confusion Matrix:\n", confusion_matrix(y_valid, smallClassTree_predictions))
print(f"Net Profit : ${net_profit_rf:.2f}")


# In[88]:


y_pred = smallClassTree.predict_proba(X_valid)[:, 1] > best_cutoff

# Create a DataFrame with actual and predicted probabilities
results = pd.DataFrame({'Actual': y_valid,
                        'Predicted_Prob': y_pred})

# Sort the DataFrame by predicted probabilities
results = results.sort_values(by=['Predicted_Prob'], ascending=False)


# In[89]:


# Plot Cumulative Gains Chart and Lift Chart
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ax1 = gainsChart(results.Actual, ax=axes[0])
ax1.set_ylabel('  ')
ax1.set_title('Cumulative Gains Chart')

ax2 = liftChart(results.Actual, ax=axes[1], labelBars=False)
ax2.set_ylabel('Lift')

plt.tight_layout()
plt.show()


# In[90]:


# Print the rules of the decision tree
rules = export_text(smallClassTree, feature_names=X_train.columns.tolist())
print("Decision Tree Rules:")
print(rules)


# In[91]:


# Choose the number of components
n_components = 3  #

# Apply PCA
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)

# You can check the explained variance
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

max_depth_value = 5  # You can adjust this value as needed
fullClassTree = DecisionTreeClassifier(max_depth=max_depth_value)  # use your chosen max_depth value
fullClassTree.fit(X_train_pca, y_train)

# Transform X_valid using the same PCA transformation
X_valid_pca = pca.transform(X_valid)

# Model Evaluation
y_pred = fullClassTree.predict(X_valid_pca)

accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)

# Generating feature names for PCA components
pca_feature_names = [f"PC{i+1}" for i in range(n_components)]

plt.figure(figsize=(20, 10))
# Correctly use PCA feature names when plotting the decision tree
plotDecisionTree(fullClassTree, feature_names=pca_feature_names, class_names=fullClassTree.classes_)



# In[92]:


# Transform X_train using the same PCA transformation
X_train_pca = pca.transform(X_train)  

# Model Prediction for training data
y_train_pred = fullClassTree.predict(X_train_pca)
classificationSummary(y_train, y_train_pred)

# Model Prediction for validation data
y_valid_pred = fullClassTree.predict(X_valid_pca)
classificationSummary(y_valid, y_valid_pred)


# In[94]:


# Define your parameter space
param_dist = {
    'max_depth': randint(10, 40),  # Specifying a distribution
    'min_samples_split': randint(20, 100),
    'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01]
}

# Number of iterations for RandomizedSearch
n_iter_search = 20  # You can adjust this based on computational resource

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=1),
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit to the training data
random_search.fit(X_train, y_train)

# Print the results
print('Best score achieved:', "{:.5f}".format(random_search.best_score_))
print('Best parameters:', random_search.best_params_)


# In[95]:


# Create a DecisionTreeClassifier with the best parameters
best_classifier = DecisionTreeClassifier(
    max_depth=18,
    min_impurity_decrease=0,
    min_samples_split=86,
    random_state=1
)

# Fit the classifier to your training data
best_classifier.fit(X_train, y_train)


# In[96]:


def plotLimitedDepthDecisionTree(classifier, feature_names, max_depth=2):
    plt.figure(figsize=(20,10))  # Set the size of the figure
    plot_tree(classifier, 
              feature_names=feature_names, 
              class_names=True, 
              filled=True, 
              rounded=True,
              max_depth=max_depth)  # Limit the depth
    plt.show()

# Visualize the tree with limited depth
plotLimitedDepthDecisionTree(best_classifier, feature_names=X_train.columns)


# In[98]:


# Print the rules of the decision tree
rules = export_text(best_classifier, feature_names=X_train.columns.tolist())
print("Decision Tree Rules:")
print(rules)


# ## Classification Trees

# ### Single Tree

# In[99]:


# single tree
defaultTree = DecisionTreeClassifier(criterion="entropy", random_state=1)
defaultTree.fit(X_train, y_train)
classes = defaultTree.classes_

y_pred =  defaultTree.predict_proba(X_valid)

result = pd.DataFrame({
    'actual': y_valid,
    'p(0)': [p[0] for p in y_pred],
    'p(1)': [p[1] for p in y_pred],
    'predicted': defaultTree .predict(X_valid),
})

result = result.sort_values(by=['p(1)'], ascending=False)
classificationSummary(result.actual, result.predicted, class_names=defaultTree.classes_)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
gainsChart(result.actual, ax=axes[0])
liftChart(result['p(1)'], title=False, ax=axes[1])
plt.show()


# ### Best Cutoff for Single Tree

# In[100]:


# Define costs
cost_fp = -5 * 0.05
cost_tp = 0.05

# Compute probabilities once
probabilities = defaultTree.predict_proba(X_valid)[:, 1]

# Define cutoffs and initialize arrays
cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = np.zeros_like(cutoffs)
specificity_scores = np.zeros_like(cutoffs)
youden_scores = np.zeros_like(cutoffs)
net_profit = np.zeros_like(cutoffs)
accuracy_scores = np.zeros_like(cutoffs)

# Vectorized computation for each cutoff
for i, cutoff in enumerate(cutoffs):
    y_pred = (probabilities > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    sensitivity_scores[i] = sensitivity
    specificity_scores[i] = specificity
    youden_scores[i] = sensitivity + specificity - 1
    net_profit[i] = cost_fp * conf_matrix[0, 1] + cost_tp * conf_matrix[0, 0]

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    accuracy_scores[i] = accuracy

    # Print results
    print(f"Cutoff: {cutoff:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Accuracy: {accuracy:.2%}, Net Profit: {net_profit[i]:.2f}")

# Filter for total cost > 0 using NumPy arrays
mask = net_profit > 0
filtered_cutoffs = cutoffs[mask]
filtered_sensitivity_scores = sensitivity_scores[mask]
filtered_specificity_scores = specificity_scores[mask]
filtered_net_profit = net_profit[mask]
filtered_accuracy_scores = accuracy_scores[mask]

# Plot the results
plt.plot(cutoffs, sensitivity_scores, label='Sensitivity', marker='o')
plt.plot(cutoffs, specificity_scores, label='Specificity', marker='o')
plt.plot(cutoffs, accuracy_scores, label='Accuracy', marker='o')
plt.title('Sensitivity, Specificity, and Accuracy vs. Cutoff Value')
plt.xlabel('Cutoff Value')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot the results for net profit (where net profit is greater than 0)
plt.plot(cutoffs, net_profit, label="Net Profit", marker='o', color='red')
plt.title("Net Profit vs. Cutoff Value")
plt.xlabel('Cutoff Value')
plt.ylabel("Net Profit")
plt.legend()
plt.show()
    

# Print the best cutoff and corresponding net profit
best_index = np.argmax(net_profit)
best_cutoff = cutoffs[best_index]
best_net_profit = net_profit[best_index]
print(f"Best Cutoff: {best_cutoff}")
print(f"Corresponding Net Profit: {best_net_profit:.2f}")


# In[101]:


# Use the best cutoff for predictions on the validation set
defaultTree_predictions = (defaultTree.predict_proba(X_valid)[:, 1] > best_cutoff).astype(int)

# Define the cost matrix
cost_matrix = np.array([[0.05, -5*0.05], [0, 0]])

# Calculate net profit for single tree model
net_profitdefaultTree = calculate_net_profit(confusion_matrix(y_valid, defaultTree_predictions), cost_matrix)

# Calculate accuracy
accuracy_defaultTree = accuracy_score(y_valid, defaultTree_predictions)

# Display results
print(f"Accuracy (SingleTree): {accuracy_defaultTree:.2%}")
print("Confusion Matrix (SingleTree):\n", confusion_matrix(y_valid, defaultTree_predictions))
print(f"Net Profit (SingleTree): ${net_profitdefaultTree:.2f}")


# ### Bagging

# In[103]:


# Bagging with parallel processing
bagging = BaggingClassifier(defaultTree, n_estimators=100, n_jobs=-1, random_state=1, verbose=1)
bagging.fit(X_train, y_train)

# Efficient probability extraction using NumPy
y_pred = bagging.predict_proba(X_valid)
p_0, p_1 = y_pred[:, 0], y_pred[:, 1]  # Splitting probabilities

# Create DataFrame using efficient methods
result = pd.DataFrame({
    'actual': y_valid,
    'p(0)': p_0,
    'p(1)': p_1,
    'predicted': bagging.predict(X_valid),
})

result = result.sort_values(by=['p(1)'], ascending=False)
classificationSummary(result.actual, result.predicted, class_names=bagging.classes_)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
gainsChart(result.actual, ax=axes[0])
liftChart(result['p(1)'], title=False, ax=axes[1])
plt.show()


# ### Best Cutoff for Bagging 

# In[104]:


# Define costs
cost_fp = -5 * 0.05
cost_tp = 0.05

# Compute probabilities once
probabilities = bagging.predict_proba(X_valid)[:, 1]

# Define cutoffs and initialize arrays
cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = np.zeros_like(cutoffs)
specificity_scores = np.zeros_like(cutoffs)
youden_scores = np.zeros_like(cutoffs)
net_profit = np.zeros_like(cutoffs)
accuracy_scores = np.zeros_like(cutoffs)

# Vectorized computation for each cutoff
for i, cutoff in enumerate(cutoffs):
    y_pred = (probabilities > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    sensitivity_scores[i] = sensitivity
    specificity_scores[i] = specificity
    youden_scores[i] = sensitivity + specificity - 1
    net_profit[i] = cost_fp * conf_matrix[0, 1] + cost_tp * conf_matrix[0, 0]

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    accuracy_scores[i] = accuracy

    # Print results
    print(f"Cutoff: {cutoff:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Accuracy: {accuracy:.2%}, Net Profit: {net_profit[i]:.2f}")

# Filter for total cost > 0 using NumPy arrays
mask = net_profit > 0
filtered_cutoffs = cutoffs[mask]
filtered_sensitivity_scores = sensitivity_scores[mask]
filtered_specificity_scores = specificity_scores[mask]
filtered_net_profit = net_profit[mask]
filtered_accuracy_scores = accuracy_scores[mask]

# Plot the results
plt.plot(cutoffs, sensitivity_scores, label='Sensitivity', marker='o')
plt.plot(cutoffs, specificity_scores, label='Specificity', marker='o')
plt.plot(cutoffs, accuracy_scores, label='Accuracy', marker='o')
plt.title('Sensitivity, Specificity, and Accuracy vs. Cutoff Value')
plt.xlabel('Cutoff Value')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot the results for net profit (where net profit is greater than 0)
plt.plot(cutoffs, net_profit, label="Net Profit", marker='o', color='red')
plt.title("Net Profit vs. Cutoff Value")
plt.xlabel('Cutoff Value')
plt.ylabel("Net Profit")
plt.legend()
plt.show()

# Print the best cutoff and corresponding net profit
best_index = np.argmax(net_profit)
best_cutoff = cutoffs[best_index]
best_net_profit = net_profit[best_index]
print(f"Best Cutoff: {best_cutoff}")
print(f"Corresponding Net Profit: {best_net_profit:.2f}")


# In[105]:


# Use the best cutoff for predictions on the validation set
bagging_predictions = (bagging.predict_proba(X_valid)[:, 1] > best_cutoff).astype(int)

# Define the cost matrix
cost_matrix = np.array([[0.05, -5*0.05], [0, 0]])

# Calculate net profit for single tree model
net_profitbagging = calculate_net_profit(confusion_matrix(y_valid, bagging_predictions), cost_matrix)

# Calculate accuracy
accuracy_Bagging = accuracy_score(y_valid, bagging_predictions)

# Display results
print(f"Accuracy (Bagging): {accuracy_Bagging:.2%}")
print("Confusion Matrix (Bagging):\n", confusion_matrix(y_valid, bagging_predictions))
print(f"Net Profit (Bagging): ${net_profitbagging:.2f}")


# ### Boosting 

# In[106]:


boost = AdaBoostClassifier(defaultTree, n_estimators=100, random_state=1)
boost.fit(X_train, y_train)

y_pred =  boost.predict_proba(X_valid)

result = pd.DataFrame({
    'actual': y_valid,
    'p(0)': [p[0] for p in y_pred],
    'p(1)': [p[1] for p in y_pred],
    'predicted': boost.predict(X_valid),
})

result = result.sort_values(by=['p(1)'], ascending=False)

classificationSummary(result.actual, result.predicted, class_names=boost.classes_)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
gainsChart(result.actual, ax=axes[0])
liftChart(result['p(1)'], title=False, ax=axes[1])
plt.show()


# ### Best Cutoff for Boosting

# In[107]:


# Define costs
cost_fp = -5 * 0.05
cost_tp = 0.05

# Compute probabilities once
probabilities = boost.predict_proba(X_valid)[:, 1]

# Define cutoffs and initialize arrays
cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = np.zeros_like(cutoffs)
specificity_scores = np.zeros_like(cutoffs)
youden_scores = np.zeros_like(cutoffs)
net_profit = np.zeros_like(cutoffs)
accuracy_scores = np.zeros_like(cutoffs)

# Vectorized computation for each cutoff
for i, cutoff in enumerate(cutoffs):
    y_pred = (probabilities > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    sensitivity_scores[i] = sensitivity
    specificity_scores[i] = specificity
    youden_scores[i] = sensitivity + specificity - 1
    net_profit[i] = cost_fp * conf_matrix[0, 1] + cost_tp * conf_matrix[0, 0]

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    accuracy_scores[i] = accuracy

    # Print results
    print(f"Cutoff: {cutoff:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Accuracy: {accuracy:.2%}, Net Profit: {net_profit[i]:.2f}")

    # Filter for total cost > 0 using NumPy arrays
mask = net_profit > 0
filtered_cutoffs = cutoffs[mask]
filtered_sensitivity_scores = sensitivity_scores[mask]
filtered_specificity_scores = specificity_scores[mask]
filtered_net_profit = net_profit[mask]
filtered_accuracy_scores = accuracy_scores[mask]

# Plot the results
plt.plot(cutoffs, sensitivity_scores, label='Sensitivity', marker='o')
plt.plot(cutoffs, specificity_scores, label='Specificity', marker='o')
plt.plot(cutoffs, accuracy_scores, label='Accuracy', marker='o')
plt.title('Sensitivity, Specificity, and Accuracy vs. Cutoff Value')
plt.xlabel('Cutoff Value')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot the results for net profit (where net profit is greater than 0)
plt.plot(cutoffs, net_profit, label="Net Profit", marker='o', color='red')
plt.title("Net Profit vs. Cutoff Value")
plt.xlabel('Cutoff Value')
plt.ylabel("Net Profit")
plt.legend()
plt.show()


# Print the best cutoff and corresponding net profit
best_index = np.argmax(net_profit)
best_cutoff = cutoffs[best_index]
best_net_profit = net_profit[best_index]
print(f"Best Cutoff: {best_cutoff}")
print(f"Corresponding Net Profit: {best_net_profit:.2f}")


# In[108]:


# Use the best cutoff for predictions on the validation set
boost_predictions = (boost.predict_proba(X_valid)[:, 1] > best_cutoff).astype(int)

# Define the cost matrix
cost_matrix = np.array([[0.05, -5*0.05], [0, 0]])

# Calculate net profit for single tree model
net_profitboost = calculate_net_profit(confusion_matrix(y_valid, boost_predictions), cost_matrix)

# Calculate accuracy
accuracy_boost = accuracy_score(y_valid, boost_predictions)

# Display results
print(f"Accuracy (boost): {accuracy_boost:.2%}")
print("Confusion Matrix (boost):\n", confusion_matrix(y_valid, boost_predictions))
print(f"Net Profit (boost): ${net_profitboost:.2f}")


# ### Discriminant Analysis

# In[109]:


print('Training set:', X_train.shape, 'Validation set:', X_valid.shape)

da_reg = LinearDiscriminantAnalysis()
da_reg.fit(X_train, y_train)
intercept = da_reg.intercept_[0]
print('Coefficients', da_reg.coef_)
print('Intercept', da_reg.intercept_)


# In[110]:


print('intercept',da_reg.intercept_[0])
coefficients_df = pd.DataFrame({'coeff': da_reg.coef_[0][:len(X.columns)]}, index=X.columns)
coefficients_df = coefficients_df.transpose()
print(coefficients_df)


# In[111]:


da_reg.priors_


# In[112]:


da_reg_pred = da_reg.predict_proba(X_valid)
result = pd.DataFrame({
    'actual': y_valid,
    'p(0)': [p[0] for p in da_reg_pred],
    'p(1)': [p[1] for p in da_reg_pred],
    'predicted': da_reg.predict(X_valid),
})

result = result.sort_values(by=['p(1)'], ascending=False)

# Confusion matrix
classificationSummary(result.actual, result.predicted, class_names=da_reg.classes_)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
gainsChart(result.actual, ax=axes[0])
liftChart(result['p(1)'], title=False, ax=axes[1])
plt.show()


# In[113]:


print ("Intercept:", da_reg.intercept_)
print("Priors:", da_reg.priors_)


# In[114]:


#adjusting the priors manually to see if the accuracy improves
da_reg_10 = LinearDiscriminantAnalysis(priors=[0.9, 0.1])
da_reg_10.fit(X_train, y_train)
classificationSummary(y_valid, da_reg_10.predict(X_valid))
print(da_reg_10.intercept_)


# ### Best Cutoff For Discriminant Analysis

# In[115]:


#DiscriminantAnalysis_Try different cutoff values
# Define costs
cost_fp = -5 * 0.05
cost_tp = 0.05

# Compute probabilities once
probabilities =da_reg.predict_proba(X_valid)[:, 1]

# Define cutoffs and initialize arrays
cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = np.zeros_like(cutoffs)
specificity_scores = np.zeros_like(cutoffs)
youden_scores = np.zeros_like(cutoffs)
net_profit = np.zeros_like(cutoffs)
accuracy_scores = np.zeros_like(cutoffs)

# Vectorized computation for each cutoff
for i, cutoff in enumerate(cutoffs):
    y_pred = (probabilities > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    sensitivity_scores[i] = sensitivity
    specificity_scores[i] = specificity
    youden_scores[i] = sensitivity + specificity - 1
    net_profit[i] = cost_fp * conf_matrix[0, 1] + cost_tp * conf_matrix[0, 0]

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    accuracy_scores[i] = accuracy

    # Print results
    print(f"Cutoff: {cutoff:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Accuracy: {accuracy:.2%}, Net Profit: {net_profit[i]:.2f}")

    # Filter for total cost > 0 using NumPy arrays
mask = net_profit > 0
filtered_cutoffs = cutoffs[mask]
filtered_sensitivity_scores = sensitivity_scores[mask]
filtered_specificity_scores = specificity_scores[mask]
filtered_net_profit = net_profit[mask]
filtered_accuracy_scores = accuracy_scores[mask]

# Plot the results
plt.plot(cutoffs, sensitivity_scores, label='Sensitivity', marker='o')
plt.plot(cutoffs, specificity_scores, label='Specificity', marker='o')
plt.plot(cutoffs, accuracy_scores, label='Accuracy', marker='o')
plt.title('Sensitivity, Specificity, and Accuracy vs. Cutoff Value')
plt.xlabel('Cutoff Value')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot the results for net profit (where net profit is greater than 0)
plt.plot(cutoffs, net_profit, label="Net Profit", marker='o', color='red')
plt.title("Net Profit vs. Cutoff Value")
plt.xlabel('Cutoff Value')
plt.ylabel("Net Profit")
plt.legend()
plt.show()

# Print the best cutoff and corresponding net profit
best_index = np.argmax(net_profit)
best_cutoff = cutoffs[best_index]
best_net_profit = net_profit[best_index]
print(f"Best Cutoff: {best_cutoff}")
print(f"Corresponding Net Profit: {best_net_profit:.2f}")


# In[116]:


# Use the best cutoff for predictions on the validation set
da_reg_predictions = (da_reg.predict_proba(X_valid)[:, 1] > best_cutoff).astype(int)

# Define the cost matrix
cost_matrix = np.array([[0.05, -5*0.05], [0, 0]])

# Calculate net profit for single tree model
net_profitda_reg = calculate_net_profit(confusion_matrix(y_valid, da_reg_predictions), cost_matrix)

# Calculate accuracy
accuracy_da_reg = accuracy_score(y_valid, da_reg_predictions)

# Display results
print(f"Accuracy (Discriminant Analysis): {accuracy_da_reg:.2%}")
print("Confusion Matrix (Discriminant Analysis):\n", confusion_matrix(y_valid, da_reg_predictions))
print(f"Net Profit (Discriminant Analysis): ${net_profitda_reg:.2f}")


# ### Logistic Regression

# In[128]:


# Setting up logistic regression models with different regularization techniques
# Lasso (L1 regularization)
logit_lasso = LogisticRegressionCV(cv=3, penalty='l1', solver='saga', max_iter=1000, tol=0.001, random_state=1, n_jobs=-1, verbose=1)

# Ridge (L2 regularization)
logit_ridge = LogisticRegressionCV(cv=3, penalty='l2', solver='saga', max_iter=1000, tol=0.001, random_state=1, n_jobs=-1, verbose=1)

# ElasticNet (Combination of L1 and L2 regularization)
logit_elasticnet = LogisticRegressionCV(cv=3, penalty='elasticnet', solver='saga', l1_ratios=[0.1, 0.5, 0.9], max_iter=1000, tol=0.001, random_state=1, n_jobs=-1, verbose=1)

# Fit the models
logit_lasso.fit(X_train_scaled, y_train)
logit_ridge.fit(X_train_scaled, y_train)
logit_elasticnet.fit(X_train_scaled, y_train)

# Best hyperparameters from cross-validation
best_l1_ratio = logit_elasticnet.l1_ratio_[0]


# ### Eveluation of three logistic regression models
# 
# Lasso (L1 regularization)
# 
# Ridge (L2 regularization) 
# 
# ElasticNet (combination of L1 and L2 regularization)

# In[129]:


# Function to calculate metrics
def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = round(accuracy_score(y_test, y_pred), 6)
    precision = round(precision_score(y_test, y_pred), 6)
    recall = round(recall_score(y_test, y_pred), 6)
    f1 = round(f1_score(y_test, y_pred), 6)
    roc_auc = round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 6)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

    return metrics, conf_matrix

# Evaluating each model
metrics_lasso, conf_matrix_lasso = evaluate_model(logit_lasso, X_valid_scaled, y_valid)
metrics_ridge, conf_matrix_ridge = evaluate_model(logit_ridge, X_valid_scaled, y_valid)
metrics_elasticnet, conf_matrix_elasticnet = evaluate_model(logit_elasticnet, X_valid_scaled, y_valid)

# Create a DataFrame to display the metrics
df_metrics = pd.DataFrame([metrics_lasso, metrics_ridge, metrics_elasticnet],
                          index=['Lasso', 'Ridge', 'ElasticNet'])

print("Metrics:")
print(df_metrics)

# Print confusion matrices separately
print("\nConfusion Matrices:")
print("Lasso:\n", conf_matrix_lasso)
print("\nRidge:\n", conf_matrix_ridge)
print("\nElasticNet:\n", conf_matrix_elasticnet)


# ### Hyperparameter tuning for logistic regression models 

# In[131]:


# Suppress Intel MKL warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mkl")
# Suppress all warnings
#warnings.filterwarnings("ignore")

# Define the number of components or the variance threshold for PCA
n_components = 0.95  # For example, keep 95% of the variance

# Create pipelines for each model
pipelines = {
    'lasso': Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('logistic', LogisticRegression(penalty='l1', max_iter=5000, tol=0.001, random_state=1))
    ]),
    'ridge': Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('logistic', LogisticRegression(penalty='l2', max_iter=5000, tol=0.001, random_state=1))
    ]),
    'elasticnet': Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('logistic', LogisticRegression(penalty='elasticnet', max_iter=5000, tol=0.001, random_state=1))
    ])
}

# Parameter grids
param_grids = {
    'lasso': {
        'logistic__C': [0.01, 0.1, 1, 10, 100],
        'logistic__solver': ['saga']
    },
    'ridge': {
        'logistic__C': [0.01, 0.1, 1, 10, 100],
        'logistic__solver': ['saga']
    },
    'elasticnet': {
        'logistic__C': [0.01, 0.1, 1, 10, 100],
        'logistic__l1_ratio': [0.1, 0.5, 0.7, 0.9],
        'logistic__solver': ['saga']
    }
}

# Randomized Search for each model
results = {}
for model_name, pipeline in pipelines.items():
    logit_rs = RandomizedSearchCV(pipeline, param_grids[model_name], n_iter=3, cv=3, scoring='roc_auc', n_jobs=-1, random_state=1, verbose=1)
    logit_rs.fit(X_train_scaled, y_train)
    results[model_name] = logit_rs

# Best parameters and scores
best_params = {model_name: results[model_name].best_params_ for model_name in results}
best_scores = {model_name: results[model_name].best_score_ for model_name in results}

(best_params['lasso'], best_scores['lasso'], best_params['ridge'], best_scores['ridge'], best_params['elasticnet'], best_scores['elasticnet'])


# ### Creating new pipelines with the best parameters

# In[137]:


# Creating new pipelines with the best parameters
pipeline_lasso_best = Pipeline([
    ('pca', PCA(n_components=n_components)),
    ('logistic', LogisticRegression(penalty='l1', C=best_params['lasso']['logistic__C'], solver=best_params['lasso']['logistic__solver'], max_iter=5000, tol=0.001, random_state=1))
])

pipeline_ridge_best = Pipeline([
    ('pca', PCA(n_components=n_components)),
    ('logistic', LogisticRegression(penalty='l2', C=best_params['ridge']['logistic__C'], solver=best_params['ridge']['logistic__solver'], max_iter=5000, tol=0.001, random_state=1))
])

pipeline_elasticnet_best = Pipeline([
    ('pca', PCA(n_components=n_components)),
    ('logistic', LogisticRegression(penalty='elasticnet', C=best_params['elasticnet']['logistic__C'], l1_ratio=best_params['elasticnet']['logistic__l1_ratio'], solver=best_params['elasticnet']['logistic__solver'], max_iter=5000, tol=0.001, random_state=1))
])

# Fitting the best models
pipeline_lasso_best.fit(X_train_scaled, y_train)
pipeline_ridge_best.fit(X_train_scaled, y_train)
pipeline_elasticnet_best.fit(X_train_scaled, y_train)

# Evaluating the models
# Replace 'evaluate_model' with your actual evaluation function
metrics_lasso_best = evaluate_model(pipeline_lasso_best, X_valid_scaled, y_valid)
metrics_ridge_best = evaluate_model(pipeline_ridge_best, X_valid_scaled, y_valid)
metrics_elasticnet_best = evaluate_model(pipeline_elasticnet_best, X_valid_scaled, y_valid)

(metrics_lasso_best, metrics_ridge_best, metrics_elasticnet_best)


# In[138]:


# Lasso model
lasso_model = pipeline_lasso_best  # Use logit_lasso_best, logit_ridge_best, or logit_elasticnet_best

cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = []
specificity_scores = []
youden_scores = []
fpr_values = []
tpr_values = []

for cutoff in cutoffs:
    y_pred_prob = lasso_model.predict_proba(X_valid_scaled)[:, 1]
    y_pred = (y_pred_prob > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)
    
    # Correct calculation of sensitivity (TPR) and specificity (TNR)
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # TPR
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # TNR
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)
    
    # Calculate Youden's J statistic
    youden = sensitivity + specificity - 1
    youden_scores.append(youden)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_valid, y_pred_prob)
    fpr_values.append(false_positive_rate)
    tpr_values.append(true_positive_rate)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
for i in range(len(cutoffs)):
    plt.plot(fpr_values[i], tpr_values[i], label=f'Cutoff: {cutoffs[i]:.2f}')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('ROC Curve for Different Cutoff Values (Lasso Model)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Find the cutoff that maximizes Youden's J statistic
best_cutoff = cutoffs[np.argmax(youden_scores)]

# Print the best cutoff
best_cutoff


# In[140]:


# Ridge model
ridge_model = pipeline_ridge_best  # Use logit_lasso_best, logit_ridge_best, or logit_elasticnet_best

cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = []
specificity_scores = []
youden_scores = []
fpr_values = []
tpr_values = []

for cutoff in cutoffs:
    y_pred_prob = ridge_model.predict_proba(X_valid_scaled)[:, 1]
    y_pred = (y_pred_prob > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)
    
    # Correct calculation of sensitivity (TPR) and specificity (TNR)
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # TPR
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # TNR
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)
    
    # Calculate Youden's J statistic
    youden = sensitivity + specificity - 1
    youden_scores.append(youden)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_valid, y_pred_prob)
    fpr_values.append(false_positive_rate)
    tpr_values.append(true_positive_rate)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
for i in range(len(cutoffs)):
    plt.plot(fpr_values[i], tpr_values[i], label=f'Cutoff: {cutoffs[i]:.2f}')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('ROC Curve for Different Cutoff Values (Lasso Model)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Find the cutoff that maximizes Youden's J statistic
best_cutoff = cutoffs[np.argmax(youden_scores)]

# Print the best cutoff
best_cutoff


# In[141]:


# Elastic Net model
elasticnet_model = pipeline_elasticnet_best  # Use logit_lasso_best, logit_ridge_best, or logit_elasticnet_best

cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = []
specificity_scores = []
youden_scores = []
fpr_values = []
tpr_values = []

for cutoff in cutoffs:
    y_pred_prob = elasticnet_model.predict_proba(X_valid_scaled)[:, 1]
    y_pred = (y_pred_prob > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)
    
    # Correct calculation of sensitivity (TPR) and specificity (TNR)
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # TPR
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # TNR
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)
    
    # Calculate Youden's J statistic
    youden = sensitivity + specificity - 1
    youden_scores.append(youden)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_valid, y_pred_prob)
    fpr_values.append(false_positive_rate)
    tpr_values.append(true_positive_rate)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
for i in range(len(cutoffs)):
    plt.plot(fpr_values[i], tpr_values[i], label=f'Cutoff: {cutoffs[i]:.2f}')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('ROC Curve for Different Cutoff Values (Lasso Model)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Find the cutoff that maximizes Youden's J statistic
best_cutoff = cutoffs[np.argmax(youden_scores)]

# Print the best cutoff
best_cutoff


# ### Best Cutoff logistic Regression Model

# In[142]:


# Define costs
cost_fp = -5 * 0.05
cost_tp = 0.05

# Compute probabilities once
probabilities = elasticnet_model.predict_proba(X_valid_scaled)[:, 1]

# Define cutoffs and initialize arrays
cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = np.zeros_like(cutoffs)
specificity_scores = np.zeros_like(cutoffs)
youden_scores = np.zeros_like(cutoffs)
net_profit = np.zeros_like(cutoffs)
accuracy_scores = np.zeros_like(cutoffs)

# Vectorized computation for each cutoff
for i, cutoff in enumerate(cutoffs):
    y_pred = (probabilities > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    sensitivity_scores[i] = sensitivity
    specificity_scores[i] = specificity
    youden_scores[i] = sensitivity + specificity - 1
    net_profit[i] = cost_fp * conf_matrix[0, 1] + cost_tp * conf_matrix[0, 0]

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    accuracy_scores[i] = accuracy

    # Print results
    print(f"Cutoff: {cutoff:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Accuracy: {accuracy:.2%}, Net Profit: {net_profit[i]:.2f}")

#Filter for total cost > 0 using NumPy arrays
mask = net_profit > 0
filtered_cutoffs = cutoffs[mask]
filtered_sensitivity_scores = sensitivity_scores[mask]
filtered_specificity_scores = specificity_scores[mask]
filtered_net_profit = net_profit[mask]
filtered_accuracy_scores = accuracy_scores[mask]

# Plot the results
plt.plot(filtered_cutoffs, sensitivity_scores[:len(filtered_cutoffs)], label='Sensitivity', marker='o')
plt.plot(filtered_cutoffs, specificity_scores[:len(filtered_cutoffs)], label='Specificity', marker='o')
plt.plot(filtered_cutoffs, accuracy_scores[:len(filtered_cutoffs)], label='Accuracy', marker='o')
plt.title('Sensitivity, Specificity, and Accuracy vs. Cutoff Value' )
plt.xlabel('Cutoff Value')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot the results for net profit (where net profit is greater than 0)
plt.plot(filtered_cutoffs, filtered_net_profit, label="Net Profit", marker='o', color='red')
plt.title("Net Profit vs. Cutoff Value")
plt.xlabel('Cutoff Value')
plt.ylabel("Net Profit")
plt.legend()
plt.show()

# Print the best cutoff and corresponding net profit
best_index = np.argmax(filtered_net_profit)
best_cutoff = filtered_cutoffs[best_index]
best_net_profit = filtered_net_profit[best_index]
print(f"Best Cutoff: {best_cutoff}")
print(f"Corresponding Net Profit: {best_net_profit:.2f}")


# In[ ]:


elasticnet_pred = elasticnet_model.predict_proba(X_valid_scaled)
result = pd.DataFrame({
    'actual': y_valid,
    'p(0)': [p[0] for p in elasticnet_pred],
    'p(1)': [p[1] for p in elasticnet_pred],
    'predicted': elasticnet_model.predict(X_valid),
})

result = result.sort_values(by=['p(1)'], ascending=False)

# Confusion matrix
classificationSummary(result.actual, result.predicted, class_names=elasticnet_model.classes_)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
gainsChart(result.actual, ax=axes[0])
liftChart(result['p(1)'], title=False, ax=axes[1])
plt.show()


# ![Screen%20Shot%202024-03-30%20at%202.51.57%20AM.png](attachment:Screen%20Shot%202024-03-30%20at%202.51.57%20AM.png)

# In[143]:


# Use the best cutoff for predictions on the validation set
elasticnet_model_predictions = (elasticnet_model.predict_proba(X_valid_scaled)[:, 1] > best_cutoff).astype(int)

# Define the cost matrix
cost_matrix = np.array([[0.05, -5*0.05], [0, 0]])

# Calculate net profit for elastic net model model
net_profit_elasticnet_model = calculate_net_profit(confusion_matrix(y_valid, elasticnet_model_predictions), cost_matrix)

# Calculate accuracy
accuracy_elasticnet_model = accuracy_score(y_valid, elasticnet_model_predictions)

# Display results
print(f"Accuracy (elasticnet_model): {accuracy_elasticnet_model:.2%}")
print("Confusion Matrix (elasticnet_model):\n", confusion_matrix(y_valid, elasticnet_model_predictions))
print(f"Net Profit (elasticnet_model): ${net_profit_elasticnet_model:.2f}")


# ## Neural Nets

# In[144]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# Initialize the constructor
model = Sequential()

# Add an input layer and a hidden layer with ReLU activation
model.add(Dense(units=int((X_train_scaled.shape[1] + 1) / 2), 
                activation='relu', 
                input_shape=(X_train_scaled.shape[1],)))

# Add an output layer with sigmoid activation
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Summary of the model
model.summary()


# ### Keras-based K-Fold Cross-Validation with Custom Early Stopping

# This code performs k-fold cross-validation using a neural network model built with Keras. It incorporates a custom early stopping mechanism to halt training if the validation loss fails to improve, providing a robust evaluation of the model's performance while mitigating the risk of overfitting.

# In[145]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# Define custom early stopping callback
from keras.callbacks import Callback

class CustomEarlyStopping(Callback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'):
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best_val_loss = np.Inf if mode == 'auto' else -np.Inf
        if mode not in ['auto', 'min', 'max']:
            warnings.warn(f'EarlyStopping mode {mode} is unknown, '
                          'fallback to auto mode.', RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get(self.monitor)
        if current_val_loss is None:
            warnings.warn(f'Early stopping conditioned on metric `{self.monitor}` '
                          'which is not available. Available metrics are: '
                          f'{",".join(list(logs.keys()))}', RuntimeWarning)
            return

        if self.monitor_op(current_val_loss - self.min_delta, self.best_val_loss):
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.verbose > 0:
                    print(f'\nEpoch {epoch+1}: Early stopping')

# Define your model creation function
def create_model():
    # Assume X_train_scaled is defined elsewhere in your code
    model = Sequential()
    model.add(Dense(units=int((X_train_scaled.shape[1] + 1) / 2), 
                    activation='relu', 
                    input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Define the k-fold cross-validator
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

print("Starting cross-validation...")

# Convert X_train_scaled and y_train to numpy arrays if they are Pandas DataFrames or Series
X_train_scaled_np = X_train_scaled.values if isinstance(X_train_scaled, pd.DataFrame) else X_train_scaled
y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

# Perform cross-validation with progress update
accuracies = []
for i, (train, test) in enumerate(kfold.split(X_train_scaled_np, y_train_np)):
    print(f"Training on fold {i+1}/10...")
    
    # Create model for each fold
    model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=2560, verbose=1)
    
    # Define custom early stopping callback for each fold
    custom_early_stopping = CustomEarlyStopping(monitor='val_loss', min_delta=0.005, patience=2, verbose=1, mode='auto')
    
    # Fit the model for the current fold
    model.fit(X_train_scaled_np[train], y_train_np[train], callbacks=[custom_early_stopping])
    
    # Evaluate the model
    y_pred = model.predict(X_train_scaled_np[test])
    score = accuracy_score(y_train_np[test], y_pred)
    accuracies.append(score)

accuracy_mean = np.mean(accuracies)
accuracy_std = np.std(accuracies)

# Output the results
print(f'Cross-validation accuracy: {accuracy_mean} +/- {accuracy_std}')


# ### Hyperparameter Tuning with RandomizedSearchCV for Keras Neural Network Classifier

# In[146]:


from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


# Function to create the model with hyperparameters as arguments
def create_model(neurons=10, dropout_rate=0.0, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# EarlyStopping and ReduceLROnPlateau callbacks
early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.001)

# Model with callbacks
model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=320, verbose=1)

# Define hyperparameters for RandomizedSearchCV
param_distributions = {
    'neurons': [10, 20, 30, 40, 50],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'learning_rate': [0.001, 0.005, 0.01]
}

# Using RandomizedSearchCV with n_jobs=-1 for parallel processing
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_jobs=-1, cv=2, n_iter=5, verbose=1)
random_search_result = random_search.fit(X_train_scaled, y_train)

# Summarize results
print("Best: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))


# ### Rebuilding the model with the best parameters from RandomizedSearchCV

# In[147]:


# Rebuild the model with the best parameters from RandomizedSearchCV
best_neurons = 10
best_learning_rate = 0.01
best_dropout_rate = 0.1

model = Sequential([
    Dense(best_neurons, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(best_dropout_rate),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=best_learning_rate), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model on the full training dataset
model.fit(X_train_scaled, y_train, epochs=30, batch_size=2560, verbose=1)

# Use the model to make predictions
probabilities = model.predict(X_valid_scaled)

# The probabilities array will contain the probability of each loan being paid in full according to the best parameters


# In[148]:


# Convert probabilities to binary predictions using a 0.5 threshold
# Adjust this threshold based on requirements
threshold = 0.5
predicted_labels = np.where(probabilities >= threshold, 1, 0)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_valid, predicted_labels)

# Print the confusion matrix
print(conf_matrix)


# In[149]:


# Predict the probabilities for the positive class
y_pred = model.predict(X_valid_scaled)

# Construct the results DataFrame
results = pd.DataFrame({'Actual': y_valid,
                        'p(0)': [1 - p[0] for p in y_pred],
                        'p(1)': [p[0] for p in y_pred]})
results = results.sort_values(by=['p(1)'], ascending=False)
# Confusion matrix
classificationSummary(result.actual, result.predicted)


# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
ax = gainsChart(results.Actual, ax=axes[0])  # Assuming gainsChart is a defined function
ax.set_ylabel(' ')
ax.set_title('Cumulative Gains Chart')
ax = liftChart(results.Actual, ax=axes[1], labelBars=False)  # Assuming liftChart is a defined function
ax.set_ylabel('Lift')
plt.tight_layout()
plt.show()


# In[150]:


# Try different cutoff values
cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = []
specificity_scores = []
youden_scores = []
fpr_values = []
tpr_values = []

for cutoff in cutoffs:
    # Get the predicted probabilities for the positive class
    y_pred_prob = model.predict(X_valid_scaled).flatten()
    y_pred = (y_pred_prob > cutoff).astype(int)

    # Compute confusion matrix and calculate sensitivity and specificity
    conf_matrix = confusion_matrix(y_valid, y_pred)
    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])  # True Positive Rate
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])  # True Negative Rate
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)
    
    # Calculate Youden's J statistic
    youden = sensitivity + specificity - 1
    youden_scores.append(youden)

    # Calculate ROC curve points
    false_positive_rate, true_positive_rate, _ = roc_curve(y_valid, y_pred_prob)
    fpr_values.append(false_positive_rate)
    tpr_values.append(true_positive_rate)


# Plot the ROC curve
plt.figure(figsize=(8, 8))
for i in range(len(cutoffs)):
    plt.plot(fpr_values[i], tpr_values[i], label=f'Cutoff: {cutoffs[i]:.2f}')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('ROC Curve for Different Cutoff Values')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Find the cutoff that maximizes Youden's J statistic
best_cutoff = cutoffs[np.argmax(youden_scores)]

# Print the best cutoff
print(f"Best Cutoff: {best_cutoff}")


# In[151]:


# Define costs
cost_fp = -5 * 0.05
cost_tp = 0.05

# Compute probabilities once
probabilities = model.predict(X_valid_scaled)

# Define cutoffs and initialize arrays
cutoffs = np.arange(0.1, 1.0, 0.1)
sensitivity_scores = np.zeros_like(cutoffs)
specificity_scores = np.zeros_like(cutoffs)
youden_scores = np.zeros_like(cutoffs)
net_profit = np.zeros_like(cutoffs)
accuracy_scores = np.zeros_like(cutoffs)

# Vectorized computation for each cutoff
for i, cutoff in enumerate(cutoffs):
    y_pred = (probabilities > cutoff).astype(int)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    sensitivity_scores[i] = sensitivity
    specificity_scores[i] = specificity
    youden_scores[i] = sensitivity + specificity - 1
    net_profit[i] = cost_fp * conf_matrix[0, 1] + cost_tp * conf_matrix[0, 0]

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    accuracy_scores[i] = accuracy

    # Print results
    print(f"Cutoff: {cutoff:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Accuracy: {accuracy:.2%}, Net Profit: {net_profit[i]:.2f}")

#Filter for total cost > 0 using NumPy arrays
mask = net_profit > 0
filtered_cutoffs = cutoffs[mask]
filtered_sensitivity_scores = sensitivity_scores[mask]
filtered_specificity_scores = specificity_scores[mask]
filtered_net_profit = net_profit[mask]
filtered_accuracy_scores = accuracy_scores[mask]

# Plot the results
plt.plot(filtered_cutoffs, sensitivity_scores[:len(filtered_cutoffs)], label='Sensitivity', marker='o')
plt.plot(filtered_cutoffs, specificity_scores[:len(filtered_cutoffs)], label='Specificity', marker='o')
plt.plot(filtered_cutoffs, accuracy_scores[:len(filtered_cutoffs)], label='Accuracy', marker='o')
plt.title('Sensitivity, Specificity, and Accuracy vs. Cutoff Value' )
plt.xlabel('Cutoff Value')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot the results for net profit (where net profit is greater than 0)
plt.plot(filtered_cutoffs, filtered_net_profit, label="Net Profit", marker='o', color='red')
plt.title("Net Profit vs. Cutoff Value")
plt.xlabel('Cutoff Value')
plt.ylabel("Net Profit")
plt.legend()
plt.show()

# Print the best cutoff and corresponding net profit
best_index = np.argmax(filtered_net_profit)
best_cutoff = filtered_cutoffs[best_index]
best_net_profit = filtered_net_profit[best_index]
print(f"Best Cutoff: {best_cutoff}")
print(f"Corresponding Net Profit: {best_net_profit:.2f}")


# In[152]:


# Use the best cutoff for predictions on the validation set
neuralnet_model_predictions = (model.predict(X_valid_scaled) > best_cutoff).astype(int)

# Define the cost matrix
cost_matrix = np.array([[0.05, -5*0.05], [0, 0]])

# Calculate net profit for elastic net model model
net_profit_neuralnet_model = calculate_net_profit(confusion_matrix(y_valid, neuralnet_model_predictions), cost_matrix)

# Calculate accuracy
accuracy_neuralnet_model = accuracy_score(y_valid, neuralnet_model_predictions)

# Display results
print(f"Accuracy (neuralnet_model): {accuracy_neuralnet_model:.2%}")
print("Confusion Matrix (neuralnet_model):\n", confusion_matrix(y_valid, neuralnet_model_predictions))
print(f"Net Profit (neuralnet_model): ${net_profit_neuralnet_model:.2f}")


# ### Comparing the models 

# In[153]:


all_model_results = {
    'Model': ['kNN', 'RandomForest', 'SingleTree', 'Bagging', 'Boosting', 'Discriminant Analysis', 'LogisticRegression', 'NeuralNets'],
    'Accuracy': [accuracy_knn, accuracy_rf, accuracy_defaultTree, accuracy_Bagging, accuracy_boost, accuracy_da_reg, accuracy_elasticnet_model, accuracy_neuralnet_model],
    'Net Profit': [net_profit_knn, net_profit_rf, net_profitdefaultTree, net_profitbagging, net_profitboost, net_profitda_reg, net_profit_elasticnet_model, net_profit_neuralnet_model]
}

# Create DataFrame
df_results = pd.DataFrame(all_model_results)

# Sorting by 'Net Profit' and then by 'Accuracy' in descending order
df_sorted = df_results.sort_values(by=['Net Profit', 'Accuracy'], ascending=[False, False])

df_sorted


# ### Report

# In this project, the initial step involved cleaning the data and addressing missing values, achieved by either dropping incomplete entries or substituting them with the mean of other values. Subsequently, categorical variables such as MIS-Status and NewExist were encoded, resulting in the creation of new variables named default and new. Additional variables, including Portion, Real-estate, and Recession, were computed, and appended to the dataset.
# An analysis of loans backed by real estate with a term of 240 months or more revealed that out of the total, 2475 defaulted, comprising 0.28% of the total loans, while 149437 loans were paid in full, representing 16.65% of the total. For loans not supported by real estate with a term of less than 240 months, 155008 loans defaulted, accounting for 17.27% of the total, whereas 590617 loans were paid in full, constituting 65.80% of the total.
# During the recession period, 32.10% of active loans defaulted, contrasting with 67.90% of active loans that were paid in full. Conversely, among loans not active during the recession, 16.64% defaulted, while 83.36% were paid in full.
# 
# Subsequently, industry default rates were analyzed. Initially, default rates were computed using six-digit NAICS codes, followed by extracting the first two digits to identify industry descriptions. The real estate and rental and leasing sector exhibited the highest default rate at 29.00%, whereas agriculture, forestry, fishing, and hunting, alongside mining, quarrying, and oil and gas extraction, reported the lowest default rates at 9%.
# Lastly, calculations were performed to determine the gross SBA and average SBA loan disbursements by industry from 1984 to 2010. Notably, industries categorized under NAICS code 55, which encompasses management of companies and enterprises, received the highest average SBA loan disbursement during the period.
# 
# Through the implementation of Recursive Feature Elimination and Random Forest Classifier for feature ranking, the predictors for the models were selected. Subsequently, the models were arranged in ascending order based on their accuracy scores, as follows:
# 
# 1.	Discriminant Analysis: Accuracy - 0.842430
# 2.	Logistic Regression: Accuracy - 0.959863
# 3.	kNN: Accuracy - 0.971221
# 4.	Bagging: Accuracy - 0.977773
# 5.	Neural Networks: Accuracy - 0.984123
# 6.	Boosting: Accuracy - 0.988675
# 7.	Single Tree: Accuracy - 0.988168
# 8.	Random Forest: Accuracy - 0.993332
# 
# 
# 
# 
# 
# 
# 
# Discriminant Analysis:
# 
# Accuracy: 0.842430
# Net Profit: $7402.45
# Interpretation: This model has a relatively lower accuracy compared to others, but it achieves a higher net profit. This suggests that although it might not classify instances as accurately as some other models, it manages to make more profitable decisions, possibly by focusing on correctly classifying instances with higher profitability.
# 
# 
# Logistic Regression:
# 
# Accuracy: 0.959863
# Net Profit: $7298.35
# Interpretation: Logistic Regression shows a high accuracy level and a decent net profit. It's a baseline model that is often used as a starting point for classification tasks. It's efficient and interpretable, making it a popular choice in various scenarios.
# 
# kNN:
# 
# Accuracy: 0.971221
# Net Profit: $7217.35
# Interpretation: kNN achieves a high accuracy level and a good net profit. It's a non-parametric algorithm that can capture complex patterns in the data. However, it might be computationally expensive, especially for large datasets.
# 
# 
# Bagging:
# 
# Accuracy: 0.977773
# Net Profit: $7212.55
# Interpretation: Bagging shows high accuracy and a good net profit. It's an ensemble technique that combines multiple models to improve performance. It reduces variance and tends to generalize well on unseen data.
# 
# 
# Neural Networks:
# 
# Accuracy: 0.984123
# Net Profit: $7190.95
# Interpretation: Neural Networks achieve very high accuracy and a good net profit. They are powerful models capable of learning complex relationships in data. However, they require significant computational resources and might be prone to overfitting without proper regularization.
# Boosting:
# 
# Accuracy: 0.988675
# Net Profit: $7123.15
# Interpretation: Boosting achieves extremely high accuracy and a good net profit. It's an ensemble technique that builds models sequentially, focusing on instances that were previously misclassified. Boosting algorithms often perform well in practice.
# 
# 
# Single Tree:
# 
# Accuracy: 0.988168
# Net Profit: $7100.95
# Interpretation: Single Trees achieve high accuracy and a good net profit. While they might not be as powerful as ensemble methods like Random Forest or Boosting, they are simple and interpretable models.
# 
# Random Forest:
# 
# Accuracy: 0.993332
# Net Profit: $7096.45
# Interpretation: Random Forest achieves extremely high accuracy and a good net profit. It's an ensemble method based on decision trees, which provides robustness and generalization performance. It's often considered one of the top-performing algorithms for classification tasks.
# 

# In[ ]:




