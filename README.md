# Naan-mudhalvan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('../data/customer_churn.csv')

# Handle missing values
df.dropna(inplace=True)

# Encode categorical variables
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df = pd.get_dummies(df, drop_first=True)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Churn', axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_scaled['Churn'] = df['Churn'].values

# Exploratory Data Analysis
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()
