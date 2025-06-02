
# COVID-19 Global EDA - Full Updated Version

## 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

## 2. Load Dataset
df = pd.read_csv('../data/covid_19_data.csv')

# Convert date column and fix errors
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'], errors='coerce')
df = df.dropna(subset=['ObservationDate'])

## 3. Basic Info
print(df.info())
print(df.describe())

## 4. Missing Values
print("Missing values per column:")
print(df.isnull().sum())

## 5. Overview
print("Unique countries:", df['Country/Region'].nunique())
print("Date range:", df['ObservationDate'].min(), "to", df['ObservationDate'].max())

## 6. Global Trends Over Time
daily = df.groupby('ObservationDate')[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(daily['ObservationDate'], daily['Confirmed'], label='Confirmed')
plt.plot(daily['ObservationDate'], daily['Deaths'], label='Deaths')
plt.plot(daily['ObservationDate'], daily['Recovered'], label='Recovered')
plt.legend()
plt.title('Global COVID-19 Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 7. Top Countries by Confirmed Cases
latest_date = df['ObservationDate'].max()
latest = df[df['ObservationDate'] == latest_date]

country_total = latest.groupby('Country/Region')[['Confirmed', 'Deaths', 'Recovered']].sum()
top_countries = country_total.sort_values(by='Confirmed', ascending=False).head(15)

fig = px.bar(top_countries.reset_index(), x='Country/Region', y='Confirmed', color='Country/Region',
             title='Top 15 Countries by Confirmed Cases', text='Confirmed')
fig.show()

## 8. Recovery vs Death Scatter Plot
fig = px.scatter(top_countries.reset_index(), 
                 x='Recovered', y='Deaths', 
                 color='Country/Region', size='Confirmed',
                 title='Recovered vs Deaths by Country')
fig.show()

## 9. Correlation Heatmap
corr = df[['Confirmed', 'Deaths', 'Recovered']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Between COVID-19 Metrics')
plt.show()

## 10. Insights (To be added manually in notebook markdown)
# - Most confirmed cases are from countries X, Y, Z
# - Recovery rate was increasing steadily from mid-2020
# - Death rate remained relatively low compared to confirmed cases
# - Strong positive correlation between confirmed and recovered cases
# - Slight data anomalies in recovered column (spikes, drops)
