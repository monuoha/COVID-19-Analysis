#!/usr/bin/env python
# coding: utf-8

# Link to the COVID-19 Dataset: https://ourworldindata.org/covid-cases

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
from prophet import Prophet  # install the prophet package if necessary

#note: you will have to update the data set using the link above every month to observe significant changes in the forecasted values
file = '/Users/michaelonuoha/Downloads/owid-covid-data.csv'  # change the directory to load the data set

covidData = pd.read_csv(file)
covidData = covidData[['location', 'date', 'new_cases', 'new_deaths', 'new_vaccinations']]
covidData.set_index(['location'], inplace = True)
covidData['date'] = pd.to_datetime(covidData['date'])
covidData


# # Time Series Forecasting of COVID-19 Cases and Deaths

# In[2]:


plt.rcParams['figure.figsize'] = [10, 5]
plt.style.use('ggplot')
def covid_over_time(region, var):
    '''This function creates a time series plot of one of the following variables that the user chooses to analyze 
for a particular region: cases, deaths, and vaccinations. 
region: The region on which the user would like to focus 
var: The variable on which the user would like to focus (valid inputs are new_cases, new_deaths & 
new_vaccinations) 
Output: A time series plot along with the table of values'''
    if region not in covidData.index and var not in covidData.columns:
        return "Invalid region and variable entry."
    elif var not in covidData.columns:
        return "Invalid variable entry."
    elif region not in covidData.index:
        return "Invalid region entry."
    covid_nation = covidData.loc[region]
    plt.xticks(rotation = 45)
    plt.xlabel("Date")
    if var == 'new_cases':
        plt.ylabel("Cases")
        plt.title("COVID-19 Cases in {0}".format(region), fontsize = 18)
    elif var == 'new_deaths':
        plt.ylabel("Deaths")
        plt.title("COVID-19 Deaths in {0}".format(region), fontsize = 18)
    elif var == 'new_vaccinations':
        plt.ylabel("Vaccinations")
        plt.title("COVID-19 Vaccinations in {0}".format(region), fontsize = 18)
    plt.plot(covid_nation['date'], covid_nation[var], color = 'b');
    return covid_nation[['date', var]]


# In[3]:


def covid_forecasting(region, var, days):
    '''This function forecasts one of the following variables that the user chooses to analyze for a particular 
region: cases, deaths, and vaccinations. 
region: The region on which the user would like to focus 
var: The variable on which the user would like to focus (valid inputs are new_cases, new_deaths & 
new_vaccinations) 
days: The number of days for which the user would like to predict values 
Output: A time series forecast plot along with the table of forecasted values'''
    if region not in covidData.index and var not in covidData.columns:
        return "Invalid region and variable entry."
    elif var not in covidData.columns:
        return "Invalid variable entry."
    elif region not in covidData.index:
        return "Invalid region entry."
    covid_nation = covidData.loc[region]
    targetData = covid_nation[['date', var]].copy()
    targetData.rename(columns = {'date':'ds', var:'y'}, inplace = True)
    forecast = Prophet(daily_seasonality = True)
    forecast.fit(targetData)
    future_values = forecast.make_future_dataframe(periods = days, include_history = False)
    forecast_values = forecast.predict(future_values)
    
    #Negative values for the forecasted values as well as the upper and lower bounds are replaced with zeros because 
    #the number of cases, deaths, and vaccinations cannot be negative 
    invalid_val = [n for n in forecast_values.yhat if n < 0]
    invalid_val_lower = [n for n in forecast_values.yhat_lower if n < 0]
    invalid_val_upper = [n for n in forecast_values.yhat_upper if n < 0]
    forecast_values = forecast_values.replace(invalid_val, [0]*len(invalid_val))
    forecast_values = forecast_values.replace(invalid_val_lower, [0]*len(invalid_val_lower))
    forecast_values = forecast_values.replace(invalid_val_upper, [0]*len(invalid_val_upper))
    
    plt.xticks(rotation = 45)
    plt.xlabel("Date")
    if var == 'new_cases':
        plt.ylabel("Cases")
        plt.title("Forecasted COVID-19 Cases in {0}".format(region), fontsize = 18)
    elif var == 'new_deaths':
        plt.ylabel("Deaths")
        plt.title("Forecasted COVID-19 Deaths in {0}".format(region), fontsize = 18)
    elif var == 'new_vaccinations':
        plt.ylabel("Vaccinations")
        plt.title("Forecasted COVID-19 Vaccinations in {0}".format(region), fontsize = 18)
    plt.plot(covid_nation['date'], covid_nation[var], color = 'b', label = 'Numbers so far')
    plt.plot(forecast_values.ds, forecast_values.yhat, color = 'g', label = 'Forecasted numbers')
    plt.plot(forecast_values.ds, forecast_values.yhat_lower, color = 'r', label = 'Lower bound')
    plt.plot(forecast_values.ds, forecast_values.yhat_upper, color = 'y', label = 'Upper bound')
    plt.legend();
    return forecast_values[['ds','yhat', 'yhat_lower', 'yhat_upper']]


# In[4]:


covid_over_time('United Kingdom', 'new_cases')


# In[5]:


covid_forecasting('United Kingdom', 'new_cases', 365)


# In[6]:


covid_forecasting('United Kingdom', 'new_deaths', 365)


# In[7]:


covid_forecasting('United Kingdom', 'new_vaccinations', 365)


# In[8]:


help(covid_over_time)


# In[9]:


help(covid_forecasting)

