#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:04:09 2022

@author: giuseppecangemi
"""

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import math
from numpy import nan
from numpy import mean
from statsmodels.miscmodels.ordinal_model import OrderedModel
from rdd import rdd

df = pd.read_excel("/Users/giuseppecangemi/Desktop/dati_RDD/df_yield_2016->.xlsx")
df = df.dropna()  

df["cat"] = 1    
for i, row in df.iterrows():
    if row["sp_rating"] == "CCC":
        df["cat"][i]= 0
    elif row["sp_rating"] == "B-":
        df["cat"][i]= 1
    elif row["sp_rating"] == "B":
        df["cat"][i]= 2
    elif row["sp_rating"] == "B+":
        df["cat"][i]= 3
    elif row["sp_rating"] == "BB-":
        df["cat"][i]= 4
    elif row["sp_rating"] == "BB":
        df["cat"][i]= 5 
    elif row["sp_rating"] == "BB+":
        df["cat"][i]= 6
    elif row["sp_rating"] == "BBB-":
        df["cat"][i]= 7
    elif row["sp_rating"] == "BBB":
        df["cat"][i]= 8
    elif row["sp_rating"] == "BBB+":
        df["cat"][i]= 9
    elif row["sp_rating"] == "A-":
        df["cat"][i]= 10


df["group"] = np.where(df["cat"]>=7, "Treatment", "Control")   
for i in df["group"]:
    if i == "Treatment":
        print(i)
        
        
        

data_0 = list()            
for i, row in df.iterrows():
    if row["cat"] == 0:
        data_0.append(row["oas"])
data_0 = np.array(data_0)
data_1 = list()        
for i, row in df.iterrows():
    if row["cat"] == 1:
        data_1.append(row["oas"])
data_1 = np.array(data_1)
data_2 = list()        
for i, row in df.iterrows():
    if row["cat"] == 2:
        data_2.append(row["oas"])
data_2 = np.array(data_2)
data_3 = list()        
for i, row in df.iterrows():
    if row["cat"] == 3:
        data_3.append(row["oas"])
data_3 = np.array(data_3)
data_4 = list()        
for i, row in df.iterrows():
    if row["cat"] == 4:
        data_4.append(row["oas"])
data_4 = np.array(data_4)
data_5 = list()        
for i, row in df.iterrows():
    if row["cat"] == 5:
        data_5.append(row["oas"])  
data_5 = np.array(data_5)
data_6 = list()        
for i, row in df.iterrows():
    if row["cat"] == 6:
        data_6.append(row["oas"])  
data_6 = np.array(data_6)
data_7 = list()        
for i, row in df.iterrows():
    if row["cat"] == 7:
        data_7.append(row["oas"]) 
data_7 = np.array(data_7)
data_8 = list()        
for i, row in df.iterrows():
    if row["cat"] == 8:
        data_8.append(row["oas"])    
data_8 = np.array(data_8)
data_9 = list()        
for i, row in df.iterrows():
    if row["cat"] == 9:
        data_9.append(row["oas"])  
data_9 = np.array(data_9)
data_10 = list()        
for i, row in df.iterrows():
    if row["cat"] == 10:
        data_10.append(row["oas"])
data_10 = np.array(data_10)       
                                 
        
#data_1 = np.array(data_1)     

data_nan = [data_0, data_1, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10]


data_0 = data_0[np.logical_not(np.isnan(data_0))]
print(data_0)
data_1 = data_1[np.logical_not(np.isnan(data_1))]
print(data_1)
data_2 = data_2[np.logical_not(np.isnan(data_2))]
print(data_2)
data_3 = data_3[np.logical_not(np.isnan(data_3))]
print(data_3)
data_4 = data_4[np.logical_not(np.isnan(data_4))]
print(data_4)
data_5 = data_5[np.logical_not(np.isnan(data_5))]
print(data_5)
data_6 = data_6[np.logical_not(np.isnan(data_6))]
print(data_6)
data_7 = data_7[np.logical_not(np.isnan(data_7))]
print(data_7)
data_8 = data_8[np.logical_not(np.isnan(data_8))]
print(data_8)
data_9 = data_9[np.logical_not(np.isnan(data_9))]
print(data_9)
data_10 = data_10[np.logical_not(np.isnan(data_10))]
print(data_10)

#boxplot intero campione:
data = [data_0, data_1, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10]

plt.boxplot(data)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
           ['B-', 'B', 'B+', 'BB-', 'BB', 'BB+', 'BBB-', 'BBB', 'BBB+', 'A-'])

#boxplot da B+ in poi:
data1 = [ data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10]

plt.boxplot(data1)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], 
           [ 'B+', 'BB-', 'BB', 'BB+', 'BBB-', 'BBB', 'BBB+', 'A-'])
plt.ylabel("Bond Spread")
plt.xlabel("Rating")

############################################################
############################################################
############################################################
############################################################
        
data_1 = np.array(data_1)     

data_amt = [data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10]

plt.boxplot(data_amt)
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], 
           ['B+', 'BB-', 'BB', 'BB+', 'BBB-', 'BBB', 'BBB+', 'A-'])
#############################
#############################

from pandas.api.types import CategoricalDtype
#creo l'etichetta trattamento/controllo:
cat_type = CategoricalDtype(categories=['Control','Treatment'], ordered=True)
df["cut"] = df["group"].astype(cat_type)

#analizzo il numero di bond dei due differenti gruppi:
count_control = len(df[df["group"]=="Control"])
count_treatment = len(df[df["group"]=="Treatment"])

#calcolo la percentuale dei due differenti gruppi all'interno del campione:
perc_control = count_control/(count_control+count_treatment)
print("% gruppo controllo:", perc_control*100)
perc_treatment = count_treatment/(count_control+count_treatment)
print("% gruppo trattamento:", perc_treatment*100)

#analizzo il numero di bond all'interno delle due differenti classi: Control/Treatment \\
    #usando un countplot:
df["group"].value_counts()
sns.countplot(x="group", data=df, palette="hls")
plt.title("N° of Bonds falling on the two different group")
plt.xlabel("Group")
plt.show()


#runno modello probit per stimare quale è la probabilità di rientrare \\
    #nel gruppo del trattamento rispetto lo spread:
mod_prob = OrderedModel(df['cut'],
                        df[[ "oas" ]],
                        distr='probit')
res_prob = mod_prob.fit(method='bfgs')
res_prob.summary()

yhat = res_prob.model.predict(res_prob.params, exog=df[["oas"]])
yhat


#creo grafico probit:
ln_oas = np.log(df["oas"])
sns.regplot(x=ln_oas, y=df["dummy"], data=df, logistic=True, ci=None)


df["ln_rev"] = np.log(df["revenue"])

###############
###############

x = df["cat"]        
threshold = 7
treatment = np.where(x >= threshold, 1, 0)

#####CON OPT. BANDWIDTH
bandwidth_opt = rdd.optimal_bandwidth(df['oas'], df['cat'], cut=threshold)
print("Optimal bandwidth:", bandwidth_opt)

data_rdd = df
data_rdd = rdd.truncated_data(df, "cat", bandwidth_opt, cut=threshold)

plt.figure(figsize=(12, 8))
plt.scatter(data_rdd['cat'], data_rdd['oas'], facecolors='none', edgecolors='r')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(x=threshold, color='b')
plt.show()
#########################
#########################
# SENZA OPT. BANDWIDTH:
bandwidth_opt = 4
data_rdd = df
data_rdd = rdd.truncated_data(df, "cat", bandwidth_opt, cut=threshold)

plt.figure(figsize=(12, 8))
plt.scatter(data_rdd['cat'], data_rdd['oas'], facecolors='none', edgecolors='r')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(x=threshold, color='b')
plt.show()


#########################
#########################
import plotnine as p
mod = smf.ols(formula='oas ~ 1 + cat*C(group) + cat', data=df)
res = mod.fit()
print(res.summary())

df['D'] = 0
df.loc[df.cat>=7, 'D'] = 1

df['y1'] = 2026.2170 + 0*df.D + -345.1769 * df.cat 

df['y2'] = 2026.2170 -1592.7145*df.D + 314.9720 * df.cat 


p.ggplot(df, p.aes(x='cat', y="oas", color = 'factor(D)')) +\
    p.geom_point(alpha = 0.5) +\
    p.geom_vline(xintercept = 6.5, colour = "grey") +\
    p.stat_smooth(method = "lm") +\
    p.labs(x = "Test score (X)", y = "Potential Outcome (Y1)")
    

p.ggplot(df, p.aes(x='cat', y='y1', color = 'factor(D)')) +\
    p.geom_point(alpha = 0.5) +\
    p.geom_vline(xintercept = 6.5, colour = "grey") +\
    p.stat_smooth(method = "lm", se = 'F') +\
    p.labs(x = "Test score (X)", y = "Potential Outcome (Y)")   
    

sns.scatterplot(df["cat"], np.log(df["oas"]))






