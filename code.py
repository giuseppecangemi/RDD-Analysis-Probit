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
        
data_1 = np.array(data_1)     

data_amt = [data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10]

plt.boxplot(data_amt)
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], 
           ['B+', 'BB-', 'BB', 'BB+', 'BBB-', 'BBB', 'BBB+', 'A-'])
############################################################
############################################################

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

############################################################
############################################################

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
############################################################
############################################################
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


############################################################
############################################################
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






