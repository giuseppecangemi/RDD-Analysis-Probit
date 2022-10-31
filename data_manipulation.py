#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:04:09 2022

@author: giuseppecangemi
"""

import pandas as pd
import numpy as np

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

