# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 13:56:47 2017

@author: Amar Sehic
"""

import pandas as pd

import numpy as np
import math
import random


file_init = pd.read_csv('C:/Users/Amar Sehic/Documents/Rubik/Python/Python Data Analysis/Rolling Sales Brooklyn multifamily -anomalie 2003 -2017 (2).csv')
hoods = list(set(file_init.ix[:,'NEIGHBORHOOD']))
yearz = list(set(file_init.ix[:,'SALE YEAR']))
yearz.sort()
trig = 0

print(yearz)

sales = []

for nbhd in hoods:
        file = file_init[file_init['NEIGHBORHOOD']== nbhd]

        for year in yearz:
            
            year_picked = file[file['SALE YEAR']==year]
            year_picked[year_picked['GROSS SQUARE FEET']>0]
            sales.append(year_picked['PRICE PER SQUARE FEET'].mean())
            print(year_picked['PRICE PER SQUARE FEET'].mean())
            
        
        if trig == 1:
            df[nbhd]=pd.Series(sales)
            sales = []
        else:
            sales = pd.Series(sales)
            d = {'Year':yearz,
                 nbhd : sales
                    }
            df = pd.DataFrame(d)
            trig = 1
            sales = []


df.to_csv('C:/Users/Amar Sehic/Documents/Rubik/Python/Python Data Analysis/Aggregated Brooklyn Multifamily Avg Price Sq Ft.csv')
