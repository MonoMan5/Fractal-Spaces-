# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 13:56:47 2017

@author: Amar Sehic
"""

import pandas as pd

import numpy as np
import math
import random


file_init = pd.read_csv('C:/Users/Amar Sehic/Documents/Rubik/Python/Python Data Analysis/FINAL Rolling Sales Manhattan Multifamily -anomalie 2003 -2017.csv')
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
           
            sales.append(year_picked['GROSS SQUARE FEET'].sum())
            
        
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


df.to_csv('C:/Users/Amar Sehic/Documents/Rubik/Python/Python Data Analysis/Aggregated Manhattan Multifamily Total GSF.csv')
