# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:12:21 2017

@author: Amar Sehic
"""

import pandas as pd

import numpy as np
import math
import random


adresses = pd.read_csv('C:/Users/Amar Sehic/Downloads/Address-for-Phython.csv')
sales = pd.read_csv('C:/Users/Amar Sehic/Downloads/Office-Rolling-Sale-2003-2017.csv')

freq_count = []

for address in adresses['Address']:
    count = 0
    for add2 in sales['ADDRESS']:
        if add2 == address:
            count+=1
    freq_count.append(count)

adresses['Frequency Count'] = pd.Series(freq_count)

print(adresses[adresses['Frequency Count']>0])