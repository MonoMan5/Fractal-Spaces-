# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:24:06 2017

@author: Amar Sehic
"""

import os
import pandas as pd
import pyparsing as pp
from pyparsing import Word,alphas,nums

###############################################################################
#   CLASSES
###############################################################################



###############################################################################
#   FUNCTIONS
###############################################################################



###############################################################################
#   MAIN PROGRAM BODY
###############################################################################

input_file = 'test_amar-2'+'.csv'
data = pd.read_csv('C:/Users/Amar Sehic/Documents/Fractal/Python Web Scrapping/Liquidspace Raw Text/'+input_file)
d = data['table']
d = d[0]

date = Word( alphas ) + Word(nums) +'-' + Word( alphas ) + Word(nums) + pp.Suppress(',') + pp.Suppress(Word(nums))
location = Word( pp.alphanums+'-()#.' )
price_loc = pp.Suppress('$') + Word(nums) + pp.Suppress('/') + Word(alphas) + pp.OneOrMore(location)

week_date = date.parseString(d)
prices_loc = price_loc.searchString(d)

week = week_date[0]+ ' ' +week_date[1] + ' ' + week_date[2] + ' ' + week_date[3]+ ' '+week_date[4]

prices_arr = []

for s in prices_loc:
    key = s[0] +' '+s[1]
    value = " ".join(s[2:])
    prices_arr.append((key,value))
    

 
