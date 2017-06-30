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

def raw_parse(filename):
    '''
    The first function in the parsing pipeline. Takes in the raw block of data
    from the CSV file and parses the text to return two arrays:
        
        week - A string to identify the week
        prices_arr - Tuples of prices and offices names
        timeslot_arr - Big array encoding office usage over the week, hit this
                       with the unwrap function to make sense of it
        
    '''
    input_file = filename
    data = pd.read_csv('C:/Users/Amar Sehic/Documents/Fractal/Python Web Scrapping/Liquidspace Raw Text/'+input_file)
    d = data['table']
    d = d[0]
    d = d.replace(u'\xa0', ' ')  #Encoding is messed up, 
                                 #this character \xa0 causes problems if not removed
    
    remove = Word(alphas) + pp.Literal('This reservation was created in different reservation mode for workspace, please correct.')
    
    date = Word( alphas ) + Word(nums) +'-' + Word( alphas ) + Word(nums) + pp.Suppress(',') + pp.Suppress(Word(nums))
    location = Word( pp.alphanums+'-()#.' )
    price_loc = pp.Suppress('$') + Word(nums) + pp.Suppress('/') + Word(alphas) + pp.OneOrMore(location)
    
    week_date = date.parseString(d)
    prices_loc = price_loc.searchString(d)
    
    week = week_date[0]+ ' ' +week_date[1] + ' ' + week_date[2] + ' ' + week_date[3]+ ' '+week_date[4]
    
    start_day = week_date[1]
    
    
    prices_arr = []   #One of the two arrays returned, returns tuples of price per
                      # time period and the name of the office
    
    for s in prices_loc:
        key = s[0] +' '+s[1]
        value = " ".join(s[2:])
        prices_arr.append((key,value))
        
    
    timeslot_arr = []   #The second array returned, contains a big mess of data,
                        #basically encodes all the usage info for each office over the week.
                        #Call the unwrap function on this beast to make sense of it
    
    for ii in range(7):
        day = int(start_day) +ii
        
        timeslot = pp.OneOrMore(Word(nums+':') + Word(alphas) + pp.Suppress('-') + Word(nums+':') + Word(alphas) + pp.Suppress(remove))
        extra = pp.Or([Word(alphas) + Word(alphas), timeslot, pp.Empty()]) 
        start = Word(alphas) + pp.Suppress(',') + Word( alphas) + str(day) + extra 
        
        dd = start.searchString(d)
        timeslot_arr.append(dd)

    return week, prices_arr,timeslot_arr


###############################################################################
#   MAIN PROGRAM BODY
###############################################################################

input_file = 'test_amar-2'+'.csv'

print (raw_parse(input_file))
