# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:24:06 2017

@author: Amar Sehic
"""

import os
import pandas as pd
import pyparsing as pp
from pyparsing import Word,alphas,nums
import numpy as np
import time as TT
import timeit

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
        time = s[0]
        period = s[1]
        value = " ".join(s[2:])
        prices_arr.append((time,period,value))
        
    
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


def unwrap(input_file):
    '''
    Unwraps the complicated processed text file from raw_parse and creates a 
    dataframe that stores all the week's data.
    '''
    offices_arr = []
    
    RP = raw_parse(input_file)
    
    which_week = RP[0] 
    prices = RP[1]
    mess = RP[2]
    
    for ii in range(8):
        week_arr = []
        for week in mess:
            day = week[ii]
            week_arr.append(day)
        offices_arr.append((prices[ii],week_arr))
    
    MESS = offices_arr

#This is the structure of MESS    
#MESS[office][price_loc/schedule][(price,period,name)/(Sun,Mon...,Sat)][NaN/(day,month,date,status)]
    
    price_pd = []
    hours_pd = []
    name_pd=[]
    day_of_week_pd = []
    month_pd = []
    day_date_pd = []
    time_diffs = []
    
    
    for l in range(len(offices_arr)):
        
        for k in range(7):
            price_hour = MESS[l][0]
            price_pd.append(price_hour[0])
            hours_pd.append(price_hour[1])
            name_pd.append(price_hour[2])
            
       
            office = MESS[l]
            temp_day = office[1][k][0]
            temp_month = office[1][k][1]
            temp_date = office[1][k][2]
            
            day_of_week_pd.append(temp_day)
            month_pd.append(temp_month)
            day_date_pd.append(temp_date)
    
       
            test = list(MESS[l][1][k])
            if (len(test)<6):
                time_diffs.append(np.NaN)
            else:
                time = test[3:]
                time_format = [TT.strptime(time[ii]+time[ii+1], "%I:%M%p").tm_hour+(TT.strptime(time[ii]+time[ii+1], "%I:%M%p").tm_min)/60 for ii in range(0,len(time),2)]
                result = [abs(time_format[ii]-time_format[ii+1]) for ii in range(0,len(time_format),2)]
                result = sum(result)
                time_diffs.append(result)
        
    d = {'Day': day_of_week_pd,
         'Date': day_date_pd,
         'Month': month_pd,
         'Name': name_pd,
         'Price': price_pd,
         'Period': hours_pd,
         'Time Used': time_diffs}
    
    d = pd.DataFrame(d, index = d['Month'])  #Choose index as appropriate
    
    return d,which_week
    

###############################################################################
#   MAIN PROGRAM BODY
###############################################################################

input_file = 'test_amar-2'+'.csv'

d = unwrap(input_file)

print(d)


    
