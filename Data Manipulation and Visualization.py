# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 00:38:51 2017

@author: Amar Sehic
"""

import os
import sys

sys.path.append('C:/Users/Amar Sehic/Documents/Fractal/Python Web Scrapping/')


import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, scale, normalize

import numpy as np
import math
import random

import pyparsing as pp
from pyparsing import Word,alphas,nums
import liquidspacetextwrangler as ltw


###############################################################################
#   GLOBAL VARIABLES
###############################################################################

read_dir = 'C:/Users/Amar Sehic/Documents/Fractal/Python Web Scrapping/Liquidspace Raw Text/Building Weekly Data CSV RAW/'
write_dir = 'C:/Users/Amar Sehic/Documents/Fractal/Python Web Scrapping/Liquidspace Raw Text/Building Data CSV Processed/'
am_dir = 'C:/Users/Amar Sehic/Documents/Fractal/Python Web Scrapping/Liquidspace Raw Text/Building Data Amenities/'

###############################################################################
# CLASSES
###############################################################################

class RevenueDist(object):
    
    """
    Object that takes in time difference data of slots and creates a pandas dataframe,
    which has the form (slot duration, annual revenue from slot).
    """
    
    def __init__(self, time_slots, price_per_hour):
        self.price_per_hour = price_per_hour
        
        times = np.array([i for i in time_slots if not math.isnan(i) ])
        unique, counts = np.unique(times, return_counts = True)
        
        revenue_dist = list(unique*counts*self.price_per_hour)
        unique = list(unique)
        tuples = list(zip(unique, revenue_dist))
        
        labels = ['Time Slot Duration','Annual Revenue']
        self.data_frame = pd.DataFrame.from_records(tuples, columns = labels)
        
    def get_plot(self):     #Plot Annual Rev vs Time Slots
        
        f = plt.figure(figsize=(8,5))
        ax = plt.subplot()
        ax.plot(self.data_frame['Time Slot Duration'],self.data_frame['Annual Revenue'], '--r')
        plt.title("Annual Revenue vs. Slot Duration")
        plt.xlabel("Time Slot Duration (hours)")
        plt.ylabel("Annual Revenue ($)")
        plt.grid()
        plt.show()
    
    def get_cumplot(self):  #Plot Cumulative Annual Rev vs Time Slots
        
        f = plt.figure(figsize=(8,5))
        ax = plt.subplot()
        ax.plot(self.data_frame['Time Slot Duration'],self.data_frame.cumsum()['Annual Revenue'], 'b')
        plt.title("Cumulative Annual Revenue vs. Slot Duration")
        plt.xlabel("Time Slot Duration (hours)")
        plt.ylabel("Cumulative Annual Revenue ($)")
        plt.grid()
        plt.show()
                
class Tickers(object):
    
    """
    Takes in a list of strings representing the different stock tickers to be
    loaded into a pandas dataframe.
    """
    
    def __init__(self, tickers = 'all'):    # Construct class with dataframe
        
        d={}
        default_tickers = ['VNO','SLG','ESRT', 'CLI','BXP','CIO','BPY','LPT','TIER','PDM',
           'BDN','CXP','OFC','HPP', 'ARE', 'EQC','HIW']
        
        if tickers == 'all':
            tickers = default_tickers
                
        for t in tickers:
            if not isinstance(t,str):
                print('TypeValue Error: Enter the ticker symbol!')
                break
            else:
                dummy = pd.read_csv('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/'
                                +t+'.csv')
                
                d[t] = pd.Series(dummy['Close'])
        
        maximum = max([d[key].size for key in d])
        
        for key in d:                # Some columns are too short, this fills the missing space with NAN.
            temp = np.empty((maximum))
            if d[key].size != maximum:
                dist = maximum - d[key].size
                temp[:dist] = np.NAN
                temp[dist:] = d[key]
                d[key] = temp
        
        self.ticker_data = pd.DataFrame(d)
                
                
    def corr(self):                 #Display correlation matrix
    
        print(self.ticker_data.corr())
        return self.ticker_data.corr()
    
    def plot(self, tickers="all"):  #Plot tickers over time
        
        for t in self.ticker_data:
            X = self.ticker_data[t]
            plt.plot(range(len(X)),X, label = t)
        plt.title('Closing Price')
        plt.xlabel('Time (days)')
        plt.ylabel('Closing Price ($)')
        plt.legend()
        plt.grid()
        plt.show()
    
    def high_corr(self, tickers = 'all', treshold = 0.7):
        corr_table = self.ticker_data.corr()
        
        if tickers == 'all':
            for t in corr_table:
                corr_table_cut = corr_table[corr_table[t]>treshold]
                print (corr_table_cut[t])
        else:        
            for t in tickers:
                corr_table_cut = corr_table[corr_table[t]>treshold]
                print (corr_table_cut[t])
        

class KPITable(object):
    
    """
    Create a DataFrame that stores KPI data for properties in Manhattan
    """
    
    def __init__(self, split_by = 'Broad Location'):      #Construct object and split data based on chosen column
        
        d_split = {}
        
        input_file = 'Manhattan-Data-Collection' + '.csv'
        data = pd.read_csv('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/' + input_file)

        if split_by == 'Broad Location':
            for tt in data[split_by]:
                d_split[tt] = data[data['Location'] == tt + ' Market Totals']
                
        if split_by == 'Location':
            for tt in data[split_by]:
                d_split[tt] = data[data['Location'] == tt]
                
        
        self.split_table = d_split
        self.split = split_by
            

    def get_KPI(self, KPI):         #Returns dataframe with specified KPI in columns
    
        d={}
        d_split = self.split_table
        maximum = max([d_split[key].ix[:,KPI].size for key in d_split])
    
        for key in d_split:
            d[key] = list(d_split[key].ix[:,KPI])
            if len(d[key])<maximum:
                d[key].insert(0,np.NaN)
            d[key] = np.array(d[key])
        
        d = (pd.DataFrame(d, index = d_split[list(d_split.keys())[0]]['Time']),KPI)
        
        return d
    
    def plot_KPI(self, KPI):        #Plots chosen KPI against time
    
        KPI_frame = self.get_KPI(KPI)[0]
        KPI_frame.plot(grid = True, title = KPI + ' vs. Time')

class MacroVariables(object):
    '''
    Creates an object to store macroeconomic variables
    '''
    
    def __init__(self):
        
        input_file = 'Macro- NYC & US - Sheet1' + '.csv'
        data = pd.read_csv('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/' + input_file)
        self.macro_data = data

class MacroData(object):
    '''
    Instantiates the three different macroeconomic object: KPIs, Tickers
    and Macro Variables. Holds methods to compare and analyze the 
    relationships between the datasets.
    '''
    
    def __init__(self):
        
        self.KPI = KPITable()
        self.Tickers = Tickers()
        self.MacroVar = MacroVariables()
        


###############################################################################
# FUNCTIONS
###############################################################################

def poly_fit(X,Y, limit = 10):
    
    """
    Determines degree of optimal polynomial fitted with linear regression.
    """
    
    i = 0
    scores = []
    
    X = np.transpose(np.matrix(X))
    Y = np.transpose(np.matrix(Y))
    lm = linear_model.LinearRegression()
    
    while i<limit:
        s = CV_linear(X,Y,lm, deg = i+1)[1]
        if s<0 and i!=0:
            break
        else:
            scores.append(s)
            i+=1
        
    print (scores)
    print ('The best fit is of degree: ' + str(scores.index(max(scores))+1))
    return scores.index(max(scores))+1
    

def CV_linear(X,Y,model, deg = 1, size =0.2, cv = 50):
    
    """
    Performs cross validation on target linear regression model.
    """
    
    scores=[]
    
    X = PolynomialFeatures(degree=deg).fit_transform(X)
    
    for ii in range(cv):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=random.randint(0,100))
        model.fit(X_train,y_train)
        scores.append(model.score(X_test,y_test))
    
    mean = np.mean(scores)
    std = np.std(scores)
    
    print ("The average score is: {0:.2f} with a deviation of +/- {1:.2f}".format(mean,std))
    
    return scores,mean,std

def lin_reg(X,Y,deg=1):
    
    """
    General Linear regression, returns coefficients, intercept and R^2 value.
    Note that the input arrays are transposed once they enter the fuction, thus
    this only takes arrays in.
    """
    
    X = np.transpose(np.matrix(X))
    Y = np.transpose(np.matrix(Y))
    
    lm = linear_model.LinearRegression()
    
    #Pick either linear or polynomial model
    
    if deg == 1:
        lm.fit(X,Y)
    else: 
        X_Mod = PolynomialFeatures(degree=deg).fit_transform(X)
        lm.fit(X_Mod,Y)
        
    m = lm.coef_
    b = lm.intercept_
    R2 = metrics.r2_score(Y,lm.predict(X_Mod))

    print ("The R Squared is:" + str(R2))
    print (m,b)
    
    return m,b,R2,lm
    
def gen_hist(X, name):
    
    """
    Generates a histogram for the array X, saves it with the specified name
    
    """
    
    rows = X
 
    f2, axarr2 = plt.subplots(1, len(rows), sharey = True)
       
    for i in range(len(rows)):
        axarr2[i].hist(rows[i], bins = 30)
        axarr2[i].set_title('Histogram ' + str(i))
        axarr2[i].grid()
    
    f2.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.ylabel("Frequency")
    
    plt.savefig(name + ' - Histograms.jpg')
    plt.show()

def gen_lin_fit(X, Y, name):
    
    """
    Generates a linear fit of X to Y, saves it with the specified name.
    
    """
    
    rows = X
    f1, axarr  = plt.subplots(1, len(rows) ,sharey = True)

    for i in range(len(rows)):
        x = range(max(rows[i]))
        m,b,R = lin_reg(rows[i],Y)
        
        axarr[i].scatter(rows[i],Y)
        axarr[i].plot(x,list(map(lambda y: y*m+b,x)), 'r')
        if rows[i] == Y :
            axarr[i].set_title('Dummy')
        else:
            axarr[i].set_title('Scatter Plot Test ' + str(i))
        axarr[i].grid()
    
    plt.savefig(name + ' - Linear Fit.jpg')
    plt.show()

def gen_2Dlinfit(X1,X2,Z,name):

    """
    Generates a linear fit of X1 and X2 to Y, saves it with the specified name.
    
    """
    
    rows=[]
    rows.append(Z)
    rows.append(X1)    
    rows.append(X2)   
        
        
    doublet = np.array(list(zip(rows[1],rows[2]))).T
    
    m,b,R = lin_reg(doublet, rows[0])
    
    X,Y = rows[1],rows[2]
    X, Y = np.meshgrid(np.linspace(X.min(), X.max(), 100), 
                           np.linspace(Y.min(), Y.max(), 100))
    
    fig = plt.figure(figsize =(8,8) )
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(rows[1]),np.array(rows[2]),np.array(rows[0]), c= 'k')
    ax.plot_surface(X,Y, np.array(b+m[0][0]*X+m[0][1]*Y ).astype(float), color = 'b')
    ax.view_init(elev=60,)
    
    plt.savefig(name + ' - 2D Linear Fit.jpg')
    plt.show()
        

###############################################################################
#   MAIN PROGRAM BODY
###############################################################################

a = os.listdir(write_dir)

revenues = []
names = []
zips = []


DF = []




for file in a:
    res = ltw.get_address(file)
    zips.append(int(res[1]))
    data = pd.read_csv(write_dir+file)
    total = data['Revenue'].sum()
    revenues.append(total)
    names.append(res[0])

for file in a:

    DF.append(pd.read_csv(write_dir + file))

for df in DF:
    cleaned = []
    for time in df['Time Used']:
        if time == 'Closed':
            cleaned.append(2)
        else:
            cleaned.append(time)
    df['Time Used'] = cleaned
    print(df)


for df in DF:
    occupancy = []
    for time in df['Time Used']:
        if time == 2:
            occupancy.append(time)
        elif math.isnan(time) or time == 0 :
            occupancy.append(False)
        else:
            occupancy.append(True)
    df['Occupancy'] = occupancy
    print(df)


'''
for ii in range(len(DF)):
        build_sums = []
        for time in timeline:
           k = DF[ii]
           k = k[k['Timeline'] == time]
           s = k.sum(0)['Revenue']
           if s==None:
               s = 0
           build_sums.append(s)
       
        plt.hist(build_sums, bins = 40)
        plt.grid()
        plt.show()
            
'''


'''

filename = 'BUILDING_DETAILS_amenities_address_rating_etc Amar copy'

for file in a:
    res = get_address(file)
    zips.append(int(res[1]))
    data = pd.read_csv(write_dir+file)
    total = data['Revenue'].sum()
    revenues.append(total)
    names.append(res[0])

k = pd.Series(revenues, index = names)


d ={'Zip':zips,
    'Revenue': revenues}

data = pd.DataFrame(d)
amens = pd.read_csv(am_dir+filename+'.csv')
amens = amens.sort_values(['Name of Building'])

amens['Revenue'] = data['Revenue']

data = pd.DataFrame(d, index = names)

amens = amens[amens['Revenue']<1000000]
amens = amens.sort_values('Zip Code')


summed = []

for am in amens['Zip Code']:
    d = amens[amens['Zip Code']==am]
    summed.append(d.sum(0)['Revenue'])
    
plt.bar(list(amens['Zip Code']), summed)
plt.grid()
plt.title('Revenue per Zipcode')
plt.xlabel('Zipcode')
plt.ylabel('Revenue ($)')
plt.show()
 
max_zip = amens['Zip Code'][summed.index(max(summed))]
print(max(summed),max_zip)

print(amens.corr())
'''
