# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 00:38:51 2017

@author: Amar Sehic
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


import numpy as np
import statistics as stat
from mpl_toolkits.mplot3d import Axes3D
import math
import random

###############################################################################
#CLASSES
###############################################################################

class Revenue_Dist(object):
    
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
    
    def __init__(self, tickers):    # Construct class with dataframe
        
        d={}
                
        for t in tickers:
            if not isinstance(t,str):
                print('TypeValue Error: Enter the ticker symbol!')
                break
            else:
                dummy = pd.read_csv('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/'
                                +t+'.csv')
                d[t] = pd.Series(dummy['Close'])
            
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
    


###############################################################################
#FUNCTIONS
###############################################################################

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
    
    return m,b,R2
    
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


input_file = '419-Park-Ave-S-2nd-Floor-NY-10016_1-year-till_jun-24-17' + '.csv'

data = pd.read_csv('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/' + input_file)
time_slots  = data['Time difference'] = data['Time difference'].apply(lambda x: x/60)
series = Revenue_Dist(time_slots, 75)

Xp = series.data_frame['Time Slot Duration']
Yp = series.data_frame.cumsum()['Annual Revenue']

X = np.transpose(np.matrix(Xp))
Y = np.transpose(np.matrix(Yp))

lm = linear_model.LinearRegression()

'''
Tp = np.linspace(0,max(Xp),len(Xp))
Tp = np.reshape(Tp,(26,1))

for i in range(5,6):
    X_Mod = PolynomialFeatures(degree=i).fit_transform(X)
    T = PolynomialFeatures(degree=i).fit_transform(Tp)
    lm.fit(X_Mod,Y)
    plt.plot(Tp,lm.predict(T),'b')
    R2 = metrics.r2_score(Y,lm.predict(X_Mod))
    print ("The R Squared is:" + str(R2))

plt.plot(X,Y,'--r')
plt.xlim(0,10)
plt.grid()
'''

X = PolynomialFeatures(degree=4).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=random.randint(0,100))
lm.fit(X_train,y_train)
print(lm.score(X_test,y_test))


