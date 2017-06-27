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
from sklearn.preprocessing import PolynomialFeatures, scale, normalize


import numpy as np
import statistics as stat
from mpl_toolkits.mplot3d import Axes3D
import math
import random

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

'''
input_file = '419-Park-Ave-S-2nd-Floor-NY-10016_1-year-till_jun-24-17' + '.csv'

data = pd.read_csv('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/' + input_file)
time_slots  = data['Time difference'] = data['Time difference'].apply(lambda x: x/60)

series = RevenueDist(time_slots,75)
deg = 3

Xp = series.data_frame['Time Slot Duration']
Yp = series.data_frame.cumsum()['Annual Revenue']

Xp, Yp = scale(Xp), scale (Yp)

X = np.transpose(np.matrix(Xp))
#Y = np.transpose(np.matrix(Yp))
Y = Yp

X_poly = PolynomialFeatures(degree=deg).fit_transform(X)

Lasso = linear_model.RidgeCV()

Lasso.fit(X_poly,Y)

print("The R2 score is : " + str(Lasso.score(X_poly,Y)))


Tp = np.linspace(min(Xp),max(Xp),Xp.size)
T = np.transpose(np.matrix(Tp))
T = PolynomialFeatures(degree=deg).fit_transform(T)

plt.plot(X,Y,'--r')
plt.plot(X,Lasso.predict(X_poly),'g')
plt.plot(Tp, Lasso.predict(T),'b')
plt.grid()
plt.show()

print(Lasso.coef_)
'''

input_file = 'Annual-report-data-Test' + '.csv'

data = pd.read_csv('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/' + input_file)

Midtown = data[data['Location'] == 'Midtown']

Downtown = data[data['Location']=='Downtown']

Midtown_s = data[data['Location'] == 'Midtown South']

Manhattan = data[data['Location'] == 'Manhattan Total']

Midtown_rent = Midtown.ix[:,'Overall asking rent (gross $ PSF)']
Downtown_rent = Downtown.ix[:,'Overall asking rent (gross $ PSF)']
Midtown_s_rent = Midtown_s.ix[:,'Overall asking rent (gross $ PSF)']
Manhattan_rent = Manhattan.ix[:,'Overall asking rent (gross $ PSF)']

Midtown_rent = np.array(Midtown_rent[::-1])
Downtown_rent = np.array(Downtown_rent[::-1])
Midtown_s_rent = np.array(Midtown_s_rent[::-1])
Manhattan_rent = np.array(Manhattan_rent[::-1])

d ={'Mid': Midtown_rent,
    'Down': Downtown_rent,
    'Mid_s': Midtown_s_rent,
    'Man': Manhattan_rent}

d = pd.DataFrame(d)
print (d.corr())

X = range(Midtown_rent.size)

'''
plt.plot(X,Midtown_rent,'b', label = 'Midtown')
plt.plot(X,Downtown_rent,'k', label = 'Downtown')
plt.plot(X, Midtown_s_rent,'r', label = 'Midtown South')
plt.plot(X, Manhattan_rent,'g', label = 'Manhattan')
plt.title('Rent vs time')
plt.ylabel('Asking rent ($ PSF)')
plt.xlabel('Time (quarters)')
plt.legend()
plt.grid()
plt.show()
'''



TT = Tickers()
VNO = TT.ticker_data['VNO']
VNO_quarter = []

slicer = int(len(VNO)/14)

for ii in range(14):
    VNO_quarter.append(np.mean(VNO[ii*slicer:(ii+1)*slicer]))

VNO_quarter = np.array(VNO_quarter)

Manhattan_rent_2 = normalize(Manhattan_rent)
VNO_quarter_2 = normalize(VNO_quarter)

print(Manhattan_rent_2,VNO_quarter_2)    


dd = {'Man': Manhattan_rent_2[0],
      'VNO' :VNO_quarter_2[0]}

Manhattan_rent_3 = normalize(Manhattan_rent).T
VNO_quarter_3 = normalize(VNO_quarter).T 

plt.plot(X,Manhattan_rent_3,'r', label = 'Manhattan')
plt.plot(X,VNO_quarter_3, 'b', label = 'VNO')
plt.legend()
plt.grid()
plt.show()

dd = pd.DataFrame(dd)
print (dd.head())
print(dd.corr())

