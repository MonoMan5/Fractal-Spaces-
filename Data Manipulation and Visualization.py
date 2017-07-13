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
from statsmodels.nonparametric.kde import KDEUnivariate

import numpy as np
import math
import random
from statistics import mode, StatisticsError
from scipy.stats import pearsonr as PRS

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

class OfficeHist(object):
    
    '''
    Builds an office histogram object, with the distribution of occupancy over
    days for a particular office.
    '''
    
    def __init__(self,frame):
        
        self.name = frame[1]
        
        total_days = len(frame[0]['Timeline'])
        a = frame[0][frame[0]['Occupancy'] == True]
        
        if total_days != 0:
            probs = []
            for of in set(frame[0]['Name']):
                temp = frame[0][frame[0]['Name'] == of]
                tot = len(temp[temp['Occupancy'] == False]['Timeline']) + len(temp[temp['Occupancy'] == True]['Timeline'])
                occ = len(a[a['Name']==of]['Timeline'])
                probs.append(occ/tot)
                
            daily_prob = np.mean(probs)
            self.daily_prob = daily_prob
        else:
            daily_prob = 'None'
            self.daily_prob = daily_prob
        
        a = a.sort_values(['Day'])
        
        days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        days_dict = []
        
        for day in days:
            t = a[a['Day']==day]
            t = t['Time Used']
            t = pd.to_numeric(t, errors='coerce')
            days_dict.append(t.sum())
            
        
        self.time_dict = days_dict
        
        dddd = days_dict
        values = dddd
        maximum = max(values)
        max_day = values.index(maximum)+1
        
        self.profitable_day = max_day
        self.hours = values
        
        
        b = list(a['Day'])
        ccc = list(map(lambda x: day_number(x),b))
        
        if len(ccc) == 0:
            print(frame[1] + ' is Empty!')
            self.empty = True
            pass
        else:
            self.empty = False
        
        mean = np.mean(ccc)
        
        try:
            m = mode(b)
        except StatisticsError:
            m = None
        
        self.days_numerical = ccc
        self.days = b
        self.mean = mean
        self.mode = m
        
        
        
    def load_offices(self):
        print('Error: Already an Office Object!')
    
    def plot(self):
        
        if self.empty == True:
            
            print('No booked days!')
        
        else:
            
            plt.hist(self.days_numerical, bins = [1,2,3,4,5,6,7,8],normed = True,align = 'left')
            plt.title(self.name[:30])
            plt.axvline(self.mean, color ='r')
            plt.xlabel('Day of the week')
            plt.ylabel('Frequency (days)')
            plt.grid()
            plt.savefig('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/Graphs/Detailed Frequency per Building per Day/'+k.name + ' Frequency')
            plt.show()
            
            print('The sample mean is ' +str(self.mean))
            print('The Daily Probability is: ' + str(self.daily_prob))
            if self.mode == None :
                print('No unique mode!')
            else:
                print('The sample mode is ' + str(self.mode))
    
    def duration_plot(self):
        
        if self.empty == True:
            
            print('No booked days!')
        
        else:
        
            ind = range(1,8)
            max_day = self.profitable_day 
            
            plt.bar(ind,self.hours)
            plt.title(self.name[:30])
            plt.xlabel('Day of the week')
            plt.ylabel('Hours Booked')
            plt.grid()
            plt.show()
            
            print('The mode is: ' + str(max_day))


class BuildingHist(OfficeHist):
    
    '''
    Extends the Office hist class, this is a Building object containing 
    multiple offices inside.
    '''
    
    def __init__(self, frame):
        
        offices = []
        
        raw = frame[0][frame[0]['Occupancy'] == True]
        
        for office in set(raw['Name']):
            off = raw[raw['Name']==office]
            offices.append((off,office))
        
        self.offices = offices
        self.offices_loaded = False
        
        super().__init__(frame)
        
    
    def load_offices(self):
        
        if self.offices_loaded == False and self.empty == False:
        
            office_objects = []
            offices = self.offices
            
            for of in offices:
                d = OfficeHist(of)
                office_objects.append(d)
            
            self.offices = office_objects
            self.offices_loaded = True
            
        elif self.offices_loaded == True:
            print('Error: Offices are already loaded!')
        else:
            print('Building is not occupied!')
    
    def plot(self):
        
        if self.empty == True:
            
            print('No booked days!')
        
        elif self.offices_loaded == True and len(self.offices)>1:
            
            fig = plt.figure(figsize=(6,6))
            fig.suptitle('Building Data - Distribution of Days', fontsize=15)
            cols = math.ceil((len(self.offices)/3))+1
            ax1 = plt.subplot2grid((3,cols),(0,0),rowspan = 3)
            ax1.hist(self.days_numerical, bins = [1,2,3,4,5,6,7,8],normed = True,align = 'left')
            plt.axvline(self.mean, color ='r')
            plt.xlabel('Day of the week')
            plt.ylabel('Frequency (days)')
            plt.title(self.name[:30])
            plt.grid()
            axarray = []
            
            cc = 1
            rr = 0
            
            for ii in range(len(self.offices)):
                
                axarray.append(plt.subplot2grid((3,cols),(rr,cc)))
              
                
                axarray[ii].hist(self.offices[ii].days_numerical,bins = [1,2,3,4,5,6,7,8],normed = True,align = 'left')
                plt.axvline(self.offices[ii].mean, color ='r')
                plt.title(self.offices[ii].name)
                plt.grid()
                if rr == 2:
                     cc+=1
                     rr = 0
                else:
                    rr+=1
          
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.savefig('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/Graphs/Detailed Frequency per Building per Day/'+k.name + ' Frequency Detailed')
            plt.show()
            
            print('The sample mean is ' +str(self.mean))
            print('The Daily Probability is: ' + str(self.daily_prob))
            if self.mode == None :
                print('No unique mode!')
            else:
                print('The sample mode is ' + str(self.mode))
        
        else:
            
            super().plot()
            

                
                
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

def day_number(var):
    '''
    Convert day name into number of day in week and vice-versa.
    '''
    
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    
    p = days.index(var)
    
    return p+1

    
        
def building_const(directory):
    
    '''
    Builds a bunch of building histogram objects from specified directory and
    returns them in an array.
    '''
    
    a = os.listdir(directory)


    names = []
    zips = []
    
    
    DF = []
    BHists = []
    
    
    for file in a:
        res = ltw.get_address(file)
        zips.append(int(res[1]))
        names.append(res[0])
    
        DF.append((pd.read_csv(directory + file), res[0]))
        
    for df in DF:
        BHists.append(BuildingHist(df))
        
    return BHists

def aggregate(show_time = True):
    
    ll =[]

    modes = []
    
    A = building_const(write_dir)
    
    for k in A:
        
        modes = modes + k.days_numerical
    
        ll.append(k.hours)
        
    agg = [sum(x) for x in zip(*ll)]   
    
    ind = [1,2,3,4,5,6,7]
    
    if show_time == True:
        plt.bar(ind, agg, color = 'r', label = 'Hours')
    plt.hist(modes, bins = [1,2,3,4,5,6,7,8], label = 'Frequency', align = 'left')
    plt.legend()
    plt.title('Aggregate over all buildings')
    plt.xlabel('Day of the week')
    plt.ylabel('Frequency/Hours Booked')
    plt.grid()
    plt.savefig('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/Graphs/Aggregated Data/'+ 'Building Aggregates')
    plt.show()
    
    test = np.histogram(modes, bins = [1,2,3,4,5,6,7,8])[0]
    
    print('The Pearson Correlation Coefficient is : ' + str(PRS(agg,test)[0]))


###############################################################################
#   MAIN PROGRAM BODY
###############################################################################

A = building_const(write_dir)
    

'''
X = np.histogram(k.days_numerical, bins = [1,2,3,4,5,6,7,8], density = True)[0]

Bins = range(1,len(X)+1)

X = np.array(list((zip(Bins,X))))
X.reshape(-1,1)

Test = np.histogram(j.days_numerical, bins = [1,2,3,4,5,6,7,8], density = True)

X_plot = np.array(list((zip(Bins,Test[0]))))
X_plot.reshape(-1,1)

print(X)
print(X_plot)

X = np.array(k.days_numerical)

X_plot = np.random.randint(0,10,50)
X_plot.sort()

print(X)
print(X_plot)

kde = KDEUnivariate(X.astype(float))
kde.fit()
plt.plot(X_plot, list(map(lambda x: kde.evaluate(x),X_plot)), color = 'k')
k.plot()
'''



'''
    
    modes = k.days_numerical
    agg = k.hours
    ind = [1,2,3,4,5,6,7]
    
    f = plt.figure(figsize=(5,5))
    plt.bar(ind, agg, color = 'r', label = 'Hours')
    plt.hist(modes, bins = [1,2,3,4,5,6,7,8], label = 'Frequency', align = 'left')
    plt.legend()
    plt.title(k.name)
    plt.xlabel('Day of the week')
    plt.ylabel('Frequency/Hours Booked')
    plt.grid()
    plt.savefig('C:/Users/Amar Sehic/Documents/Fractal/Python Data Analysis/Graphs/'+k.name + ' Freq vs Hours')
    plt.show()
    
 '''   

        

'''
#Amenities comparison

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
