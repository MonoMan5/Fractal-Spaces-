# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:31:17 2017

@author: Amar Sehic
"""

import requests 
import pandas as pd

#Open List of NYC BBLS for OFFICES

file_init = pd.read_csv('C:/Users/Amar Sehic/Documents/Rubik/Python/Python Web Scrapping/DOF Map Scraper/BBL Offices.csv')


# Enter Session Parameters

parameters = {
        'f':'json',        
        'returnGeometry':'false',
        'spatialRel':'esriSpatialRelIntersects',
        'outFields':'*',
        'outSR': 102100,
        'resultRecordCount':3
}

#Run session and request data for each BBL

d1={}
d2={}

print('Running Session...')

with requests.Session() as s:
    
    d1_array = []
    d2_array = []
    s.params = parameters
    
    for ii in range(len(file_init)):
    
        boro = str(file_init['BORO'][ii])
        block = str(file_init['BLOCK'][ii])
        lot = str(file_init['LOT'][ii])
        
        bbl_sp = '\''+boro+'/'+block+'/'+lot+'\''
        bl_sp = '\''+block+'/'+lot+'\''
    
        r1 = s.get('http://services3.arcgis.com/aD88pT4hjL80xq0F/arcgis/rest/services/tc4_MH_cdsuffix/FeatureServer/0/query', params = {'where':'bbl_sp = '+bbl_sp+' or bl_sp = '+bl_sp}, timeout=5)
        r2 = s.get('http://services3.arcgis.com/aD88pT4hjL80xq0F/arcgis/rest/services/TC4_MH_noncd//FeatureServer/0/query', params = {'where':'bbl_sp = '+bbl_sp+' or bl_sp = '+bl_sp}, timeout = 5)
        
        
    
    #Check if JSON exists, if not skip
    
        if len(r1.json()['features'])== 0:
            r = r2
            json_type = 2
        elif len(r2.json()['features'])== 0:
            print('Nothing found!')
            json_type = 0
        else:
            r= r1
            json_type = 1
    
    
    
    #Check if there are multiple units for that BBL and split accordingly
    #Extract data for each property
    
    
        if json_type == 1:
           
            json_response = r.json()['features']
            
            for prop in range(len(json_response)):
                prop_data = json_response[prop]['attributes']
                d1_array.append(prop_data)    
                
        elif json_type == 2:
           
            json_response = r.json()['features']
            
            for prop in range(len(json_response)):
                prop_data = json_response[prop]['attributes']            
                d2_array.append(prop_data)
                
    if len(d1_array) !=0:
        for k in d1_array[0]:
            d1[k] = tuple(d_i[k] for d_i in d1_array)
    
    for k in d2_array[0]:
        d2[k] = tuple(d_i[k] for d_i in d2_array)

print('Closing Session, creating dictionaries...')
              
#Save data in a dataframe

d1 = pd.DataFrame.from_dict(d1)
d2 = pd.DataFrame.from_dict(d2)
    
#Create CSV to save data in

print('Saving to CSV...')

d1.to_csv('C:/Users/Amar Sehic/Documents/Rubik/Python/Python Web Scrapping/DOF Map Scraper/Offices Scraped CONDO.csv')
d2.to_csv('C:/Users/Amar Sehic/Documents/Rubik/Python/Python Web Scrapping/DOF Map Scraper/Offices Scraped NON-CONDO.csv')

print('DONE!')

