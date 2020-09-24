# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:16:18 2020

@author: Ray
@wechat: RayTing0305
@email: 1324789704@qq.com
"""
import numpy as np
import pandas as pd
import re
from scipy import stats

'''
Final Quiz
'''
a = np.arange(8)
b = a[4:6]
b[:] = 40
c = a[4] + a[6]

s = 'ABCAC'


s = 'ACAABAACAAABACDBADDDFSDDDFFSSSASDAFAAACBAAAFASD'

result = []
# compete the pattern below
pattern = '([A-Z]*?)(A{3})'
for item in re.finditer(pattern, s):
    # identify the group number below.
    result.append(item.group(1))
#print(result)

df = pd.Series(data=[4, 7, -5, 3], index=list('dbac'))

s1 = pd.Series(data=[20, 15, 18, 31], 
               index=['Mango', 'Strawberry',
                      'Blueberry','Vanilla'])
s2 = pd.Series(data=[20, 30, 15, 20, 20],
               index=['Strawberry',
                      'Vanilla',
                      'Banana',
                      'Mango',
                      'Plain'])
s3 = s1.add(s2)

S = pd.Series(np.arange(5), index=['a', 'b', 'c', 'd', 'e'])

df = pd.DataFrame(data={'a':[5, 5, 71, 67],
                        'b':[6, 82, 31, 37],
                        'c':[20, 28, 92, 49]},
                  index = ['R1', 'R2', 'R3', 'R4'])


f = lambda x: x.max() + x.min()
df_new = df.apply(f)

df = pd.DataFrame(data={'Item':['item_1', 'item_1', 'item_1'
                                ,'item_2','item_2', 'item_2'],
                        'Store':['A', 'B', 'C',
                                 'A', 'B', 'C'],
                        'Quantity sold':[10.0, 20.0, np.nan,
                                         5.0, 10.0, 15.0]},
                  index = np.arange(6))

'''
Assignment4
'''

def GetDF(df, cities, string):
    city = []
    for i in cities[string]:
        if i[0] != '—' and i[0] != '[':
            city.append(i)
    cities = cities[cities[string].isin(city)]
    need_drop = ['Atlantic Division', 'Metropolitan Division', 'Central Division', 'Pacific Division']
    flag = df['team'].isin(need_drop)
    df = df[~flag]
    df['W'] = df.W.astype('float')
    df['L'] = df.L.astype('float')
    df['WLR'] = df['W'] / (df['W']+df['L'])
    df = df.loc[:, ['team', 'WLR', 'W']]
    
    return df

def GetCity(cities, string):
    city = []
    for i in cities[string]:
        if i[0] != '—' and i[0] != '[':
            city.append(i)
    cities = cities[cities[string].isin(city)]
    cities.index = np.arange(cities.shape[0])
    patt='\\[.*\\]'
    for i in range(len(cities[string])):
        s=cities[string][i]
        try:
            a=re.findall(patt,s)[0]
            cities[string][i]=s.replace(a,'')
        except:
            continue
    return cities[['Metropolitan area', string, 'Population (2016 est.)[8]']]

def nhl_correlation(): 
    # YOUR CODE HERE
    nhl_df=pd.read_csv("assets/nhl.csv")
    cities=pd.read_html("assets/wikipedia_data.html")[1]
    cities=cities.iloc[:-1,[0,3,5,6,7,8]]
    nhl_df = GetDF(nhl_df, cities, 'NHL')
    cities = GetCity(cities, 'NHL')
    
    #raise NotImplementedError()
    
    population_by_region = [int(i) for i in cities['Population (2016 est.)[8]'].tolist()] 
    # pass in metropolitan area population from cities
    win_loss_by_region = []
    for i in range(len(cities)):
        item = cities.iloc[i]
        if item['Metropolitan area']=='New York City' or item['Metropolitan area']=='Los Angeles':
            continue
        nhl_name = item['NHL']
        for j in range(len(nhl_df)):
            searchFor = nhl_df.iloc[j]['team']
            if nhl_name in searchFor:
                win_loss_by_region.append(nhl_df.iloc[j]['WLR'])
                break
    win_loss_by_region.insert(0, 0.5182014205986808)
    win_loss_by_region.insert(1, 0.622894633764199)
    # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"
    
    return stats.pearsonr(population_by_region, win_loss_by_region)[0]


def nba_correlation():
    nba_df=pd.read_csv("assets/nba.csv")
    cities=pd.read_html("assets/wikipedia_data.html")[1]
    cities=cities.iloc[:-1,[0,3,5,6,7,8]]
    nba_df=nba_df[nba_df.year == 2018]

    nba_df = GetDF(nba_df, cities, 'NBA')
    #print(nba_df)
    cities = GetCity(cities, 'NBA')
    #print(cities)
    win_loss_by_region = []
    for i in range(len(cities)):
        item = cities.iloc[i]
        if item['Metropolitan area']=='New York City' or item['Metropolitan area']=='Los Angeles':
            continue
        nba_name = item['NBA']
        for j in range(len(nba_df)):
            searchFor = nba_df.iloc[j]['team']
            if nba_name in searchFor:
                win_loss_by_region.append(nba_df.iloc[j]['WLR'])
                break
    win_loss_by_region.insert(0, 0.4695121951219512)
    win_loss_by_region.insert(0, 0.3475609756097561)
    
    population_by_region = [int(i) for i in cities['Population (2016 est.)[8]'].tolist()] 

    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]


def mlb_correlation(): 
    mlb_df=pd.read_csv("assets/mlb.csv")
    cities=pd.read_html("assets/wikipedia_data.html")[1]
    cities=cities.iloc[:-1,[0,3,5,6,7,8]]
    mlb_df=mlb_df[mlb_df.year == 2018]

    mlb_df = GetDF(mlb_df, cities, 'MLB')
    cities = GetCity(cities, 'MLB')
    population_by_region = [int(i) for i in cities['Population (2016 est.)[8]'].tolist()] 
    win_loss_by_region = []
    for i in range(len(cities)):
        item = cities.iloc[i]
        if item['Metropolitan area']=='New York City' or item['Metropolitan area']=='Los Angeles' or item['Metropolitan area'] == 'San Francisco Bay Area' or item['Metropolitan area'] == 'Chicago':
            continue
        mlb_name = item['MLB']
        for j in range(len(mlb_df)):
            searchFor = mlb_df.iloc[j]['team']
            if mlb_name in searchFor:
                win_loss_by_region.append(mlb_df.iloc[j]['WLR'])
                break
    win_loss_by_region.insert(0, 0.5462962962962963)
    win_loss_by_region.insert(1, 0.5291221692039687)
    win_loss_by_region.insert(2, 0.5246913580246914)
    win_loss_by_region.insert(3, 0.48276906763614325)
    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]


def nfl_correlation(): 
    nfl_df=pd.read_csv("assets/nfl.csv")
    cities=pd.read_html("assets/wikipedia_data.html")[1]
    cities=cities.iloc[:-1,[0,3,5,6,7,8]]
    nfl_df=nfl_df[nfl_df.year == 2018]
    #nfl_df
    need_drop = ['AFC East', 'AFC North', 'AFC South', 'AFC West', 
                  'NFC East', 'NFC North', 'NFC South', 'NFC West']
    flag = nfl_df['team'].isin(need_drop)
    nfl_df = nfl_df[~flag]
    nfl_df['WLR'] = pd.to_numeric(nfl_df['W-L%'])
    nfl_df = nfl_df.loc[:, ['team', 'WLR', 'W']]
    nfl_df.team = nfl_df.team.str.strip('*+')
    cities = GetCity(cities, 'NFL')
    patt='\\[.*\\]'
    string = 'NFL'
    for i in range(len(cities[string])):
        s=cities[string][i]
        try:
            a=re.findall(patt,s)[0]
            cities[string][i]=s.replace(a,'')
        except:
            continue
    population_by_region = [int(i) for i in cities['Population (2016 est.)[8]'].tolist()] 
    win_loss_by_region = []
    for i in range(len(cities)):
        item = cities.iloc[i]
        if item['Metropolitan area']=='New York City' or item['Metropolitan area']=='Los Angeles' or item['Metropolitan area'] == 'San Francisco Bay Area' :
            continue
        nfl_name = item['NFL']
        for j in range(len(nfl_df)):
            searchFor = nfl_df.iloc[j]['team']
            if nfl_name in searchFor:
                win_loss_by_region.append(nfl_df.iloc[j]['WLR'])
                break
    win_loss_by_region.insert(0, 0.28125)
    win_loss_by_region.insert(1, 0.78125)
    win_loss_by_region.insert(2, 0.25)

    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]


