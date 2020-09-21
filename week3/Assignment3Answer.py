# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 00:24:43 2020

@author: Ray
@email: 1324789704@qq.com
@wechat: RayTing0305
"""

import pandas as pd
import numpy as np
import re


'''
Quiz
'''
index1 = ['James', 'Mike', 'Sally']
col1 = ['Business', 'Law', 'Engineering']
student_df = pd.DataFrame(col1, index1)
student_df.index.name='Name'
student_df.columns = ['School']

index2 = ['Kelly', 'Sally', 'James']
col2 = ['Director of HR', 'Course liasion', 'Grader']
staff_df = pd.DataFrame(col2, index2)
staff_df.index.name = 'Name'
staff_df.columns = ['Role']


df = pd.DataFrame({'P2010':[100.1, 200.1], 
                  'P2011':[100.1, 200.1], 
                  'P2012':[100.1, 200.1], 
                  'P2013':[100.1, 200.1], 
                  'P2014':[100.1, 200.1], 
                  'P2015':[100.1, 200.1]})
frames = ['P2010', 'P2011', 'P2012', 'P2013','P2014', 'P2015']
df['AVG'] = df[frames].apply(lambda z: np.mean(z), axis=1)
result_df = df.drop(frames,axis=1)

df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'], index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'], columns = ['Grades'])
my_categories= pd.CategoricalDtype(categories=['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'], ordered=True)
grades = df['Grades'].astype(my_categories)
result = grades[(grades>'B') & (grades<'A')]


(pd.Timestamp('11/29/2019') + pd.offsets.MonthEnd()).weekday()

pd.Period('01/12/2019', 'M') + 5

'''
Assignment
'''


def answer_one():
    # YOUR CODE HERE
    x = pd.ExcelFile('assets/Energy Indicators.xls')
    energy = x.parse(skiprows=17,skip_footer=(38))
    energy = energy[['Unnamed: 1','Petajoules','Gigajoules','%']]
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] =  energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
    energy['Energy Supply'] = energy['Energy Supply']*1000000
    energy['Country'] = energy['Country'].replace({
        "Republic of Korea": "South Korea", 
        "United States of America": "United States",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "China, Hong Kong Special Administrative Region": "Hong Kong",
        'Iran (Islamic Republic of)':'Iran',
        'Bolivia (Plurinational State of)':'Bolivia'
        })
    energy['Country'] = energy['Country'].str.replace(r" \(.*\)","")

    GDP = pd.read_csv('assets/world_bank.csv',skiprows=4)
    GDP['Country Name'] = GDP['Country Name'].replace({
        "Korea, Rep.": "South Korea", 
        "Iran, Islamic Rep.": "Iran",
        "Hong Kong SAR, China": "Hong Kong"
        })
    GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']

    ScimEn = pd.read_excel(io='assets/scimagojr-3.xlsx')
    ScimEn_m = ScimEn[:15]

    df1 = pd.merge(ScimEn_m, energy, how='inner', left_on='Country', right_on='Country')
    df2 = pd.merge(df1, GDP, how='inner', left_on='Country', right_on='Country')

    res = df2.set_index('Country')
    return res
    raise NotImplementedError()

def answer_two():
    # YOUR CODE HERE
    x = pd.ExcelFile('assets/Energy Indicators.xls')
    energy = x.parse(skiprows=17,skip_footer=(38))
    energy = energy[['Unnamed: 1','Petajoules','Gigajoules','%']]
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] =  energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
    energy['Energy Supply'] = energy['Energy Supply']*1000000
    energy['Country'] = energy['Country'].replace({
        "Republic of Korea": "South Korea", 
        "United States of America": "United States",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "China, Hong Kong Special Administrative Region": "Hong Kong",
        'Iran (Islamic Republic of)':'Iran',
        'Bolivia (Plurinational State of)':'Bolivia'
        })
    energy['Country'] = energy['Country'].str.replace(r" \(.*\)","")

    GDP = pd.read_csv('assets/world_bank.csv',skiprows=4)
    GDP['Country Name'] = GDP['Country Name'].replace({
        "Korea, Rep.": "South Korea", 
        "Iran, Islamic Rep.": "Iran",
        "Hong Kong SAR, China": "Hong Kong"
        })
    GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']

    ScimEn = pd.read_excel(io='assets/scimagojr-3.xlsx')
    ScimEn_m = ScimEn[:15]

    dfSe = pd.merge(ScimEn_m, energy, how='inner', left_on='Country', right_on='Country')
    dfGe = pd.merge(energy, GDP, how='inner', left_on='Country', right_on='Country')
    dfGS = pd.merge(GDP, ScimEn_m, how='inner', left_on='Country', right_on='Country')

    dfSe_s = dfSe.shape[0]
    dfGe_s = dfGe.shape[0]
    dfGS_s = dfGS.shape[0]

    res = dfSe_s + dfGe_s + dfGS_s - 4*15
    return res
    raise NotImplementedError()

def answer_three():
    # YOUR CODE HERE
    Top15entries = answer_one()
    Top15entries['avgGDP'] = Top15entries[['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']].mean(axis=1)
    return Top15entries.avgGDP.sort_values(ascending=False)
    raise NotImplementedError()
    
def answer_four():
    # YOUR CODE HERE
    target_country = answer_three().index.tolist()[5]
    data = answer_one().loc[target_country]
    return data.loc['2015'] - data.loc['2006']
    raise NotImplementedError()
answer_four()

def answer_five():
    # YOUR CODE HERE
    Top15entries = answer_one()
    ans = Top15entries['Energy Supply per Capita'].mean()
    return ans
    raise NotImplementedError()
answer_five()

def answer_six():
    # YOUR CODE HERE
    Top15entries = answer_one()
    ans = Top15entries.sort_values(by='% Renewable', ascending=False).iloc[0]
    return ans.name, ans['% Renewable']
    raise NotImplementedError()

def answer_seven():
    # YOUR CODE HERE
    Top15entries = answer_one()
    Top15entries['Citation_ratio'] = Top15entries['Self-citations'] / Top15entries['Citations']
    ans = Top15entries.sort_values(by='Citation_ratio', ascending=False).iloc[0]
    return ans.name, ans['Citation_ratio']
    raise NotImplementedError()

def answer_eight():
    # YOUR CODE HERE
    Top15entries = answer_one()
    Top15entries['EstiPop'] = Top15entries['Energy Supply'] / Top15entries['Energy Supply per Capita']
    ans = Top15entries.sort_values(by='EstiPop', ascending=False).iloc[2]
    return ans.name
    raise NotImplementedError()

def answer_nine():
    # YOUR CODE HERE
    Top15entries = answer_one()
    Top15entries['EstiPop'] = Top15entries['Energy Supply'] / Top15entries['Energy Supply per Capita']
    Top15entries['cdpc'] = Top15entries['Citable documents'] / Top15entries['EstiPop']
    ans = Top15entries['cdpc'].corr(Top15entries['Energy Supply per Capita'])
    return ans
    raise NotImplementedError()
    
def answer_ten():
    # YOUR CODE HERE
    Top15entriese = answer_one()
    median_ = Top15entriese['% Renewable'].median()
    Top15entriese['HighRenew'] = Top15entriese['% Renewable'].apply(lambda x:1 if x>=median_ else 0)
    return Top15entriese['HighRenew']
    raise NotImplementedError()

def answer_eleven():
    # YOUR CODE HERE
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    Top15entries = answer_one()
    Top15entries['EstiPop'] = (Top15entries['Energy Supply'] / Top15entries['Energy Supply per Capita']).astype(float)
    #Top15entries.reset_index()
    conti = []
    for country in Top15entries.index:
        conti.append(ContinentDict[country])
    Top15entries['Continent'] = conti
    gp = Top15entries.groupby(by='Continent').EstiPop
    return gp.agg({'size': np.size, 'sum': np.sum, 'mean': np.mean,'std': np.std})
    raise NotImplementedError()

def answer_twelve():
    # YOUR CODE HERE
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    Top15entries = answer_one()
    Top15entries['EstiPop'] = (Top15entries['Energy Supply'] / Top15entries['Energy Supply per Capita']).astype(float)
    #Top15entries.reset_index()
    conti = []
    for country in Top15entries.index:
        conti.append(ContinentDict[country])
    Top15entries['Continent'] = conti
    Top15entries['bins'] = pd.cut(Top15entries['% Renewable'],5)
    return Top15entries.groupby(['Continent', 'bins']).size()
    raise NotImplementedError()

def answer_thirteen():
    # YOUR CODE HERE
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    Top15entries = answer_one()
    Top15entries['PopEst'] = (Top15entries['Energy Supply'] / Top15entries['Energy Supply per Capita']).astype(float)
    map_str = []
    for num in Top15entries['PopEst']:
        map_str.append(locale.format('%.2f',num,grouping=True))
    Top15entries['PopEst_str'] = map_str
    return Top15entries['PopEst_str']
    '''
    Top15entries = answer_one()
    Top15entries['PopEst'] = (Top15entries['Energy Supply'] / Top15entries['Energy Supply per Capita']).astype(float)
    return Top15entries['PopEst'].apply(lambda x: '{0:,}'.format(x))
    '''
    raise NotImplementedError()

