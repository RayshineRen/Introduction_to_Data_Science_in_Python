# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:28:11 2020

@author: Ray
@email: 1324789704@qq.com
@wechat: RayTing0305
"""

###chapter5

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)


### Series

obj = pd.Series([4, 7, -5, 3])
obj_array = obj.values
obj_range = obj.index

obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2_array = obj2.values
obj2_range = obj2.index

obj3 = obj2[['a','c','d']]
obj3_array = obj3.values
obj3_range = obj3.index

obj4 = obj2[obj2>0]
obj5 = obj2*2
obj6 = np.exp(obj2)

#print('b' in obj2)
#print('e' in obj2)


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj7 = pd.Series(sdata)

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj8 = pd.Series(sdata, index=states)

#print(pd.isnull(obj8))
#print(pd.notnull(obj8))

obj9 = obj7 + obj8

obj8.name = 'population'
obj8.index.name = 'state'



####DataFrame

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
print(frame.state)
#print(frame.head())
#print(frame.columns)

frame = pd.DataFrame(data, columns=['year', 'state', 'pop'])

frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four',
                             'five', 'six'])
fc1 = frame2['state']
fc2 = frame2.state
#print(fc1==fc2)
#print(id(fc1)==id(fc2))

fr1 = frame2.loc['two']
#print(fr1)

frame2['debt'] = np.arange(6.)
#print(frame2)

val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
#print(frame2)

frame2['eastern'] = frame2.state == 'Ohio'

del frame2['eastern']

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DataFrame(pop)

#print(frame3.T)

frame4 = pd.DataFrame(pop, index=[2001, 2002, 2003])

pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
frame5 = pd.DataFrame(pdata)

frame3.index.name='year'
frame3.columns.name = 'state'
#print(frame3.values)

### Index Objects
obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index

##index[1] = 'd' # TypeError

labels = pd.Index(np.arange(3))
dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])
frame6 = pd.Series(np.arange(4), index = dup_labels)
#print(frame6['foo'])


### Essential Functionality

obj  = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj4 = obj3.reindex(range(6), method='ffill')

frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=['a', 'c', 'd'],
                     columns=['Ohio', 'Texas', 'California'])
frame2 = frame.reindex(['a', 'b', 'c', 'd'])

states = ['Texas', 'Utah', 'California']
frame3 = frame.reindex(columns=states)

#fr = frame.loc[['a', 'c'], states]


## Dropping Entries from an Axis
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop(['c', 'd'])


obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj2 = obj[['b', 'a', 'd']]
obj3 = obj[[1, 3]]
obj4 = obj[obj<2]
obj5 = obj['b':'e']
obj['b':'c'] = 5

data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
#print(data)
#print(data[:2])
#print(data[data['three']>5])
#data[data<5]=0
#print(data)

loc = data.loc['Colorado', ['two', 'three']]

loc2 = data.iloc[2, [3, 0, 1]]
#print(loc2)
loc3 = data.iloc[2]
loc4 = data.iloc[[1, 2], [3, 0, 1]]
#print(loc4)
loc5 = data.iloc[:, :3][data.three > 5]
#print(loc5)
























