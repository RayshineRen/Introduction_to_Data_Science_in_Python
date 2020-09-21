# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:01:20 2020

@author: Ray
@email: 1324789704@qq.com
@wechat: RayTing0305
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import string
import scipy.stats as stats


'''
relax a little bit
test a quiz
'''
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj1 = pd.Series(sdata)
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj2 = pd.Series(sdata, index = states)
obj3 = pd.isnull(obj2)
x=obj2['California']
#print(obj2['California']!=x)
#print(obj2['California']==None)
#print(math.isnan(obj2['California']))
#print(pd.isna(obj2['California']))
d = {
    '1': 'Alice',
    '2': 'Bob',
    '3': 'Rita',
    '4': 'Molly',
    '5': 'Ryan'
}
S = pd.Series(d)
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])

data1 = data.rename(mapper=lambda x:x.upper(), axis=1)
#data2 = data.rename(mapper=lambda x:x.upper(), axis=1, inplace=True)
#data3 = data.rename(mapper=lambda x:x.upper(), axis='column')

data = {'gre score':[337, 324, 316, 322, 314], 
        'toefl score':[118, 107, 104, 110, 103]}
df = pd.DataFrame(data, columns=['gre score', 'toefl score'],
                  index=np.arange(1, 6, 1))
df1 = df.where(df['toefl score']>105).dropna()
df2 = df[df['toefl score']>105]
df3 = df.where(df['toefl score']>105)

#s1 = pd.Series({1: 'Alice', 2: 'Jack', 3: 'Molly'})
#s2 = pd.Series({'Alice': 1, 'Jack': 2, 'Molly': 3})
df1 = df[df['toefl score'].gt(105)&df['toefl score'].lt(115)]
df2 = df[(df['toefl score'].isin(range(106,115)))]
df3 = (df['toefl score']>105)&(df['toefl score']<115)

data = {'Name':['Alice', 'Jack'],
        'Age':[20, 22],
        'Gender':['F', 'M']}
index1 = ['Mathematics', 'Sociology']
df = pd.DataFrame(data, index=index1)
seri = df.T['Mathematics']

for col in df.columns:
    series = df[col]
#    print(series)

'''
pandas中文文档
'''

s = pd.Series([1, 3, 5, np.nan, 6, 8])

dates = pd.date_range('20130111', periods=6)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),                    
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

#print(df2.dtypes)

#print(df2.head(3))
#print(df2.tail(3))

#print(df2.index)
#print(df2.columns)

arr = df.to_numpy()
arr2 = df2.to_numpy()

df3 = df.describe()
df4 = df.T

df5 = df.sort_index(axis=1, ascending=False)
df6 = df5.sort_index(axis=0, ascending=False)

df7 = df.sort_values(by='B')
df8 = df.sort_values(by='C', ascending=False)

'''
切片索引
'''
col = df['A']
col = df.A
row012 = df[0:3]
row1 = df['20130112':'20130114']
Upper_letter = string.ascii_letters[26:]
up_letter = []
for i in range(26):
    if Upper_letter[i] in df.columns:
        up_letter.append(Upper_letter[i])
#print(up_letter)
col_full = df[up_letter]
'''
标签搜索
'''
row = df.loc[dates[0]]
row_col2 = df.loc[:, ['A', 'B']]
#df[:] == df.loc[:]
row_full = df.loc[:, up_letter]
row_full1 = df.loc[dates]

row_col = df.loc['20130112':'20130114', ['A', 'B']]

jiangwei = df.loc['20130112', ['A', 'B']]

scaler = df.loc[dates[0], 'A']

'''
位置搜索
'''
row3 = df.iloc[3]
row12 = df.iloc[:2]
row45col01 = df.iloc[3:5, 0:2]
row235col11 = df.iloc[[1, 2, 4], [2, 2]]
scaler22 = df.iloc[1, 1]
scaler22 = df.iat[1, 1]

'''
布尔索引
'''

Booldf1 = df[df.B>0]
Booldf2 = df[df > 0]
#print(Booldf2.dropna())
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df3 = df2[df2['E'].isin(['two', 'four'])]
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))

df.at[dates[0], 'A'] = 0
df.iat[0, 1] = 0
df.loc[:, 'D'] = np.array([5] * len(df))

df4 = df.copy()
df4[df4 > 0] = -df4

df5 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df5.loc[dates[0]:dates[1], 'E'] = 1

'''
填充/删除nan
'''
df6 = df5.dropna(how='any')
df7 = df5.fillna(value=5)
Booldf3 = pd.isna(df5)

'''
统计
'''

mean1 = df.mean()
mean2 = df5.mean()
mean3 = df5.fillna(value=0).mean()
#print(mean2==mean3)
mean11 = df.mean(axis = 1)
mean22 = df5.mean(1)


s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
apply_df1 = df.apply(np.cumsum)

apply_df2 = df.apply(lambda x:x.max() - x.min())

s2 = pd.Series(np.random.randint(0, 7, size=10))
s3 = s2.value_counts()


s4 = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s5 = s4.str.lower()


'''
Merge
'''

df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
df2 = pd.concat(pieces)

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
merged = pd.merge(left, right, on='key')

left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
merged = pd.merge(left, right, on='key')

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
s = df.iloc[3]
df = df.append(s, ignore_index=True)












'''
Just Test
'''


# Part 1
df = pd.read_csv('./assets/olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(')
#print(df.index)
df.index = names_ids.str[0]
df['ID'] = names_ids.str[1].str[:3]
df = df.drop('Totals')
#print(df.head())

#Question0
def answer_zero():
    # This function returns the row for Afghanistan, which is a Series object. The assignment
    # question description will tell you the general format the autograder is expecting
    return df.iloc[0]

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
#print(answer_zero())

#Question 1
'''
Which country has won the most gold medals in summer games?

This function should return a single string value.
'''
def answer_one():
    x = max(df['Gold'])     # int
    ans = df[df['Gold']==x].index[0]
    ans = df[df['Gold']==x].index.to_list()[0]
    return ans

'''
Question 2
Which country had the biggest difference between their summer and winter gold medal counts?

This function should return a single string value.
'''
def answer_two():
    x = max(df['Gold'] - df['Gold.1'])
    ans = df[(df['Gold']-df['Gold.1'])==x].index.to_list()[0]
    return ans

'''
Question 3
Which country has the biggest difference between their summer and winter gold medal counts relative to their total gold medal count? Only include countries that have won at least 1 gold in both summer and winter.

This function should return a single string value.
'''

def answer_three():
    df_gold = df[(df['Gold']>0)&(df['Gold.1']>0)]
    df_max_diff = (abs(df_gold['Gold']-df_gold['Gold.1'])/df_gold['Gold.2'])
    return df_max_diff
    #return df_max_diff.idxmax()
a = answer_three()

max_ = 0
for i in range(len(a)):
    if a[i]>max_:
        max_ = a[i]
        name = a.index[i]
        
#print(name)

'''
Question 4
Write a function to update the dataframe to include a new column called "Points" which is a weighted value where each gold medal counts for 3 points, silver medals for 2 points, and bronze mdeals for 1 point. The function should return only the column (a Series object) which you created.

This function should return a Series named Points of length 146
'''

def answer_four():
    Points = 3*df['Gold.2']+2*df['Silver.2']+df['Bronze.2']
    return Points
#print(answer_four())


'''
Part 2
For the next set of questions, we will be using census data from the United States Census Bureau. Counties are political and geographic subdivisions of states in the United States. This dataset contains population data for counties and states in the US from 2010 to 2015. See this document for a description of the variable names.

The census dataset (census.csv) should be loaded as census_df. Answer questions using this as appropriate.

Question 5
Which state has the most counties in it? (hint: consider the sumlevel key carefully! You'll need this for future questions too...)

This function should return a single string value.
'''

census_df = pd.read_csv('./assets/census.csv')

def answer_five():
    counties_df = census_df[census_df['SUMLEV']==50]
    x = counties_df.groupby('STNAME').count()['CTYNAME']
    ans = x.idxmax()
    return ans

'''
Question 6
Only looking at the three most populous counties for each state, what are the three most populous states (in order of highest population to lowest population)?

This function should return a list of string values.
'''

def answer_six():
    counties_df = census_df[census_df['SUMLEV'] == 50]
    top_counties_df = counties_df.sort_values(by=['STNAME','CENSUS2010POP'],ascending=False).groupby('STNAME').head(3)
    ans = top_counties_df.groupby('STNAME').sum().sort_values(by='CENSUS2010POP').head(3).index.tolist()
    return ans    

'''
Question 7¶
Which county has had the largest change in population within the five year period (hint: population values are stored in columns POPESTIMATE2010 through POPESTIMATE2015, you need to consider all five columns)?

This function should return a single string value.
'''
def answer_seven():
    counties_df = census_df[census_df['SUMLEV'] == 50]
    counties_df['pop_change'] = abs(counties_df['POPESTIMATE2015'] - counties_df['POPESTIMATE2014'])+abs(counties_df['POPESTIMATE2014'] - counties_df['POPESTIMATE2013'])+abs(counties_df['POPESTIMATE2013'] - counties_df['POPESTIMATE2012'])+abs(counties_df['POPESTIMATE2012'] - counties_df['POPESTIMATE2011'])+abs(counties_df['POPESTIMATE2011'] - counties_df['POPESTIMATE2010'])
    a = max(counties_df['pop_change'])
    ans = counties_df['CTYNAME'][counties_df['pop_change']==a].tolist()
    return ans[0]


#print(answer_seven())

'''
Question 8
In this datafile, the United States is broken up into four regions using the "REGION" column.

Create a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington', and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.

This function should return a 5x2 DataFrame with the columns = ['STNAME', 'CTYNAME'] and the same index ID as the census_df (sorted ascending by index).
'''
def answer_eight():
    '''
    counties_df = census_df[census_df['SUMLEV'] == 50]
    ans = counties_df[((counties_df['REGION']==1)|(counties_df['REGION']==2))&(counties_df['CTYNAME']=='Washington County')&(counties_df['POPESTIMATE2015']>counties_df['POPESTIMATE2014'])][['STNAME','CTYNAME']]
    return ans
    '''
    counties_df = census_df[census_df['SUMLEV'] == 50]
    df = counties_df[(counties_df['REGION']==1) | (counties_df['REGION']==2)]
    df = df[(df['POPESTIMATE2015']>df['POPESTIMATE2014'])]
    grouped=df.groupby(by='CTYNAME')
    for i in grouped:
        name = i[0].split(' ')[0]
        if name=='Washington':
            dff = i[1]
    dff = dff[['STNAME', 'CTYNAME']]
    return dff

answer_eight()


'''
Question 1
Write a function called proportion_of_education which returns the proportion of children in the dataset who had a mother with the education levels equal to less than high school (<12), high school (12), more than high school but not a college graduate (>12) and college degree.

This function should return a dictionary in the form of (use the correct numbers, do not round numbers):

    {"less than high school":0.2,
    "high school":0.4,
    "more than high school but not college":0.2,
    "college":0.2}
'''
df = pd.read_csv("./assets/NISPUF17.csv")
def proportion_of_education():
    # your code goes here
    # YOUR CODE HERE
    df_edu = df.EDUC1
    edu_list = [1, 2, 3, 4]
    zero_df = pd.DataFrame(np.zeros((df_edu.shape[0], len(edu_list))), columns=edu_list)
    for edu in edu_list:
        zero_df[edu][df_edu==edu]=1
    #zero_df
    sum_ret = zero_df.sum(axis=0)
    name_l = ["less than high school", "high school", "more than high school but not college", "college"]
    rat = sum_ret.values/sum(sum_ret.values)
    dic = dict()
    for i in range(4):
        dic[name_l[i]] = rat[i]
    return dic
    raise NotImplementedError()
assert type(proportion_of_education())==type({}), "You must return a dictionary."
assert len(proportion_of_education()) == 4, "You have not returned a dictionary with four items in it."
assert "less than high school" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."
assert "high school" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."
assert "more than high school but not college" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."
assert "college" in proportion_of_education().keys(), "You have not returned a dictionary with the correct"

'''
Question 2
Let's explore the relationship between being fed breastmilk as a child and getting a seasonal influenza vaccine from a healthcare provider. Return a tuple of the average number of influenza vaccines for those children we know received breastmilk as a child and those who know did not.

This function should return a tuple in the form (use the correct numbers:

(2.5, 0.1)
'''
def average_influenza_doses():
    # YOUR CODE HERE
    #是否喂养母乳
    fed_breastmilk = list(df.groupby(by='CBF_01'))
    be_fed_breastmilk = fed_breastmilk[0][1]
    not_fed_breastmilk = fed_breastmilk[1][1]
    #喂养母乳的influenza数目
    be_fed_breastmilk_influenza = be_fed_breastmilk.P_NUMFLU
    num_be_fed_breastmilk_influenza = be_fed_breastmilk_influenza.dropna().mean()
    #未喂养母乳的influenza数目
    not_be_fed_breastmilk_influenza = not_fed_breastmilk.P_NUMFLU
    num_not_be_fed_breastmilk_influenza = not_be_fed_breastmilk_influenza.dropna().mean()
    return num_be_fed_breastmilk_influenza, num_not_be_fed_breastmilk_influenza
    raise NotImplementedError()
assert len(average_influenza_doses())==2, "Return two values in a tuple, the first for yes and the second for no."


'''
Question 3
It would be interesting to see if there is any evidence of a link between vaccine effectiveness and sex of the child. Calculate the ratio of the number of children who contracted chickenpox but were vaccinated against it (at least one varicella dose) versus those who were vaccinated but did not contract chicken pox. Return results by sex.

This function should return a dictionary in the form of (use the correct numbers):

    {"male":0.2,
    "female":0.4}
Note: To aid in verification, the chickenpox_by_sex()['female'] value the autograder is looking for starts with the digits 0.0077.
'''
def chickenpox_by_sex():
    # YOUR CODE HERE
    #是否感染Varicella
    cpox = df.HAD_CPOX
    #cpox.value_counts()
    cpox_group = list(df.groupby(by='HAD_CPOX'))
    have_cpox = cpox_group[0][1]
    not_have_cpox = cpox_group[1][1]
    #男女分开
    have_cpox_group = list(have_cpox.groupby(by='SEX'))
    not_have_cpox_group = list(not_have_cpox.groupby(by='SEX'))
    have_cpox_boy = have_cpox_group[0][1]
    have_cpox_girl = have_cpox_group[1][1]
    not_have_cpox_boy = not_have_cpox_group[0][1]
    not_have_cpox_girl = not_have_cpox_group[1][1]
    #接种感染
    #have_cpox_boy_injected = have_cpox_boy[(have_cpox_boy['P_NUMMMR']>0) | (have_cpox_boy['P_NUMVRC']>0)]
    have_cpox_boy_injected = have_cpox_boy[(have_cpox_boy['P_NUMVRC']>0)]
    num_have_cpox_boy_injected = have_cpox_boy_injected.count()['SEQNUMC']
    have_cpox_girl_injected = have_cpox_girl[(have_cpox_girl['P_NUMVRC']>0)]
    num_have_cpox_girl_injected = have_cpox_girl_injected.count()['SEQNUMC']
    #接种未感染
    not_have_cpox_boy_injected = not_have_cpox_boy[(not_have_cpox_boy['P_NUMVRC']>0)]
    num_not_have_cpox_boy_injected = not_have_cpox_boy_injected.count()['SEQNUMC']
    not_have_cpox_girl_injected = not_have_cpox_girl[(not_have_cpox_girl['P_NUMVRC']>0)]
    num_not_have_cpox_girl_injected = not_have_cpox_girl_injected.count()['SEQNUMC']
    #计算比例
    ratio_boy = num_have_cpox_boy_injected / num_not_have_cpox_boy_injected
    ratio_girl = num_have_cpox_girl_injected / num_not_have_cpox_girl_injected
    dic = {}
    dic['male'] = ratio_boy
    dic['female'] = ratio_girl
    return dic
    raise NotImplementedError()

assert len(chickenpox_by_sex())==2, "Return a dictionary with two items, the first for males and the second for females."

'''
Question 4
A correlation is a statistical relationship between two variables. If we wanted to know if vaccines work, we might look at the correlation between the use of the vaccine and whether it results in prevention of the infection or disease [1]. In this question, you are to see if there is a correlation between having had the chicken pox and the number of chickenpox vaccine doses given (varicella).

Some notes on interpreting the answer. The had_chickenpox_column is either 1 (for yes) or 2 (for no), and the num_chickenpox_vaccine_column is the number of doses a child has been given of the varicella vaccine. A positive correlation (e.g., corr > 0) means that an increase in had_chickenpox_column (which means more no’s) would also increase the values of num_chickenpox_vaccine_column (which means more doses of vaccine). If there is a negative correlation (e.g., corr < 0), it indicates that having had chickenpox is related to an increase in the number of vaccine doses.

Also, pval is the probability that we observe a correlation between had_chickenpox_column and num_chickenpox_vaccine_column which is greater than or equal to a particular value occurred by chance. A small pval means that the observed correlation is highly unlikely to occur by chance. In this case, pval should be very small (will end in e-18 indicating a very small number).

[1] This isn’t really the full picture, since we are not looking at when the dose was given. It’s possible that children had chickenpox and then their parents went to get them the vaccine. Does this dataset have the data we would need to investigate the timing of the dose?
'''
def corr_chickenpox():
    cpox = df[(df.P_NUMVRC).notnull()]
    have_cpox = cpox[(cpox.HAD_CPOX==1) | (cpox.HAD_CPOX==2)]
    df1=pd.DataFrame({"had_chickenpox_column":have_cpox.HAD_CPOX,
                      "num_chickenpox_vaccine_column":have_cpox.P_NUMVRC})
    corr, pval=stats.pearsonr(df1["had_chickenpox_column"],df1["num_chickenpox_vaccine_column"])
    return corr
    raise NotImplementedError()
