# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:56:15 2020

@author: Ray
@email: 1324789704@qq.com
@wechat: RayTing0305
"""

'''
Question 1
Write a function called proportion_of_education which returns the proportion of children in the dataset who had a mother with the education levels equal to less than high school (<12), high school (12), more than high school but not a college graduate (>12) and college degree.

This function should return a dictionary in the form of (use the correct numbers, do not round numbers):

    {"less than high school":0.2,
    "high school":0.4,
    "more than high school but not college":0.2,
    "college":0.2}
'''

import scipy.stats as stats
import numpy as np
import pandas as pd

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
