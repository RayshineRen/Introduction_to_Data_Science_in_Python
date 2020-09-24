# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:46:56 2020

@author: Ray
@wechat: RayTing0305
@email: 1324789704@qq.com
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

'''
Basic statistic test
'''
df = pd.read_csv('grades.csv')
early = df[df['assignment1_submission'] <= '2015-12-31']
#late = df[df['assignment1_submission'] > '2015-12-31']
late = df[~df.index.isin(early.index)]

t1 = stats.ttest_ind(early['assignment1_grade'], late['assignment1_grade'])

df1 = pd.DataFrame([np.random.random(100) for x in range(100)])
df2 = pd.DataFrame([np.random.random(100) for x in range(100)])
def test_columns(alpha=0.1):
    num_diff = 0
    for col in df1.columns:
        teststat, pvalue = stats.ttest_ind(df1[col], df2[col])
        if pvalue <= alpha:
            print("col {} is significantly different at alpha={}, pval={}".format(col, alpha, pvalue))
            num_diff += 1
    print("total {}, which is {}%".format(num_diff, float(num_diff) / len(df1) * 100))
        
# test_columns()















































































