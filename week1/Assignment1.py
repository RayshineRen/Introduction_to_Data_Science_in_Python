# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:36:07 2020

@author: Ray
@email: 1324789704@qq.com
@wechat: RayTing0305
"""
import re
def names():
    simple_string = """Amy is 5 years old, and her sister Mary is 2 years old. 
    Ruth and Peter! their parents, have 3 kids."""

    # YOUR CODE HERE
    pattern = '[A-Z]{1}[a-z]{0,100}(?=[\s,\.!@])'
    return re.findall(pattern, simple_string)
print(len(names()))
print(names())

def grades():
    with open ("assets/grades.txt", "r") as file:
        grades = file.read()

    # YOUR CODE HERE
    pattern = '[A-Z]{1}[a-z]{0,100}\s[A-Z]{1}[a-z]{0,100}:\s[B]'
    l = re.findall(pattern, grades)
    res = []
    for item in l:
        res.append(item[:-3])
    return res
    raise NotImplementedError()

print(len(grades()))
print(grades())


def logs():
    with open("assets/logdata.txt", "r") as file:
        logdata = file.read()
    pattern="""
    (?P<host>[\d\.]+)  
    (\ -\ )
    (?P<user_name>[\w-]*)
    (\ \[)
    (?P<time>.*)
    (\]\ ")
    (?P<request>.*)
    ("\ .*\ )
    (.*)
    """
    l=[]
    for item in re.finditer(pattern,logdata,re.VERBOSE):
        l.append(item.groupdict())
    return l
    # YOUR CODE HERE
    raise NotImplementedError()
print(len(logs()))
one_item={'host': '146.204.224.152',
  'user_name': 'feest6811',
  'time': '21/Jun/2019:15:45:24 -0700',
  'request': 'POST /incentivize HTTP/1.1'}
assert one_item in logs(), "Sorry, this item should be in the log results, check your formating"








