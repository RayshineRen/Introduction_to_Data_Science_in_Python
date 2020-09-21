# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 08:37:48 2020

@author: Ray
@email: 1324789704@qq.com
@wechat: RayTing0305
"""

import numpy as np
import matplotlib.pyplot as plt
import re


'''
Numpy_ed
'''
#from PIL import Image
#from IPython.display import display
#
## And let's just look at the image I'm talking about
#im = Image.open('chris.tiff')
##display(im)
#im_array = np.array(im)
##print(im_array.shape)
##print(im_array.dtype)
##the uint means that they are
##unsigned integers (so no negative numbers) 
##and the 8 means 8 bits per byte.
##black is stored as 0 and white is stored as 255.
#mask = np.full(im_array.shape, 255)
#modified_array = (mask - im_array).astype(np.uint8)
##display(Image.fromarray(modified_array))
#
#reshaped = np.reshape(modified_array, (100, 400))
##display(Image.fromarray(reshaped))
'''
为什么是整幅图像是被复制了一份？而不是上下分割呢？
'''

#house_price = np.genfromtxt("house_prices.csv", delimiter=";", skip_header=1)

#wines = np.genfromtxt("winequality-red.csv", delimiter=";", skip_header=1)
##https://blog.csdn.net/weixin_43590796/article/details/107479796
##wines[:, [0,2,4]]
##wines[:, 0:3]
##wines[:, 0]
##wines[:, 0:1]
#graduate_admission = np.genfromtxt('Admission_Predict.csv', dtype=None, delimiter=',', skip_header=1,
#                                   names=('Serial No','GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
#                                          'LOR','CGPA','Research', 'Chance of Admit'))
#
##gra = np.genfromtxt('Admission_Predict.csv', dtype=None, delimiter=',', skip_header=1)
#
##We can retrieve a column from the array using the column's name
##print(graduate_admission['CGPA'])
#graduate_admission['CGPA'] = graduate_admission['CGPA'] /10 *4
##print(len(graduate_admission[graduate_admission['Research'] == 1]))
#
##print(graduate_admission[graduate_admission['Chance_of_Admit'] > 0.8]['GRE_Score'].mean())
##print(graduate_admission[graduate_admission['Chance_of_Admit'] < 0.4]['GRE_Score'].mean())
#
##print(graduate_admission[graduate_admission['Chance_of_Admit'] > 0.8]['CGPA'].mean())
##print(graduate_admission[graduate_admission['Chance_of_Admit'] < 0.4]['CGPA'].mean())


'''
Regex_ed
'''

text = "Amy works diligently. Amy gets good grades. Our student Amy is succesful."
l = re.split("Amy", text)

l = re.findall("Amy", text)

text = "Amy works diligently. Amy gets good grades. Our student Amy is succesful."
l = re.search("^Amy",text)
grades = "BDYACAAAABCBCBAA"

# If we want to answer the question "How many B's were in the grade list?" we would just use B
l = re.findall("B",grades)
# If we wanted to count the number of A's or B's in the list, we can't use "AB" since this is used to match
# all A's followed immediately by a B. Instead, we put the characters A and B inside square brackets
l = re.findall("[AB]",grades)

l = re.findall("[A][B-C]",grades)

l = re.findall("AB|AC",grades)

# We can use the caret with the set operator to negate our results. For instance, if we want to parse out only
# the grades which were not A's
l = re.findall("[^A]",grades)

l = re.findall("^[^A][A-Z]",grades)

# Let's use these grades as an example. How many times has this student been on a back-to-back A's streak?
l = re.findall("A{2,10}",grades) # we'll use 2 as our min, but ten as our max

# We might try and do this using single values and just repeating the pattern
l = re.findall("A{1,1}A{1,1}",grades)

# It's important to note that the regex quantifier syntax does not allow you to deviate from the {m,n}
# pattern. In particular, if you have an extra space in between the braces you'll get an empty result
l = re.findall("A{2, 2}",grades)
#correct:l = re.findall("A{2,2}",grades)

# And as we have already seen, if we don't include a quantifier then the default is {1,1}
l = re.findall("AA",grades)


# Oh, and if you just have one number in the braces, it's considered to be both m and n
l = re.findall("A{2}",grades)

# Using this, we could find a decreasing trend in a student's grades
l = re.findall("A{1,10}B{0,10}C{0,10}",grades)


with open('week1/dataset/ferpa.txt', 'r') as file:
    wiki = file.read()
#print(len(wiki))

headers = re.findall("[a-zA-Z]{1,100}\[edit\]", wiki)
headers = re.findall("[\w]{1,100}\[edit\]", wiki)
#use \w to match any letter, including digits and numbers.
#\s matches any whitespace character.
#use asterix * to match 0 or more times, so let's try that.
headers = re.findall("[\w]*\[edit\]", wiki)
#We can add in a spaces using the space character
headers = re.findall("[\w ]*\[edit\]",wiki)

# why not use [\w\s]
#for title in re.findall("[\w ]*\[edit\]",wiki):
#    # Now we will take that intermediate result and split on the square bracket [ just taking the first result
#    print(re.split("[\[]",title)[0])



# Ok, this works, but it's a bit of a pain. To this point we have been talking about a regex as a single
# pattern which is matched. But, you can actually match different patterns, called groups, at the same time,
# and then refer to the groups you want. To group patterns together you use parentheses, which is actually
# pretty natural. Lets rewrite our findall using groups
headers = re.findall("([\w ]*)(\[edit\])",wiki)
#for item in re.finditer("([\w ]*)(\[edit\])",wiki):
#    print(item.groups())

# We see here that the groups() method returns a tuple of the group. We can get an individual group using
# group(number), where group(0) is the whole match, and each other number is the portion of the match we are
# interested in. In this case, we want group(1)
#for item in re.finditer("([\w ]*)(\[edit\])",wiki):
#    print(item.group(1))

# One more piece to regex groups that I rarely use but is a good idea is labeling or naming groups. In the
# previous example I showed you how you can use the position of the group. But giving them a label and looking
# at the results as a dictionary is pretty useful. For that we use the syntax (?P<name>), where the parethesis
# starts the group, the ?P indicates that this is an extension to basic regexes, and <name> is the dictionary
# key we want to use wrapped in <>.
#for item in re.finditer("(?P<title>[\w ]*)(?P<edit_link>\[edit\])",wiki):
#    # We can get the dictionary returned for the item with .groupdict()
#    print(item.groupdict()['title'])

# Of course, we can print out the whole dictionary for the item too, and see that the [edit] string is still
# in there. Here's the dictionary kept for the last match
#print(item.groupdict())




# Let's look at some more wikipedia data. Here's some data on universities in the US which are buddhist-based
with open("week1/dataset/buddhist.txt","r",encoding='utf-8') as file:
    # we'll read that into a variable called wiki
    wiki=file.read()
# and lets print that variable out to the screen
pattern="""
(?P<title>.*)        #the university title
(–\ located\ in\ )   #an indicator of the location
(?P<city>\w*)        #city the university is in
(,\ )                #separator for the state
(?P<state>\w*)       #the state the city is located in"""
# Now when we call finditer() we just pass the re.VERBOSE flag as the last parameter, this makes it much
# easier to understand large regexes!
#for item in re.finditer(pattern,wiki,re.VERBOSE):
#    # We can get the dictionary returned for the item with .groupdict()
#    print(item.groupdict())

# Here's another example from the New York Times which covers health tweets on news items. This data came from
# the UC Irvine Machine Learning Repository which is a great source of different kinds of data
with open("week1/dataset/nytimeshealth.txt","r",encoding='utf-8') as file:
    # We'll read everything into a variable and take a look at it
    health=file.read()
# So lets create a pattern. We want to include the hash sign first, then any number of alphanumeric
# characters. And we end when we see some whitespace
pattern = '#[\w\d]*(?=\s)'

# Notice that the ending is a look ahead. We're not actually interested in matching whitespace in the return
# value. Also notice that I use an asterix * instead of the plus + for the matching of alphabetical characters
# or digits, because a + would require at least one of each

# Lets searchg and display all of the hashtags
l = re.findall(pattern, health)

'''
so what is look-ahead
'''
# One more concept to be familiar with is called "look ahead" and "look behind" matching. In this case, the
# pattern being given to the regex engine is for text either before or after the text we are trying to
# isolate. For example, in our headers we want to isolate text which  comes before the [edit] rendering, but
# we actually don't care about the [edit] text itself. Thus far we have been throwing the [edit] away, but if
# we want to use them to match but don't want to capture them we could put them in a group and use look ahead
# instead with ?= syntax
#for item in re.finditer("(?P<title>[\w ])+(?=\[edit\])",wiki):
#    # What this regex says is match two groups, the first will be named and called title, will have any amount
#    # of whitespace or regular word characters, the second will be the characters [edit] but we don't actually
#    # want this edit put in our output match objects
#    print(item)



