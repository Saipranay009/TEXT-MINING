# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:32:54 2022

@author: Sai pranay
"""
#-----------------IMPORTING THE DATA-------------------------------------------

import pandas as pd
EL = pd.read_csv("E:\\DATA_SCIENCE_ASS\\TEXT_MINING\\Elon_musk.csv",encoding= 'latin1')
print(EL)
list(EL)
EL.shape
EL.info()
EL.describe()


import re

# Clean The Data
def cleantext(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text) # Remove Mentions
    text = re.sub(r"#", "", text) # Remove Hashtags Symbol
    text = re.sub(r"RT[\s]+", "", text) # Remove Retweets
    text = re.sub(r"https?:\/\/\S+", "", text) # Remove The Hyper Link
    
    return text

# Clean The Text
EL["Text"] = EL["Text"].apply(cleantext)
EL["Text"].head()

from textblob import TextBlob

# Get The Subjectivity
def sentiment_analysis(ds):
    sentiment = TextBlob(ds["Text"]).sentiment
    return pd.Series([sentiment.subjectivity, sentiment.polarity])

# Adding Subjectivity & Polarity
EL[["subjectivity", "polarity"]] = EL.apply(sentiment_analysis, axis=1)

EL

import matplotlib.pyplot as plt
from wordcloud import WordCloud

allwords = " ".join([twts for twts in EL["Text"]])
wordCloud = WordCloud(width = 1000, height = 1000, random_state = 21, max_font_size = 119).generate(allwords)
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

# Compute The Negative, Neutral, Positive Analysis
def analysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"
    
# Create a New Analysis Column
EL["analysis"] = EL["polarity"].apply(analysis)

# Print The Data
EL

positive_tweets = EL[EL['analysis'] == 'Positive']
negative_tweets = EL[EL['analysis'] == 'Negative']

print('positive tweets')
for i, row in positive_tweets[:5].iterrows():
  print(' -' + row['Text'])

print('negative tweets')
for i, row in negative_tweets[:5].iterrows():
  print(' -' + row['Text'])


plt.figure(figsize=(10, 8))

for i in range(0, EL.shape[0]):
    plt.scatter(EL["polarity"][i], EL["subjectivity"][i], color = "Red")

plt.title("Sentiment Analysis") # Add The Graph Title
plt.xlabel("Polarity") # Add The X-Label
plt.ylabel("Subjectivity") # Add The Y-Label
plt.show() 


len(positive_tweets) / len(negative_tweets)

#Since that number is positive, and quite high of a ratio, we can also conclude that Elon is a positive guy.