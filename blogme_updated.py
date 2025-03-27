#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:35:34 2025

@author: jovannamelissa
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

articles = pd.read_excel('articles.xlsx')

articles.describe()
articles.info()


articles.groupby(['source_id'])['article_id'].count()
articles.groupby(['source_id'])['engagement_reaction_count'].sum()
articles = articles.drop('engagement_comment_plugin_count', axis = 1)
    
def keywordFlag(keyword):
    keyword_flag = []
    for i in articles['title']:
        try:
            if keyword in i:
                flag = 1
            else:
                flag = 0
        except:
            flag = 0
        keyword_flag.append(flag)
    
    return keyword_flag

k = keywordFlag('support')

articles['keyword_flag'] = pd.Series(k)

sent_int = SentimentIntensityAnalyzer()

title_neg_sentiment = []
title_neu_sentiment = []
title_pos_sentiment = []

for i in articles['title']:
    try:
        sent = sent_int.polarity_scores(i)
        
        title_neg_sentiment.append(sent['neg'])
        title_neu_sentiment.append(sent['neu'])
        title_pos_sentiment.append(sent['pos'])
    except:
        title_neg_sentiment.append(0)
        title_neu_sentiment.append(0)
        title_pos_sentiment.append(0)

articles['title_neg_sentiment'] = pd.Series(title_neg_sentiment)
articles['title_neu_sentiment'] = pd.Series(title_neu_sentiment)
articles['title_pos_sentiment'] = pd.Series(title_pos_sentiment)

articles.to_excel('blogmeCleaned.xlsx', sheet_name = 'BlogMeData', index = False)