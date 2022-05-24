import logging
from matplotlib import container
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from threading import setprofile
from pandas._config.config import options
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle as pickle
import numpy as np
import tweepy
import preprocess_kgptalkie as pp
import csv
from textblob import TextBlob
from langdetect import detect
import json


clf = pickle.load(open('./clf_svm.pkl', 'rb'))
tfidf = pickle.load(open('./tfidf_svm.pkl', 'rb'))

# twitter analysis
consumer_key = 'KLKPj3Of20KJEuErqZZKerp5E'
consumer_secret = 'XzMQP1Yg00OjLfZuUpp8AWBa5xAltSNah9sQDH55c0nmLku3ou'
access_token = '1101485729811779584-27Is7eQfkx5pgY5ZaPEzbMqeQpwgZq'
access_token_secret = 'dbOJYPCgKUzCvzFQQGM1hmtw5xrQTuDhrv9B8SbCFMBxY'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
public_tweets = api.home_timeline()


def predict_sentiment(x):
    x = [x]
    sent = clf.predict(tfidf.transform(x))
    return sent


global pos_sent
global neg_sent


def test():
    lang = detect("kudasai")
    print(lang)


test()


class MyStreamListener(tweepy.StreamListener):
    global container
    container = []

    def on_status(self, status):
        print(status.text)

    def on_data(self, data):
        raw_twitts = json.loads(data)

        try:
            if len(container) == 10:
                return False
            x = str(raw_twitts['text']).lower()
            x = pp.cont_exp(x)
            x = pp.remove_emails(x)
            x = pp.remove_html_tags(x)
            x = pp.remove_rt(x)
            x = pp.remove_special_chars(x)
            x = pp.remove_urls(x)

            if track_keyword in x:
                container.append(x)
            else:
                pass
        except:
            pass

    def on_error(self, status_code):
        if status_code == 420:
            print("Error 420")
        return False


def analyze():

    global track_keyword
    track_keyword = input("Enter keyowrd here : ")
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
    myStream.filter(track=track_keyword)
    pos_sent = 0
    neg_sent = 0
    for x in container:
        print(x)
        sent = predict_sentiment(x)[0]
        if sent == 1:
            pos_sent = pos_sent + sent
        else:
            neg_sent = neg_sent + 1
    print("Number of tweets analysed to be Non-suicidal : ", pos_sent,
          "Number of tweets analysed to be suicidal : ", neg_sent)
