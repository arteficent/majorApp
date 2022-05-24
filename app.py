
from unicodedata import numeric
import numpy as np
from flask import Flask, request, jsonify, render_template
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from threading import setprofile
from pandas._config.config import options
from textblob import TextBlob
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle as pickle
import tweepy
import preprocess_kgptalkie as pp
from langdetect import detect
import csv
import json

app = Flask(__name__)

# model loaded

clf_dt = pickle.load(open('./clf_dt.pkl', 'rb'))
tfidf_dt = pickle.load(open('./tfidf_dt.pkl', 'rb'))

clf_lr = pickle.load(open('./clf_lr.pkl', 'rb'))
tfidf_lr = pickle.load(open('./tfidf_lr.pkl', 'rb'))

clf_rf = pickle.load(open('./clf_rf.pkl', 'rb'))
tfidf_rf = pickle.load(open('./tfidf_rf.pkl', 'rb'))

clf_svm = pickle.load(open('./clf_svm.pkl', 'rb'))
tfidf_svm = pickle.load(open('./tfidf_svm.pkl', 'rb'))

# clf_xgb = pickle.load(open('./clf_xgb.pkl', 'rb'))
# tfidf_xgb = pickle.load(open('./tfidf_xgb.pkl', 'rb'))


# twitter database connection
consumer_key = 'KLKPj3Of20KJEuErqZZKerp5E'
consumer_secret = 'XzMQP1Yg00OjLfZuUpp8AWBa5xAltSNah9sQDH55c0nmLku3ou'
access_token = '1101485729811779584-27Is7eQfkx5pgY5ZaPEzbMqeQpwgZq'
access_token_secret = 'dbOJYPCgKUzCvzFQQGM1hmtw5xrQTuDhrv9B8SbCFMBxY'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
public_tweets = api.home_timeline()


# function to use model
def decision_tree(x):
    x = [x]
    sent = clf_dt.predict(tfidf_dt.transform(x))
    return sent


def logistic_regression(x):
    x = [x]
    sent = clf_lr.predict(tfidf_lr.transform(x))
    return sent


def random_forest(x):
    x = [x]
    sent = clf_rf.predict(tfidf_rf.transform(x))
    return sent


def support_vector_machine(x):
    x = [x]
    sent = clf_svm.predict(tfidf_svm.transform(x))
    return sent


# def XGBoost(x):
#     x = [x]
#     sent = clf_xgb.predict(tfidf_xgb.transform(x))
#     return sent


@app.route('/')
def home():
    return render_template('index.html')


class MyStreamListener(tweepy.StreamListener):
    global container
    container = []

    def on_status(self, status):
        print(status.text)

    def on_data(self, data):
        raw_twitts = json.loads(data)
        try:
            # print(len(container), " ", number_tweets)
            if len(container) == int(number_tweets):
                return False
            x = str(raw_twitts['text']).lower()
            x = pp.cont_exp(x)
            x = pp.remove_emails(x)
            x = pp.remove_html_tags(x)
            x = pp.remove_rt(x)
            x = pp.remove_special_chars(x)
            x = pp.remove_urls(x)
            lang = detect(x)
            # container.append(x)
            # print(lang, x)
            if lang == "en" and track_keyword in x:
                container.append(x)
                print(lang, " ", track_keyword, " ", x)

            else:
                pass
        except:
            pass

    def on_error(self, status_code):
        if status_code == 420:
            print("Error 420")
        return False


@app.route('/predict', methods=['POST'])
def predict():

    def analyze():
        global track_keyword
        global pos_sent
        global neg_sent
        global number_tweets
        global model
        track_keyword = request.form.get("input1")
        number_tweets = request.form.get("input2")
        model = request.form.get("input3")
        myStreamListener = MyStreamListener()
        myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
        myStream.filter(track=track_keyword)
        pos_sent = 0
        neg_sent = 0
        if model == "decision_tree":
            for x in container:
                # print(x)
                sent = decision_tree(x)[0]
                if sent == 1:
                    pos_sent = pos_sent + sent
                else:
                    neg_sent = neg_sent + 1
        elif model == "support_vector_machine":
            for x in container:
                # print(x)
                sent = support_vector_machine(x)[0]
                if sent == 1:
                    pos_sent = pos_sent + sent
                else:
                    neg_sent = neg_sent + 1
        elif model == "logistic_regression":
            for x in container:
                # print(x)
                sent = logistic_regression(x)[0]
                if sent == 1:
                    pos_sent = pos_sent + sent
                else:
                    neg_sent = neg_sent + 1
        elif model == "random_forest":
            for x in container:
                # print(x)
                sent = random_forest(x)[0]
                if sent == 1:
                    pos_sent = pos_sent + sent
                else:
                    neg_sent = neg_sent + 1
        else:
            for x in container:
                # print(x)
                # sent = XGBoost(x)[0]
                if sent == 1:
                    pos_sent = pos_sent + sent
                else:
                    neg_sent = neg_sent + 1

        print("Number of tweets analysed to be Non-suicidal : ", neg_sent,
              "Number of tweets analysed to be suicidal : ", pos_sent)
    analyze()
    return render_template('index.html', prediction_text='Number of tweets analysed to be suicidal :{}'.format(pos_sent), tweets=container)


if __name__ == "__main__":
    app.run(debug=True)
