# Name - Bisistha Patra
# Student ID - 24159091
# Code for Pathway 2: Aspect-Based Sentiment Analysis (ABSA)

#all imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, html, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

#1.NAIVE BAYES FROM SCRATCH
class NaiveBayesScratch:
  #this is the same naive bayes as the one implemented from scratch for part 1 - 
    def __init__(self, alpha=1.0):
        self.alpha = alpha # Smoothing parameter

    def fit(self, X, y):
        # matching sizes and finding unique classes - setting the priors or likelihoods 
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.priors = np.zeros(len(self.classes))
        self.likelihoods = np.zeros((len(self.classes), n_features))
        
        for idx, c in enumerate(self.classes):
            # selecting only those rows from the current class
            X_c = X[y == c]

            # 1. Prior Calculation: P(Class)
            self.priors[idx] = X_c.shape[0] / n_samples
            
            #2.laplacian Smoothing
            # calculating likelihoods with smoothing to avoid zero probabilities 
            # formula: (word count + alpha) / (total words in class + alpha * vocab size)
            word_counts = np.sum(X_c, axis=0)
            total_words = np.sum(X_c)
            self.likelihoods[idx, :] = (word_counts + self.alpha) / (total_words + self.alpha * n_features)

    def predict(self, X):
        # using log to prevent numbers from getting too small (underflow)
        log_priors = np.log(self.priors)
        log_likelihoods = np.log(self.likelihoods)
        
        # calculating the weight of words for each class manually
        # this helps show the step-by-step logic instead of just using a shortcut
        scores = []
        for idx in range(len(self.classes)):
            # adding the word weights to the starting probability of the class
            class_score = X.dot(log_likelihoods[idx, :]) + log_priors[idx]
            scores.append(class_score)
            
        # picking the class with the highest total score
        scores = np.array(scores).T
        return self.classes[np.argmax(scores, axis=1)]
