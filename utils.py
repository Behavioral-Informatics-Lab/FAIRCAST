from tkinter import Y
from turtle import distance
# from weakref import proxy
import numpy as np
import pandas as pd

import quapy as qp

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import operator

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from numpy.linalg import norm
import utils
import helper
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
svm = LinearSVC()

import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

import os
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

qp.environ['SAMPLE_SIZE'] = 100  # once for all


def get_features_labels(df):
    
    X = df.drop('y', axis=1)
    y = df['y'].values
    
    return X, y


def get_joint(df):
    
    FP = (df.query('y == 1 and sensitive == 1').shape[0] / df.shape[0])
    DP =  (df.query('y == 1 and sensitive == 0').shape[0] / df.shape[0])
    FN = (df.query('y == 0 and sensitive == 1').shape[0] / df.shape[0])
    DN = (df.query('y == 0 and sensitive == 0').shape[0] / df.shape[0])
    
    return np.array([[DN, FN],[DP, FP]])


def who_is_priviliged(y, sensitive):
    
    df = pd.DataFrame()
    df['y'] = y
    df['sensitive'] = sensitive
    
    # P(Y=1 | S)
    
    base_rate_S0 = df.query('y == 1 and sensitive == 0').shape[0] / df.query('sensitive == 0').shape[0]
    
    base_rate_S1 = df.query('y == 1 and sensitive == 1').shape[0] / df.query('sensitive == 1').shape[0]
    
    if base_rate_S0 > base_rate_S1:
        print('S0 is privilged')
    
    else:
        print('S1 is privilged')

    return np.array([base_rate_S0, base_rate_S1])


def base_rate_difference(y, sensitive):
    '''
    Measure the base rate difference in the dataset

    :param df: the test dataset along with the y and pred 

    :return the statistical measure  | P(Y = 1 | S = 0) - P(Y = 1 | S = 1) | 

    '''
    df = pd.DataFrame()
    df['y'] = y
    df['sensitive'] = sensitive
    
    p_y_1_given_s_0 = len(df.query('y == 1 and sensitive == 0')) / len(df.query('sensitive == 0'))


    p_y_1_given_s_1 = len(df.query('y == 1 and sensitive == 1')) / len(df.query('sensitive == 1'))
    
    
    base_rate_difference = p_y_1_given_s_1 - p_y_1_given_s_0  
    return base_rate_difference


def parity_difference(pred, sensitive):
    '''
    Measure the statistical parity from the predictions 

    :param df: the test dataset along with the y and pred 

    :return the statistical measure  | P(Y` = 1 | S = 0) - P(Y` = 1 | S = 1) | 

    '''

    df = pd.DataFrame()
    df['pred'] = pred
    df['sensitive'] = sensitive

    p_yhat1_s0 = len(df[(df['pred'] == 1) & (df['sensitive'] == 0)])
    p_s0 = len(df[df['sensitive'] == 0])

    p_yhat1_s1 = len(df[(df['pred'] == 1) & (df['sensitive'] == 1)])
    p_s1 = len(df[df['sensitive'] == 1])

    return np.abs((p_yhat1_s1 / p_s1)- (p_yhat1_s0 / p_s0))
    
def parity_difference_unfavored(pred, sensitive):
    '''
    Measure the statistical parity from the predictions - for the negative class (Just an extra)

    :param df: the test dataset along with the y and pred 

    :return the statistical measure  | P(Y` = 0 | S = 0) - P(Y` = 0 | S = 1) | 

    '''

    df = pd.DataFrame()
    df['pred'] = pred
    df['sensitive'] = sensitive

    p_yhat0_s0 = len(df[(df['pred'] == 0) & (df['sensitive'] == 0)])
    p_s0 = len(df[df['sensitive'] == 0])

    p_yhat0_s1 = len(df[(df['pred'] == 0) & (df['sensitive'] == 1)])
    p_s1 = len(df[df['sensitive'] == 1])

    return np.abs((p_yhat0_s1 / p_s1)- (p_yhat0_s0 / p_s0))

def equal_opportunity(y, pred, sensitive):
    '''
    Measure the equal opportunity from the predictions aka "Delta TPR" 

    :param df: the test dataset along with the y and pred 

    :return the delta TPR  | P(Y` = 1 | Y = 1, S = 0) - P(Y` = 1 | Y = 1, S = 1) | 

    '''
    df = pd.DataFrame()
    df['y'] = y
    df['pred'] = pred
    df['sensitive'] = sensitive

    p_yhat1_y1_s0 = len(df[(df['pred'] == 1) & (df['y'] == 1) & (df['sensitive'] == 0)])
    p_y1_s0 = len(df[(df['y'] == 1) & (df['sensitive'] == 0)])


    p_yhat1_y1_s1 = len(df[(df['pred'] == 1) & (df['y'] == 1) & (df['sensitive'] == 1)])
    p_y1_s1 = len(df[(df['y'] == 1) & (df['sensitive'] == 1)])

    TPR_S0 =  p_yhat1_y1_s0 / p_y1_s0
    TPR_S1 = p_yhat1_y1_s1 / p_y1_s1

    delta_TPR = np.abs((p_yhat1_y1_s0 / p_y1_s0) - (p_yhat1_y1_s1 / p_y1_s1))

    return np.array([TPR_S1, TPR_S0, delta_TPR])


def false_negative_equality(y, pred, sensitive):
    '''
    Measure the  "Delta FNR" 

    :param df: the test dataset along with the y and pred 

    :return the delta FNR  | P(Y` = 0 | Y = 1, S = 0) - P(Y` = 0 | Y = 1, S = 1) | 

    '''
    df = pd.DataFrame()
    df['y'] = y
    df['pred'] = pred
    df['sensitive'] = sensitive

    p_yhat0_y1_s0 = len(df[(df['pred'] == 0) & (df['y'] == 1) & (df['sensitive'] == 0)])
    p_y1_s0 = len(df[(df['y'] == 1) & (df['sensitive'] == 0)])


    p_yhat0_y1_s1 = len(df[(df['pred'] == 0) & (df['y'] == 1) & (df['sensitive'] == 1)])
    p_y1_s1 = len(df[(df['y'] == 1) & (df['sensitive'] == 1)])

    FNR_S0 =  p_yhat0_y1_s0 / p_y1_s0
    FNR_S1 = p_yhat0_y1_s1 / p_y1_s1

    delta_FNR = np.abs((p_yhat0_y1_s0 / p_y1_s0) - (p_yhat0_y1_s1 / p_y1_s1))

    return np.array([FNR_S1, FNR_S0, delta_FNR])

def predictive_equality(y, pred, sensitive):
    '''
    Measure the predictive equality from the predictions aka "Delta FPR" 

    :param df: the test dataset along with the y and pred 

    :return the delta FPR  | P(Y` = 1 | Y = 0, S = 0) - P(Y` = 1 | Y = 0, S = 1) | 

    '''

    df = pd.DataFrame()
    df['y'] = y 
    df['pred'] = pred
    df['sensitive'] = sensitive

    p_yhat1_y0_s0 = len(df[(df['pred'] == 1) & (df['y'] == 0) & (df['sensitive'] == 0)])
    p_y0_s0 = len(df[(df['y'] == 0) & (df['sensitive'] == 0)])


    p_yhat1_y0_s1 = len(df[(df['pred'] == 1) & (df['y'] == 0) & (df['sensitive'] == 1)])
    p_y0_s1 = len(df[(df['y'] == 0) & (df['sensitive'] == 1)])

    FPR_S0 = p_yhat1_y0_s0 / p_y0_s0
    FPR_S1 = p_yhat1_y0_s1 / p_y0_s1
 
    delta_FPR = np.abs((p_yhat1_y0_s0 / p_y0_s0) - (p_yhat1_y0_s1 / p_y0_s1))

    return np.array([FPR_S1, FPR_S0, delta_FPR])


def get_weight_from_joint(joint):
    
    eps = 0.0001

    deprived = joint.sum(axis=0)[0] # P(S=0)
    favored = joint.sum(axis=0)[1]   #P(S=1)
    DP = joint[1,0]
    DN = joint[0,0]
    FP = joint[1,1]
    FN = joint[0,1]

    negative =  joint.sum(axis=1)[0] #P(Y=0)
    positve =  joint.sum(axis=1)[1]  #P(Y=1)

    W_DP = (deprived * positve) / ( DP + eps)
    W_DN = (deprived * negative) / (DN + eps)
    W_FP = (favored * positve) / (FP + eps)
    W_FN = (favored * negative) / (FN + eps)
    
    return np.array([W_FP, W_DP, W_FN, W_DN])

def reweight_dataset(df, sensitive, label, weights):

    W_FP, W_DP, W_FN, W_DN = weights

    df_ = df.copy()

    df_['weight'] = np.ones(len(df_))
    
    df_['weight'] = np.where((df_[sensitive] == 0) & (df_[label] == 1), W_DP, df_['weight'])
    df_['weight'] = np.where((df_[sensitive] == 0) & (df_[label] == 0), W_DN, df_['weight'])
    df_['weight'] = np.where((df_[sensitive] == 1) & (df_[label] == 1), W_FP, df_['weight'])
    df_['weight'] = np.where((df_[sensitive] == 1) & (df_[label] == 0), W_FN, df_['weight'])
    
    return df_['weight'].values

def group_inverse_weights(df, sensitive, label, verbose = True):
    
   

    '''
        1 / P(S,Y) idrissi
    '''
    
    deprived = (df[df[sensitive] == 0]).shape[0] # P(S=0)
    favored = (df[df[sensitive] == 1]).shape[0]  # P(S=1)
    
    DP = df[(df[sensitive] == 0) & (df[label] == 1)].shape[0]  # P(S=0, Y=+)
    DN = df[(df[sensitive] == 0) & (df[label] == 0)].shape[0]  # P(S=0, Y=-)
    FP = df[(df[sensitive] == 1) & (df[label] == 1)].shape[0]  # P(S=1, Y=+)
    FN = df[(df[sensitive] == 1) & (df[label] == 0)].shape[0]  # P(S=1, Y=-)
    
    positve = df[df[label] == 1].shape[0]   # P(Y=+)
    negative = df[df[label] == 0].shape[0]  # P(Y=-)
    n = df.shape[0]
    
    ## Get the weights 
    
    W_DP = (1 / DP) 
    W_DN = (1 / DN)
    W_FP = (1 / FP) 
    W_FN = (1 / FN) 

    if verbose:
        df_ = df.copy()

        df_['weight'] = np.ones(len(df_))
        
        df_['weight'] = np.where((df_[sensitive] == 0) & (df_[label] == 1), W_DP, df_['weight'])
        df_['weight'] = np.where((df_[sensitive] == 0) & (df_[label] == 0), W_DN, df_['weight'])
        df_['weight'] = np.where((df_[sensitive] == 1) & (df_[label] == 1), W_FP, df_['weight'])
        df_['weight'] = np.where((df_[sensitive] == 1) & (df_[label] == 0), W_FN, df_['weight'])
        
        return df_['weight'].values
    else:
        return np.array([W_FP, W_DP, W_FN, W_DN])

def kamiran(df, sensitive, label, verbose = True):
    
    eps = 0.0001

    '''
        Kamiran Paper
    '''
    
    deprived = (df[df[sensitive] == 0]).shape[0] # P(S=0)
    favored = (df[df[sensitive] == 1]).shape[0]  # P(S=1)
    
    DP = df[(df[sensitive] == 0) & (df[label] == 1)].shape[0]  # P(S=0, Y=+)
    DN = df[(df[sensitive] == 0) & (df[label] == 0)].shape[0]  # P(S=0, Y=-)
    FP = df[(df[sensitive] == 1) & (df[label] == 1)].shape[0]  # P(S=1, Y=+)
    FN = df[(df[sensitive] == 1) & (df[label] == 0)].shape[0]  # P(S=1, Y=-)
    
    positve = df[df[label] == 1].shape[0]   # P(Y=+)
    negative = df[df[label] == 0].shape[0]  # P(Y=-)
    n = df.shape[0]
    
    ## Get the weights 
    
    W_DP = (deprived * positve) / ((n * DP) + eps)
    W_DN = (deprived * negative) / ((n * DN) + eps)
    W_FP = (favored * positve) / ((n * FP) + eps)
    W_FN = (favored * negative) / ((n * FN) + eps)

    if verbose:
        df_ = df.copy()

        df_['weight'] = np.ones(len(df_))
        
        df_['weight'] = np.where((df_[sensitive] == 0) & (df_[label] == 1), W_DP, df_['weight'])
        df_['weight'] = np.where((df_[sensitive] == 0) & (df_[label] == 0), W_DN, df_['weight'])
        df_['weight'] = np.where((df_[sensitive] == 1) & (df_[label] == 1), W_FP, df_['weight'])
        df_['weight'] = np.where((df_[sensitive] == 1) & (df_[label] == 0), W_FN, df_['weight'])
        
        return df_['weight'].values
    else:
        return np.array([W_FP, W_DP, W_FN, W_DN])
    
    
def base_rate(df):
    
    p_1_given_1 = df.query('y == 1 and sensitive == 1').shape[0] / df.query('sensitive == 1').shape[0]
    
    p_1_given_0 = df.query('y == 1 and sensitive == 0').shape[0] / df.query('sensitive == 0').shape[0]
    
    return p_1_given_1, p_1_given_0

def spurious_correlation(df):

    p_1_given_1 = df.query('y == 1 and sensitive == 1').shape[0] / df.query('sensitive == 1').shape[0]
    
    p_1_given_0 = df.query('y == 1 and sensitive == 0').shape[0] / df.query('sensitive == 0').shape[0]

    p_0_given_1 = df.query('y == 0 and sensitive == 1').shape[0] / df.query('sensitive == 1').shape[0]

    p_0_given_0 = df.query('y == 0 and sensitive == 0').shape[0] / df.query('sensitive == 0').shape[0]

    return np.array([p_1_given_1, p_1_given_0, p_0_given_1, p_0_given_0])

def custom_auc(y, proba, sensitive):
    
    'AUROC'
    
    df = pd.DataFrame()
    df['y'] = y
    df['pred'] = proba
    df['sensitive'] = sensitive
    
    ## AUC for S = 1
    S1 = df[df['sensitive'] == 1]
    try:
        S1_auc = metrics.roc_auc_score(S1['y'], S1['pred'])
    except ValueError:
        S1_auc = 0.01
    
    
    
    ## AUC for S = 0
    S0 = df[df['sensitive'] == 0]
    try:
        S0_auc = metrics.roc_auc_score(S0['y'], S0['pred'])
    except ValueError:
        S0_auc = 0.01
    
    
    worst_auc = np.min([S1_auc, S0_auc])
    
    return S1_auc, S0_auc, worst_auc


def custom_acc(y, pred, sensitive):
    
    'ACC'
    
    df = pd.DataFrame()
    df['y'] = y
    df['pred'] = pred
    df['sensitive'] = sensitive
    
    ## ACC for S = 1
    S1 = df[df['sensitive'] == 1]
    S1_acc = metrics.accuracy_score(S1['y'], S1['pred'])
    
    
    ## ACC for S = 0
    S0 = df[df['sensitive'] == 0]
    S0_acc = metrics.accuracy_score(S0['y'], S0['pred'])
    
    worst_acc = np.min([S1_acc, S0_acc])
    
    return S1_acc, S0_acc, worst_acc




def estimate_test_spurious(training_set, testing_set):
    '''
    This method shall return the estimated P'(Y|S)
    
    '''
    
    # Step 1
    S1_train = training_set[training_set['sensitive'] == 1].drop(['sensitive'],axis = 1)
    S0_train = training_set[training_set['sensitive'] == 0].drop(['sensitive'],axis = 1)
    
    S1_data_train = qp.data.LabelledCollection(S1_train.drop('y',axis=1), S1_train.y)
    S0_data_train = qp.data.LabelledCollection(S0_train.drop('y', axis = 1), S0_train.y)
    
    # Step 2
    model_S1 = qp.method.aggregative.EMQ(LogisticRegression()).fit(S1_data_train)
    model_S0 = qp.method.aggregative.EMQ(LogisticRegression()).fit(S0_data_train)
    
    # Step 3
    S1_test = testing_set[testing_set['sensitive'] == 1].drop(['sensitive'], axis = 1)
    S0_test = testing_set[testing_set['sensitive'] == 0].drop(['sensitive'], axis = 1)
    
    S1_data_test = qp.data.LabelledCollection(S1_test.drop('y', axis = 1), S1_test.y)
    S0_data_test = qp.data.LabelledCollection(S0_test.drop('y', axis = 1), S0_test.y)
    
    # P'(Y| S = 1)
    p_y_given_S1 = model_S1.quantify(S1_data_test.instances)
    
    # P'(Y | S=0)
    p_y_given_S0 = model_S0.quantify(S0_data_test.instances)
    
    return p_y_given_S1, p_y_given_S0, S1_data_train.prevalence(), S0_data_train.prevalence()


def estimate_test(training_set, testing_set):
    '''
    This method shall return the estimated P'(Y|S)
    
    '''
    
    # Step 1
    S1_train = training_set[training_set['sensitive'] == 1].drop(['year','sensitive'],axis = 1)
    S0_train = training_set[training_set['sensitive'] == 0].drop(['year','sensitive'],axis = 1)
    
    S1_data_train = qp.data.LabelledCollection(S1_train.drop('y',axis=1), S1_train.y)
    S0_data_train = qp.data.LabelledCollection(S0_train.drop('y', axis = 1), S0_train.y)
    
    # Step 2
    model_S1 = qp.method.aggregative.EMQ(RandomForestClassifier()).fit(S1_data_train)
    model_S0 = qp.method.aggregative.EMQ(RandomForestClassifier()).fit(S0_data_train)
    
    # Step 3
    S1_test = testing_set[testing_set['sensitive'] == 1].drop(['year','sensitive'], axis = 1)
    S0_test = testing_set[testing_set['sensitive'] == 0].drop(['year','sensitive'], axis = 1)
    
    S1_data_test = qp.data.LabelledCollection(S1_test.drop('y', axis = 1), S1_test.y)
    S0_data_test = qp.data.LabelledCollection(S0_test.drop('y', axis = 1), S0_test.y)
    
    # P'(Y| S = 1)
    p_y_given_S1 = model_S1.quantify(S1_data_test.instances)
    
    # P'(Y | S=0)
    p_y_given_S0 = model_S0.quantify(S0_data_test.instances)
    
    return p_y_given_S1, p_y_given_S0, S1_data_train.prevalence(), S0_data_train.prevalence()


def estimate_test_Toxicity(training_set, testing_set):
    '''
    This method shall return the estimated P'(Y|S)
    
    '''
    
    # Step 1
    S1_train = training_set[training_set['sensitive'] == 1].drop(['year','sensitive'],axis = 1)
    S0_train = training_set[training_set['sensitive'] == 0].drop(['year','sensitive'],axis = 1)
    
    # TFIDF-Features
    vectorizor = TfidfVectorizer(analyzer='word', tokenizer=nltk.word_tokenize,
                       preprocessor=None, stop_words='english', max_features=5000)
    
    tfidf = vectorizor.fit(training_set['comment_text'])
    S1_train_text_idf = vectorizor.transform(S1_train['comment_text'])
    S0_train_text_idf = vectorizor.transform(S0_train['comment_text'])
    
    
    S1_data_train = qp.data.LabelledCollection(S1_train_text_idf, S1_train.y)
    S0_data_train = qp.data.LabelledCollection(S0_train_text_idf, S0_train.y)
    
    # Step 2
    model_S1 = qp.method.aggregative.EMQ(LogisticRegression()).fit(S1_data_train)
    model_S0 = qp.method.aggregative.EMQ(LogisticRegression()).fit(S0_data_train)
    
    # Step 3
    S1_test = testing_set[testing_set['sensitive'] == 1].drop(['year','sensitive'], axis = 1)
    S0_test = testing_set[testing_set['sensitive'] == 0].drop(['year','sensitive'], axis = 1)
    
    S1_test_text_idf = vectorizor.transform(S1_test['comment_text'])
    S0_test_text_idf = vectorizor.transform(S0_test['comment_text'])
    
    S1_data_test = qp.data.LabelledCollection(S1_test_text_idf, S1_test.y)
    S0_data_test = qp.data.LabelledCollection(S0_test_text_idf, S0_test.y)
    
    # P'(Y| S = 1)
    p_y_given_S1 = model_S1.quantify(S1_data_test.instances)
    
    # P'(Y | S=0)
    p_y_given_S0 = model_S0.quantify(S0_data_test.instances)
    
    return p_y_given_S1, p_y_given_S0, S1_data_train.prevalence(), S0_data_train.prevalence()

def estimate_test_Claim(training_set, testing_set):
    '''
    This method shall return the estimated P'(Y|S)
    
    '''
    
    # Step 1
    S1_train = training_set[training_set['sensitive'] == 1].drop(['year','sensitive'],axis = 1)
    S0_train = training_set[training_set['sensitive'] == 0].drop(['year','sensitive'],axis = 1)
    
    # TFIDF-Features
    vectorizor = TfidfVectorizer(max_features=1000)
    
    tfidf = vectorizor.fit(training_set['text'])
    S1_train_text_idf = vectorizor.transform(S1_train['text'])
    S0_train_text_idf = vectorizor.transform(S0_train['text'])
    
    
    S1_data_train = qp.data.LabelledCollection(S1_train_text_idf, S1_train.y)
    S0_data_train = qp.data.LabelledCollection(S0_train_text_idf, S0_train.y)
    
    # Step 2
    model_S1 = qp.method.aggregative.EMQ(LogisticRegression()).fit(S1_data_train)
    model_S0 = qp.method.aggregative.EMQ(LogisticRegression()).fit(S0_data_train)
    
    # Step 3
    S1_test = testing_set[testing_set['sensitive'] == 1].drop(['year','sensitive'], axis = 1)
    S0_test = testing_set[testing_set['sensitive'] == 0].drop(['year','sensitive'], axis = 1)
    
    S1_test_text_idf = vectorizor.transform(S1_test['text'])
    S0_test_text_idf = vectorizor.transform(S0_test['text'])
    
    S1_data_test = qp.data.LabelledCollection(S1_test_text_idf, S1_test.y)
    S0_data_test = qp.data.LabelledCollection(S0_test_text_idf, S0_test.y)
    
    # P'(Y| S = 1)
    p_y_given_S1 = model_S1.quantify(S1_data_test.instances)
    
    # P'(Y | S=0)
    p_y_given_S0 = model_S0.quantify(S0_data_test.instances)
    
    return p_y_given_S1, p_y_given_S0, S1_data_train.prevalence(), S0_data_train.prevalence()

def get_results(sensitive, y_test, pred, prob):

    ## Whole Accuracy
    acc = metrics.accuracy_score(y_test, pred)

    ## Whole Accruacy (AUC)
    auc = metrics.roc_auc_score(y_test, prob)
    
    # Accuracy for S=1 and S=0 separately 
    S1_acc, S0_acc, worst_acc = custom_acc(y_test, pred, sensitive)
    
    # Delta Accuracy
    delta_acc = np.abs(S1_acc - S0_acc)
    
    # AUC for S=1 and S=0 separately 
    S1_auc, S0_auc, worst_auc = custom_auc(y_test, prob, sensitive)

    # S1_auc = metrics.roc_auc_score(y_test[sensitive==1], prob[sensitive==1])
    # S0_auc = metrics.roc_auc_score(y_test[sensitive==0], prob[sensitive==0])
    
    # Delta AUC
    delta_auc = np.abs(S1_auc - S0_auc)
    
        
    ## SPD (favored class 1)
    spd = parity_difference(pred, sensitive)

     ## SPD (unfavored class 0)
    spd_unfav = parity_difference_unfavored(pred, sensitive)
    
    ## TPR
    tpr = metrics.recall_score(y_test, pred)
    
    tpr_s1, tpr_s0, tprd = equal_opportunity(y_test, pred, sensitive)
    
    ## FPR
    fpr = 1 - metrics.recall_score(y_test, pred, pos_label=0)
    fpr_s1, fpr_s0, fprd = predictive_equality(y_test, pred, sensitive)

    ## FNR
    fnr_s1, fnr_s0, fnrd = false_negative_equality(y_test, pred, sensitive)

    ## F1
    f1 = metrics.f1_score(y_test, pred)

    ## F1 macro
    f1_macro = metrics.f1_score(y_test, pred, average='macro')

    ## F1 micro
    f1_micro = metrics.f1_score(y_test, pred, average='micro')
    
    # F1 Weighted 
    f1_weighted = metrics.f1_score(y_test, pred, average='weighted')
    

    # # Printing 
    results = [acc,
               auc, 
               S1_acc, 
               S0_acc,
               worst_acc,
               delta_acc,
               S1_auc,
               S0_auc,
               worst_auc,
               delta_auc,
               spd,
               tpr_s1,
               tpr_s0, 
               tprd,
               fpr_s1,
               fpr_s0, 
               fprd,
               fnr_s1,
               fnr_s0,
               fnrd,
               f1,
               f1_macro,
               f1_micro,
               f1_weighted
              ]

    return results



def estimate_test_lr(training_set, testing_set):
    '''
    This method shall return the estimated P'(Y|S)
    
    '''
    
    # Step 1
    S1_train = training_set[training_set['sensitive'] == 1].drop(['year','sensitive'],axis = 1)
    S0_train = training_set[training_set['sensitive'] == 0].drop(['year','sensitive'],axis = 1)
    
    S1_data_train = qp.data.LabelledCollection(S1_train.drop('y',axis=1), S1_train.y)
    S0_data_train = qp.data.LabelledCollection(S0_train.drop('y', axis = 1), S0_train.y)
    
    # Step 2
    model_S1 = qp.method.aggregative.EMQ(LogisticRegression()).fit(S1_data_train)
    model_S0 = qp.method.aggregative.EMQ(LogisticRegression()).fit(S0_data_train)
    
    # Step 3
    S1_test = testing_set[testing_set['sensitive'] == 1].drop(['year','sensitive'], axis = 1)
    S0_test = testing_set[testing_set['sensitive'] == 0].drop(['year','sensitive'], axis = 1)
    
    S1_data_test = qp.data.LabelledCollection(S1_test.drop('y', axis = 1), S1_test.y)
    S0_data_test = qp.data.LabelledCollection(S0_test.drop('y', axis = 1), S0_test.y)
    
    # P'(Y| S = 1)
    p_y_given_S1 = model_S1.quantify(S1_data_test.instances)
    
    # P'(Y | S=0)
    p_y_given_S0 = model_S0.quantify(S0_data_test.instances)
    
    return p_y_given_S1, p_y_given_S0, S1_data_train.prevalence(), S0_data_train.prevalence()


def estimate_test_MLP(training_set, testing_set):
    '''
    This method shall return the estimated P'(Y|S)
    
    '''
    
    # Step 1
    S1_train = training_set[training_set['sensitive'] == 1].drop(['year','sensitive'],axis = 1)
    S0_train = training_set[training_set['sensitive'] == 0].drop(['year','sensitive'],axis = 1)
    
    S1_data_train = qp.data.LabelledCollection(S1_train.drop('y',axis=1), S1_train.y)
    S0_data_train = qp.data.LabelledCollection(S0_train.drop('y', axis = 1), S0_train.y)
    
    # Step 2
    model_S1 = qp.method.aggregative.EMQ(MLPClassifier()).fit(S1_data_train)
    model_S0 = qp.method.aggregative.EMQ(MLPClassifier()).fit(S0_data_train)
    
    # Step 3
    S1_test = testing_set[testing_set['sensitive'] == 1].drop(['year','sensitive'], axis = 1)
    S0_test = testing_set[testing_set['sensitive'] == 0].drop(['year','sensitive'], axis = 1)
    
    S1_data_test = qp.data.LabelledCollection(S1_test.drop('y', axis = 1), S1_test.y)
    S0_data_test = qp.data.LabelledCollection(S0_test.drop('y', axis = 1), S0_test.y)
    
    # P'(Y| S = 1)
    p_y_given_S1 = model_S1.quantify(S1_data_test.instances)
    
    # P'(Y | S=0)
    p_y_given_S0 = model_S0.quantify(S0_data_test.instances)
    
    return p_y_given_S1, p_y_given_S0, S1_data_train.prevalence(), S0_data_train.prevalence()


def estimate_test_svc(training_set, testing_set):
    '''
    This method shall return the estimated P'(Y|S)
    
    '''
    
    # Step 1
    S1_train = training_set[training_set['sensitive'] == 1].drop(['year','sensitive'],axis = 1)
    S0_train = training_set[training_set['sensitive'] == 0].drop(['year','sensitive'],axis = 1)
    
    S1_data_train = qp.data.LabelledCollection(S1_train.drop('y',axis=1), S1_train.y)
    S0_data_train = qp.data.LabelledCollection(S0_train.drop('y', axis = 1), S0_train.y)
    
    # Step 2
    model_S1 = qp.method.aggregative.EMQ(LinearSVC()).fit(S1_data_train)
    model_S0 = qp.method.aggregative.EMQ(LinearSVC()).fit(S0_data_train)
    
    # Step 3
    S1_test = testing_set[testing_set['sensitive'] == 1].drop(['year','sensitive'], axis = 1)
    S0_test = testing_set[testing_set['sensitive'] == 0].drop(['year','sensitive'], axis = 1)
    
    S1_data_test = qp.data.LabelledCollection(S1_test.drop('y', axis = 1), S1_test.y)
    S0_data_test = qp.data.LabelledCollection(S0_test.drop('y', axis = 1), S0_test.y)
    
    # P'(Y| S = 1)
    p_y_given_S1 = model_S1.quantify(S1_data_test.instances)
    
    # P'(Y | S=0)
    p_y_given_S0 = model_S0.quantify(S0_data_test.instances)
    
    return p_y_given_S1, p_y_given_S0, S1_data_train.prevalence(), S0_data_train.prevalence()


def estimate_test_custom(training_set, testing_set):
    '''
    This method shall return the estimated P'(Y|S)
    
    '''
    
    # Step 1
    S1_train = training_set[training_set['sensitive'] == 1].drop(['year','sensitive'],axis = 1)
    S0_train = training_set[training_set['sensitive'] == 0].drop(['year','sensitive'],axis = 1)
    
    S1_data_train = qp.data.LabelledCollection(S1_train.drop('y',axis=1), S1_train.y)
    S0_data_train = qp.data.LabelledCollection(S0_train.drop('y', axis = 1), S0_train.y)
    
    # Step 2
    model_S1 = qp.method.aggregative.PCC(LogisticRegression()).fit(S1_data_train)
    model_S0 = qp.method.aggregative.PCC(LogisticRegression()).fit(S0_data_train)
    
    # Step 3
    S1_test = testing_set[testing_set['sensitive'] == 1].drop(['year','sensitive'], axis = 1)
    S0_test = testing_set[testing_set['sensitive'] == 0].drop(['year','sensitive'], axis = 1)
    
    S1_data_test = qp.data.LabelledCollection(S1_test.drop('y', axis = 1), S1_test.y)
    S0_data_test = qp.data.LabelledCollection(S0_test.drop('y', axis = 1), S0_test.y)
    
    # P'(Y| S = 1)
    p_y_given_S1 = model_S1.quantify(S1_data_test.instances)
    
    # P'(Y | S=0)
    p_y_given_S0 = model_S0.quantify(S0_data_test.instances)
    
    return p_y_given_S1, p_y_given_S0, S1_data_train.prevalence(), S0_data_train.prevalence()


def label_overtime(df):
    
    y_1s = []
    y_0s = []
    years = sorted(df.year.unique())
    for i in years:
        temp = df[df['year'] == i]
        rates = temp['y'].value_counts(normalize = True)
        y_0s.append(rates[0])
        y_1s.append(rates[1])
        
    plt.plot(years, y_0s, label = 'Y=0', marker='o')
    plt.plot(years, y_1s, label = 'Y=1', marker='o')
    plt.xticks(rotation = 45)
    plt.ylabel('P(Y)')
    plt.legend()
    
    
def sensitive_overtime(df):
    
    s_1s = []
    s_0s = []
    years = sorted(df.year.unique())
    for i in years:
        temp = df[df['year'] == i]
        rates = temp['sensitive'].value_counts(normalize = True)
        s_0s.append(rates[0])
        s_1s.append(rates[1])
        
    plt.plot(years, s_0s, label = 'S=0', marker='o')
    plt.plot(years, s_1s, label = 'S=1', marker='o')
    plt.xticks(rotation = 45)
    plt.ylabel('P(S)')
    plt.legend()
    
    
def condintional_overtime(df):
    
    # P(Y | S)
    y1_given_s1 = []
    y1_given_s0 = []
    years = sorted(df.year.unique())
    for i in years:
        temp = df[df['year'] == i]
        rates = utils.spurious_correlation(temp)
        y1_given_s0.append(rates[1])
        y1_given_s1.append(rates[0])
        
    plt.plot(years, y1_given_s0, label = 'P(Y=1 | S=0)', marker='o')
    plt.plot(years, y1_given_s1, label = 'P(Y=1 | S=1)', marker='o')
    plt.xticks(rotation = 45)
    plt.ylabel('P(Y|S)')
    plt.legend()
    
    
def joint_overtime(df):
    
    # P(Y , S)
    y1_and_s1 = []
    y1_and_s0 = []
    years = sorted(df.year.unique())
    for i in years:
        temp = df[df['year'] == i]
        rates = utils.get_joint(temp)
        y1_and_s0.append(rates[1][0])
        y1_and_s1.append(rates[1][1])
        
    plt.plot(years, y1_and_s0, label = 'P(Y=1 , S=0)', marker='o')
    plt.plot(years, y1_and_s1, label = 'P(Y=1 , S=1)', marker='o')
    plt.xticks(rotation = 45)
    plt.ylabel('P(Y,S)')
    plt.legend()
    
    
# def generate_synthetic_dataset(data, rates):
    
#     new_data_frame = []
    
#     years = sorted(data.year.unique())
    
    
#     for i in range(3, len(years)):
#         df = data[data['year'] == years[i]]
        
#         new_df = helper.generate_sample_joint_shift(df, rates[i], df.shape[0])
        
#         new_data_frame.append(new_df)
        
        
#     return pd.concat([i for i in new_data_frame])


def generate_synthetic_dataset(data, rates):
    
    new_data_frame = []
    years = sorted(data.year.unique())
    new_data_frame.append(data[data['year'] == years[0]])
    new_data_frame.append(data[data['year'] == years[1]])
    new_data_frame.append(data[data['year'] == years[2]])
    
    years = years[3:16]
                            
    for i in range(len(years)):
        df = data[data['year'] == years[i]]
        
        new_df = helper.generate_sample_joint_shift(df, rates[i], df.shape[0])
        
        new_data_frame.append(new_df)
        
        
    return pd.concat([i for i in new_data_frame])


def euclidean_distance(p, q):
    dist = np.linalg.norm(p-q)
    return dist


def DTO(full_results, fairness):
    
    temp = full_results.copy()
    accuracy = temp['auc'].values
    fair = 1 - temp[fairness].values
    
    results = []
    # 1. Find the maximum accuracy 
    accurate = np.max(accuracy)
    
    # 2. find the maximum fair 
    fairer = np.max(fair)
    
    # 3. normlaize both accuracy and bias 
    norm_accuracy = accuracy / accurate
    norm_fair = fair / fairer
    
    # 4. Euclidan distance from (1,1)
    
    p = np.array([1,1])
    
    points = list(zip(norm_accuracy, norm_fair))
    
    for i in points:
        results.append(euclidean_distance(p,i))
    
    
    temp[f'dto-{fairness}'] = results
    return temp


def create_data_frame_results(results):

    results_df = ({
            'Methods': ['vanilla', 'fairness', 'da', 'da_fairness', 'almuzaini','da_fairness_correction', 
                        'ours_uniform', 'ours_acc', 'ours_spd'],
            'auc': [results['vanilla'][0]['auc'].mean(),
                   results['fairness'][0]['auc'].mean(),
                   results['da'][0]['auc'].mean(),
                    results['da_fairness'][0]['auc'].mean(),
                   results['almuzaini'][0]['auc'].mean(),
                   results['da_fairness_correction'][0]['auc'].mean(),
                   results['ours_uniform'][0]['auc'].mean(),
                   results['ours_acc'][0]['auc'].mean(),
                    results['ours_spd'][0]['auc'].mean()
                   ],
        
            'auc_std': [results['vanilla'][1]['auc'].mean(),
                   results['fairness'][1]['auc'].mean(),
                   results['da'][1]['auc'].mean(),
                    results['da_fairness'][1]['auc'].mean(),
                   results['almuzaini'][1]['auc'].mean(),
                    results['da_fairness_correction'][1]['auc'].mean(),
                   results['ours_uniform'][1]['auc'].mean(),
                   results['ours_acc'][1]['auc'].mean(),
                    results['ours_spd'][1]['auc'].mean()
                   ],

            'spd': [results['vanilla'][0]['spd'].mean(),
                   results['fairness'][0]['spd'].mean(),
                   results['da'][0]['spd'].mean(),
                    results['da_fairness'][0]['spd'].mean(),
                   results['almuzaini'][0]['spd'].mean(),
                    results['da_fairness_correction'][0]['spd'].mean(),
                   results['ours_uniform'][0]['spd'].mean(),
                   results['ours_acc'][0]['spd'].mean(),
                    results['ours_spd'][0]['spd'].mean()
                   ],
        
            'spd_std': [results['vanilla'][1]['spd'].mean(),
                   results['fairness'][1]['spd'].mean(),
                   results['da'][1]['spd'].mean(),
                    results['da_fairness'][1]['spd'].mean(),
                   results['almuzaini'][1]['spd'].mean(),
                results['da_fairness_correction'][1]['spd'].mean(),
                   results['ours_uniform'][1]['spd'].mean(),
                   results['ours_acc'][1]['spd'].mean(),
                    results['ours_spd'][1]['spd'].mean()
                   ],

            'fprd': [results['vanilla'][0]['fprd'].mean(),
                   results['fairness'][0]['fprd'].mean(),
                   results['da'][0]['fprd'].mean(),
                    results['da_fairness'][0]['fprd'].mean(),
                   results['almuzaini'][0]['fprd'].mean(),
                    results['da_fairness_correction'][0]['fprd'].mean(),
                   results['ours_uniform'][0]['fprd'].mean(),
                   results['ours_acc'][0]['fprd'].mean(),
                    results['ours_spd'][0]['fprd'].mean()
                   ],
        
            'fprd_std': [results['vanilla'][1]['fprd'].mean(),
                   results['fairness'][1]['fprd'].mean(),
                   results['da'][1]['fprd'].mean(),
                    results['da_fairness'][1]['fprd'].mean(),
                   results['almuzaini'][1]['fprd'].mean(),
                    results['da_fairness_correction'][1]['fprd'].mean(),
                   results['ours_uniform'][1]['fprd'].mean(),
                   results['ours_acc'][1]['fprd'].mean(),
                    results['ours_spd'][1]['fprd'].mean()
                   ],

            'fnrd': [results['vanilla'][0]['fnrd'].mean(),
                   results['fairness'][0]['fnrd'].mean(),
                   results['da'][0]['fnrd'].mean(),
                    results['da_fairness'][0]['fnrd'].mean(),
                   results['almuzaini'][0]['fnrd'].mean(),
                    results['da_fairness_correction'][0]['fnrd'].mean(),
                   results['ours_uniform'][0]['fnrd'].mean(),
                   results['ours_acc'][0]['fnrd'].mean(),
                    results['ours_spd'][0]['fnrd'].mean()
                   ],
        
            'fnrd_std': [results['vanilla'][1]['fnrd'].mean(),
                   results['fairness'][1]['fnrd'].mean(),
                   results['da'][1]['fnrd'].mean(),
                    results['da_fairness'][1]['fnrd'].mean(),
                   results['almuzaini'][1]['fnrd'].mean(),
                    results['da_fairness_correction'][1]['fnrd'].mean(),
                   results['ours_uniform'][1]['fnrd'].mean(),
                   results['ours_acc'][1]['fnrd'].mean(),
                    results['ours_spd'][1]['fnrd'].mean()
                   ],

             'delta_auc': [results['vanilla'][0]['delta_auc'].mean(),
                   results['fairness'][0]['delta_auc'].mean(),
                   results['da'][0]['delta_auc'].mean(),
                    results['da_fairness'][0]['delta_auc'].mean(),
                   results['almuzaini'][0]['delta_auc'].mean(),
                    results['da_fairness_correction'][0]['delta_auc'].mean(),
                   results['ours_uniform'][0]['delta_auc'].mean(),
                   results['ours_acc'][0]['delta_auc'].mean(),
                    results['ours_spd'][0]['delta_auc'].mean()
                   ],
        
            'delta_auc_std': [results['vanilla'][1]['delta_auc'].mean(),
                   results['fairness'][1]['delta_auc'].mean(),
                   results['da'][1]['delta_auc'].mean(),
                    results['da_fairness'][1]['delta_auc'].mean(),
                   results['almuzaini'][1]['delta_auc'].mean(),
                    results['da_fairness_correction'][1]['delta_auc'].mean(),
                   results['ours_uniform'][1]['delta_auc'].mean(),
                   results['ours_acc'][1]['delta_auc'].mean(),
                    results['ours_spd'][1]['delta_auc'].mean()
                   ],
            
        })
    
    return pd.DataFrame(results_df)
    
    
def missclassification_rate(y, pred, sensitive):
    df = pd.DataFrame()
    df['y'] = y
    df['pred'] = pred
    df['sensitive'] = sensitive
    
    
    df['error'] = np.where(df['y'] != df['pred'], 1, 0)
    
    E_FP = df.query('sensitive == 1 & y == 1')['error'].mean()
    E_DP = df.query('sensitive == 0 & y == 1')['error'].mean()
    E_FN = df.query('sensitive == 1 & y == 0')['error'].mean()
    E_DN = df.query('sensitive == 0 & y == 0')['error'].mean()
    
    
    return np.array([E_FP, E_DP, E_FN, E_DN])


def logistic_regression_tuning(X, y):

    model = LogisticRegression()
    
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['None','l1', 'l2', 'elasticnet']
    class_weights = ['None', 'balanced']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    grid = dict(solver=solvers,penalty=penalty, class_weight=class_weights, C=c_values)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_results = grid_search.fit(X,y)

    print("Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_))


def create_synthetic_data(data):
    
    years = sorted(data.year.unique())
    
    all_data = []
    
    
    
    second_rate = helper.get_spurios_correlation_rate()[0]
    third_rate = helper.get_spurios_correlation_rate()[1]
    forth_rate = helper.get_spurios_correlation_rate()[2]
    
    for i in years:
        
        if i == '2016-06':
            temp1 = data[data['year'] == i]
            new_temp1 = helper.generate_sample_joint_shift(temp1, third_rate, temp1.shape[0])
            all_data.append(new_temp1)
            
            
        if i == '2017-01':
            temp = data[data['year'] == i]
            new_temp = helper.generate_sample_joint_shift(temp, second_rate, temp.shape[0])
            all_data.append(new_temp)
            

            
        if i == '2017-06':
            temp2 = data[data['year'] == i]
            new_temp2 = helper.generate_sample_joint_shift(temp2, third_rate, temp2.shape[0])
            all_data.append(new_temp2)
        
    
        else:
            df = data[data['year'] == i]
            all_data.append(df)
            
        
    modified_df = pd.concat([i for i in all_data])
    
    
    return modified_df


def my_ensemble(classifiers_list, weights, X):
    probas_ = np.asarray([clf_model.predict_proba(X) for clf_model in classifiers_list])
    avg = np.average(probas_, axis=0, weights=weights)
    maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)
    return avg[:,1], maj


def moving_average(df, index):
    test_estimate = (get_joint(df[df['year']==index[0]]) + get_joint(df[df['year']==index[1]])) / 2
    
    return test_estimate