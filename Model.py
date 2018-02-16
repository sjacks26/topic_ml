"""
This file creates functions to build a model from a feature file created by GetFeatures.py, then evaluates the model using test data as specified in Config.py
"""

import Config
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from datetime import datetime


class Main(object):

    if Config.platform == "BOTH":
        feature_file = Config.comb_feature_file
    elif Config.platform == "TW":
        feature_file = Config.tw_feature_file
    elif Config.platform == "FB":
        feature_file = Config.fb_feature_file

    def get_feature_matrix(file=feature_file):
        dat1 = pd.read_csv(file, quotechar='"')
        feature_matrix = dat1
        labels = list(dat1.columns[-8:])
        label_matrix = pd.DataFrame()
        label_matrix['message'] = dat1['message']
        for l in labels:
            feature_matrix = feature_matrix.drop([l], axis=1)
            label_matrix[l] = dat1[l]
        return feature_matrix, label_matrix

    feature_matrix, label_matrix = get_feature_matrix()

    def build_labeled_matrices(feature_matrix = feature_matrix, label_matrix = label_matrix):
        matrices = {}
        features = feature_matrix
        labels = list(label_matrix.keys())[1:]

        for l in labels:
            df = features.copy()
            df[l] = label_matrix[l]
            matrices[l] = df
        return matrices

    labeled_feature_matrices = build_labeled_matrices()

    def build_models(matrices=labeled_feature_matrices):
        performance_results = {}
        for m in matrices:
            """
            Chop up each matrix: lop off message, pull off final column for label
            Train model on feature matrix
            Measure, save performance
            Next matrix
            """
            df = matrices[m]
            features = df.iloc[:,1:-1].copy()
            label = df.iloc[:,-1].copy()
            if Config.classifier == "SGD":
                classifier = SGDClassifier(max_iter=1000)
            elif Config.classifier == "LinearSVC":
                classifier = LinearSVC()
            elif Config.classifier == "MNB":
                classifier = MultinomialNB()
            elif Config.classifier == "BNB":
                classifier = BernoulliNB(binarize=None)
            elif Config.classifier == "logit":
                classifier = LogisticRegression()
            else:
                raise Exception("Invalid classifier in Config")

            predicted_label = cross_val_predict(classifier, features, label, cv=Config.k_folds)
            performance = classification_report(label, predicted_label)
            performance_results[m] = performance
        return performance_results

    models = build_models(labeled_feature_matrices)

    def write_performance(models = models, feature_file = feature_file):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_file = Config.output_dir + Config.classifier + '_' + str(now) + '.txt'
        scores = models
        topics = sorted(scores)

        with open(results_file, 'w') as t:
            t.write("Results generated: " + str(now))
            t.write('\n')
            t.write("Platform: " + Config.platform)
            t.write('\n')
            t.write("Classifier: " + Config.classifier)
            t.write('\n')
            t.write("Folds: " + str(Config.k_folds))
            t.write('\n')
            t.write("Features:" + str(feature_file))
            t.write('\n')
            t.write('\t' + 'Unigrams: ' + str(Config.num_unigrams))
            t.write('\n')
            t.write('\t' + 'Bigrams: ' + str(Config.num_bigrams))
            t.write('\n')
            t.write('\t' + 'POS tags: ' + str(Config.pos_restriction))
            t.write('\n')
            t.write('\t' + 'Mentions: ' + str(Config.use_mentions))
            t.write('\n\n')
            for f in topics:
                t.write(f)
                t.write('\n')
                t.write(scores[f])
                t.write('\n')

    write_performance()