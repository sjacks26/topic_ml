"""
This file creates functions to build a model from a feature file created by GetFeatures.py, then evaluates the model using test data as specified in Config.py
"""

import Config
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


def main():

    if Config.platform == "BOTH":
        feature_file = Config.comb_feature_file
    elif Config.platform == "TW":
        feature_file = Config.tw_feature_file
    elif Config.platform == "FB":
        feature_file = Config.fb_feature_file

    def get_feature_matrix(file=feature_file):
        """
        This function reads in the feature file created in GetFeatures.py.
        It creates two pandas dataframe from that file.
         The first is a dataframe with many columns that contains all the features for each message.
         The second is a dataframe with binary columns for each label for each message.
        The returned objects are the features dataframe and the label dataframe.
        """
        dat1 = pd.read_csv(file, quotechar='"')
        dat1 = dat1.drop(['id'], axis=1)
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
        """
        This functions reads in the feature matrix and label matrix created by get_feature_matrix.
        It then creates a dataframe for each label. Each dataframe contains the entire feature matrix plus a single label column.
        The returned object is a dictionary, where there is a key for each label and the value for that key is the labeled feature matrix for that label.
        """
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
        """
        This function reads in the dictionary of labeled feature matrices made by build_labeled_matrices.
        First, it creates an empty dictionary that will contain the performance measures for each label.
        It cycles through each matrix in that dictionary.
         It builds a classifier (specified by Config.classifier).
         It then trains a model using that classifier and the labeled feature matrix. It uses the number of folds specified by Config.k_folds.
         It then builds a classification report with accuracy performance measures for that model.
         Finally, it adds the performance results to the performance results dictionary.
        The returned object is a dictionary with performance measures for each label.
        """
        performance_results = {}
        f1s = []
        confusion_matrices = {}
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
            #Confusion_matrix = confusion_matrix(label, predicted_label, labels = [True, False])
            Confusion_matrix = pd.crosstab(label, predicted_label, rownames=["True"], colnames=["Predicted"], margins=True)
            Confusion_matrix.columns = ["False", "True", "All"]
            Confusion_matrix.index = ["False", "True", "All"]
            confusion_matrices[m] = Confusion_matrix
            f1 = f1_score(label, predicted_label, average="binary")
            f1s.append(f1)
            performance_results[m] = performance
        mean_f1 = sum(f1s)/len(f1s)
        return performance_results, mean_f1, confusion_matrices

    models = build_models()

    def write_performance(models = models, feature_file=feature_file):
        """
        This function takes in the performance scores generated by build_models.
        It creates a text file, the name of which contains the name of the classifier algorithm and the time that the code is executed.
        It writes a number of lines in the text file with details about the data, classifier, and features used.
        Finally, it writes the performance measures for each label in the text file.
        """
        results_file = Config.output_dir + Config.classifier + '_' + str(Config.now) + '.txt'
        scores = models[0]
        mean_f1 = models[1]
        conf_matr = models[2]
        topics = sorted(scores)

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for cm in conf_matr:
            cf_cm = conf_matr[cm]
            tp += cf_cm.loc["True","True"]                  # First label is "True", second label is "Predicted"
            fp += cf_cm.loc["False", "True"]
            fn += cf_cm.loc["True", "False"]
            tn += cf_cm.loc["False", "False"]

        overall_precision = tp / (tp + fp)
        overall_recall = tp / (tp + fn)
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)

        if Config.pos_restriction:
            pos_tags = Config.pos_tags
        else:
            pos_tags = False

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
            t.write('\t' + 'POS tags: ' + str(pos_tags))
            t.write('\n')
            t.write('\t' + 'POS counts: ' + str(Config.pos_counts))
            t.write('\n')
            t.write('\t' + 'Exclude mentions from vocab: ' + str(Config.exclude_mentions))
            t.write('\n')
            t.write('\t' + 'Number of mentions as features: ' + str(Config.num_mention_features))
            t.write('\n')
            t.write('\t' + 'NER: ' + str(Config.use_ne_chunks))
            t.write('\n\n')
            t.write("Overall micro-F1: " + str(overall_f1))
            t.write('\n\n')
            for f in topics:
                t.write(f)
                t.write('\n')
                t.write(scores[f])
                t.write('\n')

    write_performance()

if __name__ == "__main__":
   main()
