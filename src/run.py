from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import (precision_recall_curve,PrecisionRecallDisplay)
from scipy import stats 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# Random Seed for Reproducibility
SEED = 0
# The fraction of the data to be extracted to the test set
TEST_SIZE = 0.2
# Is Standardisation Necessary


'''
Extracts the features field and response field from the data
'''
def splitFeaturesReponse(df):
    features = df.loc[:, 1:12]
    response = df.loc[:, 29]
    return features, response

'''
Checks the dataset for:
    * missing Values
 '''
def checkData(df):
    # Check for missing values
    print("number of empty entries in columns")
    print(df.isnull().sum())
    # Determining how the data is distributed
    print("Describing Data")
    print(df.describe())
    df.loc[:, 1:12].boxplot()
    # saves the plot to a file
    plt.savefig('plots/all_column_boxplot.png')
    df.loc[:, 1:12].hist()
    plt.savefig('plots/all_column_hist.png')
    plt.close()
    print("Value Counts")
    print(df.loc[:,29].value_counts())
    df.loc[:, 29].hist()
    plt.savefig('plots/features_hist.png')

def preprocessTraining(df, checks=False):
    if checks:
        checkData(df)
    print("Total Number of Samples: " + str(len(df)))
    # Removing outliers in columns 2-13 inclusively
    for i in range(1,13):
        df = df[(np.abs(stats.zscore(df[i])) < 3)]
    
    print("Total non-outlier samples: " + str(len(df)))
    features, response = splitFeaturesReponse(df)
    # Encode the labels as numbers
    encoder = LabelEncoder()
    encoder.fit(response)
    scaler= RobustScaler()
    scaler.fit(features)
    pd.DataFrame(features).boxplot()
    poly = PolynomialFeatures(2, include_bias=False)
    print("Number of Feautres : " + str(len(features.columns)))
    return poly.fit_transform(scaler.transform(features)), encoder.transform(response), encoder, scaler, poly

def evaluateModel(name, model, x, y, curve = False):
    print("\nModel: " + name)
    y_pred = model.predict(x)
    print("\nClassification Accuracy")
    print(accuracy_score(y, y_pred))
    print("\nBalanced Accuracy")
    print(balanced_accuracy_score(y, y_pred))
    print("\nConfusion Matrix")
    print(confusion_matrix(y, y_pred))
    print("\nPrecision: micro")
    print(precision_score(y, y_pred, average='micro'))
    print("\nPrecision: macro")
    print(precision_score(y, y_pred, average='macro'))
    print("\nPrecison: samples")
    # One vs all Precison Score
    for i in range(0, 7):
        y_test = (y == i).astype(int)
        pred = (y_pred == i).astype(int)
        print("CL_" + str(i) + ": " + str(precision_score(y_test, pred, average="binary")))
    print("\nRecall: micro")
    print(recall_score(y, y_pred, average='micro'))
    print("\nRecal: macro")
    print(recall_score(y, y_pred, average='macro'))
    print("\nRecall: samples")
    # One vs All Recall Score
    for i in range(0, 7):
        y_test = (y == i).astype(int)
        pred = (y_pred == i).astype(int)
        print("CL_" + str(i) + ": " + str(precision_score(y_test, pred, average="binary")))

    # No intution from human eyes of this... to many features
    # print("\nWeight Matrix")
    # print(model.coef_)

    # Sum of Euclidean distance of class weights | l2 norm
    print("\nCLASS WEIGHTS")
    for i in range(0, 7):
        print("CL_" + str(i) + ": ")
        print("\tMagnitude: " + str(np.linalg.norm(model.coef_[i])))
        print("\tMaximum: " + str(np.amax(model.coef_[i])))
        print("\tMinimum: " + str(np.amin(model.coef_[i])))
        print("\tClosest to Zero: " + str(np.amin(np.abs(model.coef_[i]))))

    # Generates One vs ALl Precision-Recall Curves for all of the Classes
    if curve == True:
        for i in range (0, 7):
            y_test = (y == i).astype(int)
            pred = (y_pred == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_test, pred)
            PrecisionRecallDisplay(precision=precision, recall=recall, pos_label="CL_" + str(i)).plot(name="One vs All plot of CL_" + str(i))
            plt.legend(loc=0)
            plt.savefig("plots/pr_one_vs_all_curve_plot_CL_" + str(i) + ".png")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "-setup":
        setup()
    else:
        experiment()

def experiment():
    #data reading
    raw_data = pd.read_csv("data/drug_consumption.data", header=None)
    train, test = train_test_split(
        raw_data, test_size=TEST_SIZE, random_state=SEED)

    x, y, encoder, scaler, poly = preprocessTraining(train)

    # Train the models


    # penalty none
    # class_weight_none
    model_1 = LogisticRegression(
        penalty='none', random_state=SEED, class_weight=None, max_iter=10000)
    model_1.fit(x, y)

    # penalty none
    # class_weight balanced
    model_2 = LogisticRegression(
           penalty='none', random_state=SEED, class_weight='balanced', max_iter=10000)
    model_2.fit(x, y)

    # penalty l2
    # class_weight balanced
    model_3 = LogisticRegression(
        penalty='l2', random_state=SEED, class_weight='balanced', max_iter=10000)
    model_3.fit(x, y)

    # preprocess the test set
    x_test, y_test_labels = splitFeaturesReponse(test)
    y_test = encoder.transform(y_test_labels)
    x_test = scaler.transform(x_test)
    x_test = poly.transform(x_test)

    # Evaluate the performance of the models on testing data
    evaluateModel("Penalty='none', class_weight=None", model_1, x_test, y_test)
    evaluateModel("Penalty='none', class_weight=balanced", model_2, x_test, y_test)
    evaluateModel("Penalty='l2', class_weight=balanced", model_3, x_test, y_test, curve = True)

'''
 Used for initation testing of program with a validation set
 Also used to generate plots of distributions of 
'''
def setup():
    # PART 1: Data Preprocessing

    #data reading 
    raw_data = pd.read_csv("data/drug_consumption.data", header=None)
    # test set is discarded
    train, test = train_test_split(raw_data, test_size=TEST_SIZE, random_state=SEED)

    train, validation = train_test_split(train, test_size=TEST_SIZE, random_state=SEED, stratify=train.loc[:,29])

    x, y, encoder, scaler, poly = preprocessTraining(train, checks=True)
    
    # Training the Dataset
    model = LogisticRegression(penalty='none', random_state=SEED, class_weight=None, max_iter=10000)
    model.fit(x, y)

    x_test, y_test_labels = splitFeaturesReponse(validation)
    y_test = encoder.transform(y_test_labels)
    x_test = scaler.transform(x_test)
    x_test = poly.transform(x_test)
    # Standardized the validation set
    evaluateModel("Penalty='none', class_weight=None", model, x_test, y_test)

if __name__ == "__main__":
    main()