# Load Libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Dataset

# fixed acidity
#
# most acids involved with wine or fixed or nonvolatile (do not evaporate readily)
# volatile acidity
#
# the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
# citric acid
#
# found in small quantities, citric acid can add 'freshness' and flavor to wines
# residual sugar
#
# the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1
# gram/liter and wines with greater than 45 grams/liter are considered sweet chlorides
#
# the amount of salt in the wine
# free sulfur dioxide
#
# the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion;
# it prevents microbial growth and the oxidation of wine total sulfur dioxide
#
# amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2
# concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine density
#
# the density of water is close to that of water depending on the percent alcohol and sugar content
# pH
#
# describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between
# 3-4 on the pH scale sulphates
#
# a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant
# alcohol
#
# the percent alcohol content of the wine
# quality
#
# output variable (based on sensory data, score between 0 and 10)
url = "./Iris.csv"
names = ['sepal-length-cm', 'sepal-width-cm', 'petal-length-cm', 'petal-length-cm', 'class']
dataset = pandas.read_csv(url, names=names)

# Dataset Summary
# print(dataset.shape)
print(dataset.head(20))

# Statistical Summary
# print(dataset.describe())

# Class Distribution
# print(dataset.groupby('class').size())

# Data Visualization
# Univariate

# Box & Whisker Plot
# dataset.plot(kind='box', subplots=True, layout=(6, 6), sharex=False, sharey=False)
# plt.show()

# Histogram
# dataset.hist()
# plt.show()

# Multivariate

# Scatter Plot Matrix
scatter_matrix(dataset)
plt.show()