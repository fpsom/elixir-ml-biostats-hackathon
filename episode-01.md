# Episode 1: Introduction to Machine Learning

## Learning outcomes
1. Mention the components of machine learning (data, features and models)
2. Argue what the role of statistics is in AI/DL/ML
3. Explain the difference between supervised and unsupervised methods
    - _Internal Note_: Mention also other types such as Reinforcement learning, Deep learning & semi-supervised learning
4. State the different categories of ML techniques
    - list some ML techniques: Linear Regression, Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Naive Bayes, Decision Tree, Random Forest, K-means Clustering 
5. Explain the difference between classification and regression
6. Explain the difference between clustering and dimensionality reduction
7. Explain the difference between continuous and discrete space
8. Explain the difference between linear and non-linear methods
9. Explain the difference between structured vs. unstructured data

In general, machine learning (ML) is a category of algorithms that allows software applications to become more accurate in predicting outcomes without being explicitly programmed. The basic premise of machine learning is to build algorithms that can receive input data and use statistical analysis to predict an output, while updating outputs as new data becomes available. Let's take it step by step to explain what this actually means in practise.\
We are going to download the [Breast Cancer Wisconsin (Diagnostic) Data Set](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29) from the [UCI Machine Learning repository](http://archive.ics.uci.edu/ml/index.php) to see how it looks like. You need to switch to your working directory and open up a new Python 3 Jupyter Notebook. The first thing we need to do is to import the [dataset](https://pypi.org/project/dataset/) toolkit in our code, which will help us read datasets from online databases

~~~
import dataset
~~~
{: .language-python}

In fact, we only need a single function from the toolkit, so it's better to use the following code instead.

~~~
from dataset import Dataset
~~~
{: .language-python}

In this way, we only import the `Dataset()` function from `dataset` toolkit. In order to download our data, we use the following lines of code.

~~~
# Downloading data file
URL_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
breastCancerData = Dataset(URL_data, delimiter=',', header=None)
data = breastCancerData.features.copy()
~~~
{: .language-python}
