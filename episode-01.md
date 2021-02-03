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

In this way, we only import the `Dataset()` function from `dataset` toolkit. In order to download our data, we use the following lines of code. - Note that if we don't specify which function to import from `dataset toolkit`, we need to call the `Dataset()` function in this way: `dataset.Dataset()` -

~~~
# Downloading data file
URL_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
breastCancerData = Dataset(URL_data, delimiter=',', header=None)
~~~
{: .language-python}

In the first line of our code we state the [URL](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data) from which the data will be dowloaded. Basically, if you attempt open up the specified URL in your browser, you will notice that data is separated by comma `,` character. So, in the second line, we use the `Dataset()` function to download data. Apart from URL argument, `Dataset()` function takes two more arguements (in fact, it may take more, you can check its documentation [here](https://dataset.readthedocs.io/en/latest/)). The first one is the `delimiter` argument, which specifies the character that separates the data. The second argument specifies whether the first row should be read as headers. Actually, this dataset contains no headers, so we pass `None` value and we're going to dowload the corresponding headers from a different URL.\
The output of the `Dataset()` function is an object of class `Dataset`. In order to check this up, we need to use the `type()` command:

~~~
type(breastCancerData)
~~~
{: .language-python}

~~~
dataset.dataset.Dataset
~~~
{: .output}

Our data table is stored inside `features` attribute of `breastCancerData` object. In order to store the data table at a different variable, we need to use the following line of code.

~~~
data = breastCancerData.features
~~~
{: .language-python}

The new `data` object is a `DataFrame` object.

~~~
type(data)
~~~
{: .language-python}

~~~
pandas.core.frame.DataFrame
~~~
{: .output}

The `DataFrame` class is pretty useful in handling data tables in Python and is widely used for this purpose. Hence, we'll constantly refer to it as the course goes on. Now, let's have a look at our `data` object.

~~~
# Downloading data file
data
~~~
{: .language-python}

~~~
 	x0 	x1 	x2 	x3 	x4 	x5 	x6 	x7 	x8 	x9 	... 	x22 	x23 	x24 	x25 	x26 	x27 	x28 	x29 	x30 	x31
0 	842302.0 	M 	17.99 	10.38 	122.80 	1001.0 	0.11840 	0.27760 	0.30010 	0.14710 	... 	25.380 	17.33 	184.60 	2019.0 	0.16220 	0.66560 	0.7119 	0.2654 	0.4601 	0.11890
1 	842517.0 	M 	20.57 	17.77 	132.90 	1326.0 	0.08474 	0.07864 	0.08690 	0.07017 	... 	24.990 	23.41 	158.80 	1956.0 	0.12380 	0.18660 	0.2416 	0.1860 	0.2750 	0.08902
2 	84300903.0 	M 	19.69 	21.25 	130.00 	1203.0 	0.10960 	0.15990 	0.19740 	0.12790 	... 	23.570 	25.53 	152.50 	1709.0 	0.14440 	0.42450 	0.4504 	0.2430 	0.3613 	0.08758
3 	84348301.0 	M 	11.42 	20.38 	77.58 	386.1 	0.14250 	0.28390 	0.24140 	0.10520 	... 	14.910 	26.50 	98.87 	567.7 	0.20980 	0.86630 	0.6869 	0.2575 	0.6638 	0.17300
4 	84358402.0 	M 	20.29 	14.34 	135.10 	1297.0 	0.10030 	0.13280 	0.19800 	0.10430 	... 	22.540 	16.67 	152.20 	1575.0 	0.13740 	0.20500 	0.4000 	0.1625 	0.2364 	0.07678
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
564 	926424.0 	M 	21.56 	22.39 	142.00 	1479.0 	0.11100 	0.11590 	0.24390 	0.13890 	... 	25.450 	26.40 	166.10 	2027.0 	0.14100 	0.21130 	0.4107 	0.2216 	0.2060 	0.07115
565 	926682.0 	M 	20.13 	28.25 	131.20 	1261.0 	0.09780 	0.10340 	0.14400 	0.09791 	... 	23.690 	38.25 	155.00 	1731.0 	0.11660 	0.19220 	0.3215 	0.1628 	0.2572 	0.06637
566 	926954.0 	M 	16.60 	28.08 	108.30 	858.1 	0.08455 	0.10230 	0.09251 	0.05302 	... 	18.980 	34.12 	126.70 	1124.0 	0.11390 	0.30940 	0.3403 	0.1418 	0.2218 	0.07820
567 	927241.0 	M 	20.60 	29.33 	140.10 	1265.0 	0.11780 	0.27700 	0.35140 	0.15200 	... 	25.740 	39.42 	184.60 	1821.0 	0.16500 	0.86810 	0.9387 	0.2650 	0.4087 	0.12400
568 	92751.0 	B 	7.76 	24.54 	47.92 	181.0 	0.05263 	0.04362 	0.00000 	0.00000 	... 	9.456 	30.37 	59.16 	268.6 	0.08996 	0.06444 	0.0000 	0.0000 	0.2871 	0.07039

569 rows × 32 columns
~~~
{: .output}

As it was previously mentioned, this data table doesn't contain any header at all, so we need to download them from a different URL. We use the following commands for this purpose.
