# Episode 1: Introduction to Machine Learning

## Learning outcomes
1. Mention the components of machine learning (data, features and models) #
2. Argue what the role of statistics is in AI/DL/ML
3. Explain the difference between supervised and unsupervised methods
    - _Internal Note_: Mention also other types such as Reinforcement learning, Deep learning & semi-supervised learning
4. State the different categories of ML techniques
    - list some ML techniques: Linear Regression, Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Naive Bayes, Decision Tree, Random Forest, K-means Clustering 
5. Explain the difference between classification and regression #
6. Explain the difference between clustering and dimensionality reduction
7. Explain the difference between continuous and discrete space #
8. Explain the difference between linear and non-linear methods
9. Explain the difference between structured vs. unstructured data

In general, machine learning (ML) is a category of algorithms that allows software applications to become more accurate in predicting outcomes without being explicitly programmed. The basic premise of machine learning is to build algorithms that can receive input data and use statistical analysis to predict an output, while updating outputs as new data becomes available. Let's take it step by step to explain what this actually means in practise.\
We are going to download the [Breast Cancer Wisconsin (Diagnostic) Data Set](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29) from the [UCI Machine Learning repository](http://archive.ics.uci.edu/ml/index.php) to see how it looks like. You need to switch to your working directory and open up a new Python 3 Jupyter Notebook. The first thing we need to do is to import the [dataset](https://pypi.org/project/dataset/) toolkit in our code, which will help us read datasets from online databases.

~~~
import dataset
~~~
{: .language-python}

In fact, we only need a single function from the toolkit, so it's better to use the following code instead.

~~~
from dataset import Dataset
~~~
{: .language-python}

In this way, we only import the `Dataset()` function from `dataset` toolkit. In order to download our data, we use the following lines of code. - Note that if we don't specify which function to load from `dataset` toolkit, we need to call the `Dataset()` function in this way: `dataset.Dataset()` -

~~~
# Downloading data file
URL_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
breastCancerData = Dataset(URL_data, delimiter=',', header=None)
~~~
{: .language-python}

In the first line of our code we state the [URL](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data) from which the data will be dowloaded. Basically, if you attempt open up the specified URL in your browser, you will notice that data is separated by comma `,` character. So, in the second line, we use the `Dataset()` function to download data. Apart from URL argument, `Dataset()` function takes two more arguements (basically, it may take more than those, you can check its documentation [here](https://dataset.readthedocs.io/en/latest/)). The first one is the `delimiter` argument, which specifies the character that separates the data. The second argument specifies whether the first row should be read as headers. Actually, this dataset contains no headers, so we pass the `None` value and we're going to dowload the corresponding headers from a different URL.\
The output of the `Dataset()` function is an object of class `Dataset`. In order to check this up, we need to use the `type()` command:

~~~
type(breastCancerData)
~~~
{: .language-python}

~~~
dataset.dataset.Dataset
~~~
{: .output}

Our data table is stored inside `.features` attribute of `breastCancerData` object. In order to store the data table at a different variable, we need to use the following line of code.

~~~
# assigning .features attribute to separate variable
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
# Printing data table
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

If all goes well, we can see that our dataset contains 569 observations across 32 variables. As it was previously mentioned, this data table doesn't contain any header at all, and the default headers are `x0, x1 ... x31`. We need to download the headers from a different URL. We use the following commands for this purpose.

~~~
# Downloading headers
URL_colnames = "https://raw.githubusercontent.com/fpsom/IntroToMachineLearning/gh-pages/data/wdbc.colnames.csv"
breastCancerDataColNames = Dataset(URL_colnames, delimiter = '\n', header = None)
col_names = breastCancerDataColNames.features.iloc[:,0].tolist()
~~~
{: .language-python}

The `.iloc()` attribute-function of a `DataFrame` object locates a specified area inside a data matrix. For more details, you can check out the documentation [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html). In order to store the elements of this area to a list, we need to use the `.tolist()` function. Now, the `col_names` list contains the headers of our data, and we use the following command for them to be assigned on the table. By typing `print(data.head())` we print only the first rows of our data table in our console.

~~~
# Specifying data column names
data.columns = col_names

# Printing data table
print(data.head())
~~~
{: .language-python}

~~~
  Diagnosis  Radius.Mean  Texture.Mean  Perimeter.Mean  Area.Mean  \
0         M        17.99         10.38          122.80     1001.0   
1         M        20.57         17.77          132.90     1326.0   
2         M        19.69         21.25          130.00     1203.0   
3         M        11.42         20.38           77.58      386.1   
4         M        20.29         14.34          135.10     1297.0   

   Smoothness.Mean  Compactness.Mean  Concavity.Mean  Concave.Points.Mean  \
0          0.11840           0.27760          0.3001              0.14710   
1          0.08474           0.07864          0.0869              0.07017   
2          0.10960           0.15990          0.1974              0.12790   
3          0.14250           0.28390          0.2414              0.10520   
4          0.10030           0.13280          0.1980              0.10430   

   Symmetry.Mean  ...  Radius.Worst  Texture.Worst  Perimeter.Worst  \
0         0.2419  ...         25.38          17.33           184.60   
1         0.1812  ...         24.99          23.41           158.80   
2         0.2069  ...         23.57          25.53           152.50   
3         0.2597  ...         14.91          26.50            98.87   
4         0.1809  ...         22.54          16.67           152.20   

   Area.Worst  Smoothness.Worst  Compactness.Worst  Concavity.Worst  \
0      2019.0            0.1622             0.6656           0.7119   
1      1956.0            0.1238             0.1866           0.2416   
2      1709.0            0.1444             0.4245           0.4504   
3       567.7            0.2098             0.8663           0.6869   
4      1575.0            0.1374             0.2050           0.4000   

   Concave.Points.Worst  Symmetry.Worst  Fractal.Dimension.Worst  
0                0.2654          0.4601                  0.11890  
1                0.1860          0.2750                  0.08902  
2                0.2430          0.3613                  0.08758  
3                0.2575          0.6638                  0.17300  
4                0.1625          0.2364                  0.07678  

[5 rows x 31 columns]
~~~
{: .output}

Basically, this dataset will be widely used in this tutorial and, thus, we are going to save it in our working direcoty at a CSV (Comma Separated Value) format, using the following command.

~~~
# storing data frame to csv file
data.to_csv('breast_cancer_data.csv')
~~~
{: .language-python}

Now, let's analyze our data set. The Breast Cancer data set is a real-valued multivariate data that consists of 569 samples (patients) that are classified into two classes, where each class signifies whether a patient is diagnosed with breast cancer or not. The two categories are malignant (M) and benign (B). Each row of the data matrix refers to a single patient. while the columns of each row are the characteristics (features) of our patient. The `Diagnosis` column is the one that indicates the nature of tumor, where M stands for malignant and B stands for benign.\
We will first remove the `ID` column, which is the unique identifier of each row. The reason for this is that the ID of each sample is given randomly during the sampling process (in most cases even prior to the analyis) and does not determine anything about the characteristics of the tumor - More or less like the name of a person -. Then we will separate the `Diagnosis` column from the main matrix and rename them as `X` and `y` tables respectively.

~~~
# Removing the first column
data = data.iloc[:,1:]

# Separating Diagnosis column
tumors = data.pop('Diagnosis')

# Renaming
X, y = data, tumors
~~~
{: .language-python}

In machine learning, we often refer to the `X` table as the samples/features table and to the `y` table as the labels/outputs. So, the reason for the renaming is more or less symbolic, to follow the conventional ML terminology. \
At this stage we should define the two main categories of ML problems: supervised and unsupervised. Generally speaking, supervised problems are those that include `y` table and unsupervised are those that do not. **Unsupervised learning** (which will be addressed later in depth), is the machine learning task of uncovering hidden patterns and structures from unlabeled data. For example, a researcher might want to group their samples into distinct groups, based on their gene expression data without in advance what these categories maybe. This is known as clustering, one branch of unsupervised learning. Another example would be, in our case, to have only information about the characteristics of the tumors (`X` table), but no information about their nature (`y` table), so we would have to detect patterns in our data by ourselves and attempt to cluster them into groups. \
On the other hand, **Supervised learning**  is the branch of machine learning that involves predicting discrete labels or continuous values given samples as inputs. Supervised learning problems are generally divided into two groups: Regression and Classification problems. Our problem here focuses on finding patterns to distinguish malignant from benign tumors and, moreover, on the assessment of a tumor based on its features and patterns detected. In other words, we want to classify tumors into groups and, thus, we are talking about a **classification problem**.
The second wide category of problems are those trying to predict continuous output values and are called Regression problems. In order to have a look at them, we're going to load the Boston house-prices dataset from [scikit-learn](https://scikit-learn.org/stable/) package in Python. `Scikit-learn` package is a set of simple and efficient tools or predictive data analysis, implemented in Python; this package is widely used in Machine Learning applications and, thus, we will find it really useful throughout this tutorial. We are using the following lines of code.

~~~
# Packages
from sklearn.datasets import load_boston
import pandas as pd

# Loading boston-houses package
boston_houses = load_boston()

# Converting to data frame
boston_houses_df = pd.DataFrame(boston_houses.data, columns=boston_houses.feature_names)
prices_df = pd.DataFrame(boston_houses.target, columns = ['Av. Price'])
~~~
{: .language-python}

Probably too many questions so let's analyze the code. In the first line of code we import the `load_boston()` function, which lies inside `datasets` subpackage, which in turn belongs to the `scikit-learn` main package. Apart from `scikit-learn` package, we also import [pandas](https://pandas.pydata.org/), which will be discussed in a bit. After that, we call the function and store its result to the `boston_houses` variable. The function returns an object of class `sklearn.utils.Bunch`; however we would prefer our data to be stored in a `DataFrame` object, that contains a bunch of functionalities. For this reason, we import `pandas` package as `pd`, meaning that whenever we want to return an attribute function of the package, we can call it by just typing `pd.function()` instead of `pandas.function()`. Pandas is another widely used library in python that contains many useful functionalitites to handle `DataFrames`. In the last line we use the `Dataframe()` function of `pandas` package to transform our data into `DataFrame` format. Targets in this example are stored in `prices_df` object as well. The input data looks like this.

~~~
print(boston_houses_df.head())
~~~
{: .language-python}

~~~
      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \
0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   
1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   
2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   
3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   
4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   

   PTRATIO       B  LSTAT  
0     15.3  396.90   4.98  
1     17.8  396.90   9.14  
2     17.8  392.83   4.03  
3     18.7  394.63   2.94  
4     18.7  396.90   5.33  

[506 rows x 13 columns]

~~~
{: .output}

Where as the targets data.

~~~
print(prices_df.head())
~~~
{: .language-python}

~~~
   Av. Price
0       24.0
1       21.6
2       34.7
3       33.4
4       36.2

[506 rows x 13 columns]

~~~
{: .output}

Full details concerning this dataset could be found [here](https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-house-prices-dataset). Actually, we are refering to this specific dataset to mention that the target values might certainly take continuous values, like the average price of the house in thousand dollars. These are called **Regression Problems** and our goal is to define a set of rules to connect inputs with outputs; in other words, we attempt to define an optimized function, such that given a house with specific features to predict its expected price. \
Apparently,  this discretion - continuity property can be passed into our feature space, meaning that the values of each feature could be either a continuous or a discrete value. For instance, a parient in our data could be either a smoker or non-smoker, hence this attribute is a boolean one. The size of his/her tumor, thouhg, is definetely a feature that can take any value inside a continuous interval. Keep in mind, however, that the discretion - continuity of feature space does not denote anything about the category of our problem; this can be only implied by the target values.\
In the rest of this first episode, we will introduce some basic machine learning terminology, in order to examine it more emphatically in the following episodes. The first significant emphasize is the difference betewwn machine learning algorithms and models. When we talk about an **algorithm** in machine learning, we mean a procedure that is run on data to create a machine learning model. So actually, the **model** is the output of the function, in other words the set of rules/parameters that link input with output data. In classification problems, a model could be function that takes as inputs feature vectors (i.e. vectors of the same dimensionality as the number of columns of input data matrix) and these vectors are classified into the corresponding groups based on whether the output of the functon is greater of lesser than zero. In regression problems, on the other hand, the output of the function is the predicted value, in our case, the average price of the house. \
Evidently, to define this optimized model that fits well in our data, we first need to apply a machine learning algorithm. In fact, there are many machine learning algorithms. Some of them are:
- Linear Regression
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machines
- Naive Bayes 
- Decision Tree
- Random Forest 
- K-means

There is no need to deepen more into them at this point, as we will analyze most of them in the following episodes.


