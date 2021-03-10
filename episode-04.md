# Episode 4: Feature Selection

_Learning Outcomes_
1. Identify the need for feature selection
2. Apply feature selection to choose optimal features
3. Define different methods for data preprocessing, and apply them to the given dataset.
    - _Internal Note_: Data Imputation, Data Normalization and Data Restructuring 


## General
Feature selection is the process of identifying and selecting a subset of input variables that are most relevant to the target variable. Perhaps the simplest case of feature selection is the case where there are numerical input variables and a numerical target for regression predictive modeling. This is because the strength of the relationship between each input variable and the target can be calculated, called correlation, and compared relative to each other. 

Feature selection can also be used in classification problems as well. Moreover, we sometimes prefer to apply feature selection in a totally unsupervised way, without taking into account the outputs (numerical or categorical). The importance of unsupervised feature selection is to avoid dependencies between features. For instance, suppose that among other features, are two features A and B, that are connected by the following formula: xA = 5*xB (the value of feature B equals five times the value of feature A). We say that the two features A and B are **linearly dependent**. The problem with that is that technically use the same information twice; we don't actually need both of them because this might have a negative effect in our model.

Concerning the importance of feature selection process, the most popular answer is that it assists in finding both patterns between features (like dependencies that we already mentioned) and how each one of them affects the total output. By isolating a subset of the most informative features we clarify which of them play the most significant role, so, in a way, we "whiten" the model. Quick reminder that many ML techniques are called Black-Box algorithms, especially neural networks, because it's practiacally impossible to identify the decisions that a neural network made at every layer; but even if we had enough patience and confidence to examine all different layers, we might still wouldn't know why these decisions were taken.

There is also a less popular view regarding the importance of feature selection, that is the **denoising** of data. To illustrate, suppose we have a dataset of houses and we want to predict their price based on their feaatures. Suppose that, among others, there is a feature specifying the color on the walls of the corresponding house. Logically speaking, the walls' color is totally unrelated with the price of the house, it's just noise. However, ML algorithms are designed in such a way to force detecting even a tiny correlation between the features and the output, which might hurt our model. Feature selection process might probably pop out the specific feature (noise). 

Generally, features can be distinguished in three main categories; noise features, those with little significance and those of high significance. Feature selection aims to identify only features of the third category. The boundaries between the three categories are more fuzzy rather than well-determined. Anyway, there is a wrong impression that whenever we pop out a feature, we lose information. It depends; sometimes we get rid of useless noise.

## Feature Selection - Regression
As we promised in the last episode, we'll begin our analysis with the Boston Houses dataset. So let's load the dataset in the same way we did and normalize data:

```python
# Packages
import pandas as pd
from sklearn.datasets import load_boston
import seaborn as sns 

# Loading boston houses
boston_houses = load_boston(return_X_y=False)

# as data frame
X = pd.DataFrame(boston_houses.data, columns=boston_houses.feature_names)
y = pd.DataFrame(boston_houses.target, columns = ['Av. Price'])
```

As you've already noticed, we have also imported `seaborn`, which is a widely used package for plotting purposes, especially for fitting, distribution and other statistical/ML stuff. Now, let’s first plot the distribution of the target variable. We will use the `distplot()` function from the `seaborn` library.

```python
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(y.values, bins=30)
plt.show()
```

<p align="center">
  <img width="842" height="595" src="images/medv_dist_e04.png">
</p>

We see that the target values are distributed almost normally with few outliers. Next, we create a **correlation matrix** that measures the **linear relationships between the variables**. The correlation matrix can be formed by using the `corr()` function from the `pandas.DataFrame` library. We will use the `heatmap()` function from the `seaborn` library to plot the correlation matrix.

```python
# Initial numpy array to dataframe - in order to include MEDV column
boston = pd.DataFrame(boston_houses.data, columns=boston_houses.feature_names)
boston['MEDV'] = boston_houses.target

# Creating correlation matrix
correlation_matrix = boston.corr().round(2)

# annot = True to print the values inside the square
plt.figure()
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()
```

<p align="center">
  <img width="842" height="595" src="images/corr_matrix_e_04.png">
</p>

The correlation coefficient ranges from -1 to 1. If the value is close to 1, it means that there is a strong positive correlation between the two variables. When it is close to -1, the variables have a strong negative correlation.

First of all, let's exclude linear dependencies. An important point in selecting features for a linear regression model is to check for multi-co-linearity. The features `RAD`, `TAX` have a correlation of 0.91. These feature pairs are strongly correlated to each other. We should not select both these features together for training the model. Same goes for the features `DIS` and `AGE` which have a correlation of -0.75.

So, we're going to pop out `TAX` and `DIS` features.

## Feature Selection - Classification

## References

<a id="1">[1]</a> 
Jason Brownlee (2020)
How to Perform Feature Selection for Regression Data
Machine Learning Mastery, [Link](https://machinelearningmastery.com/feature-selection-for-regression-data/)
