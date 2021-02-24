# Episode 3: Optimizing a model

_Learning Outcomes_ : 
1. Choose one or more appropriate ML methods (taught in the course) for the research question / dataset in question
2. For each appropriate ML method A, identify the optimal parameter set (hyperparameter optimization)
    - _Internal Note_: Also define what is hyperparameter optimization
3. Demonstrate the understanding of assumptions pertaining to the associated model of method A, its applicability to data
    - (including how to decide on the number of clusters)
4. Grasp the issue of bias and variance in ML and how ML design choices influence this balance
    - under/over-fitting. can be solved with regularization (GLM,LM), by tuning alogrithm parameters (tree depth, k in KNN)
    - dropout
    - cross-validation
5. Explain the problem of data imbalance and how to solve it properly
    - oversampling (when we have a few data)
    - undersampling (when we have enough data)

## Multi-class problem
In the first part of this lecture we are going to focus on a multi-class dataset and examine some more algorithms: t-sne (dimensionality reduction, visualization), kNN and Decision Trees (supervised problems), while in the second part we'll mainly focus on a regression problem. In the in-between stages we'll introduce more terminology so as to overcome various limitations that might occur. At first, we are going to load [Wine Recognition Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset) from `sklearn.Datasets` package. The wine dataset consists of 178 samples, 13 features each and three classes in total. The targets are [0,1,2] and correpsond to wine-type 1, type 2 and type 3 respectively. We can import and normalize our dataset using the following lines of code. By setting `return_X_y=True` and `as_frame=True` in `datasets.load_wine()` function we import the wine dataset directly in X-y matrix and `DataFrame` format. We are not goint to print the matrices in the console, so as to save some space.

```python
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

# loading wine dataset -storing directly into X,y dataframes
X,y = datasets.load_wine(return_X_y=True, as_frame=True)

# feature names
feature_names = X.columns

# Normalize
min_max_scaler = MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns=feature_names)
```
### Visualizationin
Following up the, more or less, standard routine established in the previous episodes, the first thing we do when loading our dataset is to try to get a sense of it, like how well classes are separated, what's the distribution of them, what's the variance of features and other aspects. This time we will attempt to apply **t-sne** algorithm, so as to reduce dimensions and project our samples into a 2D plot (more or less like we did in the last part of the first episode with PCA). We are not going to utilize the reduced-dimensionality features, however, in the rest of the analysis, because the new features created by t-sne do not have a physical meaning; they are just mathematical objects occured from the combination of real features. This part we'll only contribute to our better understanding of data.

*So what's t-sne?*
**T-Distributed Stochastic Neighbor Embedding (t-SNE)** is a non-linear technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets.

## Multi-class problem: kNN vs Decision Trees


