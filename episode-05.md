# Episode 5: Putting it all together

_Learning Outcomes_
1. Visualize the results of an ML method, and provide an interpretation
2. Sketch an ML pipeline (strategy) by combining the appropriate software packages/libraries to new data: 
    - _Internal Note_: Define method A/B in the final lesson
3. Identify alternative methods B, appropriate for a new research question / new data
    - _Internal Note_: method B is not taught in the course, to be performed using resources shared in course (stretch goal, as an optional aspect)


The last episode of this machine learning course can be devided into two main parts. In the first part, we'll talk about clustering, as we haven't studied it systematically in the previous episodes. We'll explain hierarchical clustering method and we'll attempt to visualize some stuff for a better interpretation. In the second part, we're going to combine the individual steps of ML into a inegrated pipeline, by utilizing Python's tools. So, let's jump directly into clustering stuff.

## Clustering
As we already mentioned, cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups. Cluster analysis itself is not one specific algorithm, but the general task to be solved. It can be achieved by various algorithms that differ significantly in their understanding of what constitutes a cluster and how to efficiently find them. Popular notions of clusters include groups with small distances between cluster members, dense areas of the data space, intervals or particular statistical distributions.

Clustering is more or less like abstract art; it definitely depends on how you interpret the data. The appropriate clustering algorithm and parameter settings (including parameters such as the distance function to use, a density threshold, the number of expected clusters or the evaluation of them) depend on the individual data set and intended use of the results. Cluster analysis as such is not an automatic task, but an iterative process of knowledge discovery or interactive multi-objective optimization that involves trial and failure[[1]](#1). 

In order to stop constantly recycling the same theoretical explanations of clustering and get a more practical sense, first of all we need a dataset. We're going to use the Wine Types dataset from the third episode, but we're going to ignore the labels vector. Actually, we'll consider it as metadata information, such as the location that each bottle of wine was produced (let's say 0 for France, 1 for Italy and 2 for Spain). By the way, this is a very common trick that data analysts use; when dealing with a vector of metadata, you can either conduct the analysis supervisedly, by considering it as a targets vector and utilizing its information in the analysis part, or conduct the analysis the analysis unsupervisedly and, afterwards, attempt to correlate your results with metadata. At any case, the second scenario is always legitimate, because the information from metadata comes to confirm the patterns extracted from data itself. Hence, let's import the dataset:

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

Now, let's introduce some basic material about hierarchical clustering, a widely used technique that we're going to apply to our data.

### Hierarchical clustering
In data mining and statistics, **hierarchical clustering** (also called hierarchical cluster analysis or HCA) is a method of cluster analysis which seeks to build a hierarchy of clusters. Strategies for hierarchical clustering generally fall into two types:
- **Agglomerative**: This is a "bottom-up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
- **Divisive**: This is a "top-down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

In general, the merges and splits are determined in a greedy manner. In our example, we're going to focus on Agglomerative Clustering and, thus, we're intrested in the merging part. Merging is accomplished in several ways, which are called **linkage methods**. Some linkage methods are: single-linkage, complete-linkage, unweighted average linkage(UPGMA), Weighted average linkage(WPGMA) etc. The results of hierarchical clustering are usually presented in a dendrogram. Except for the special case of single-linkage, none of the algorithms can be guaranteed to find the optimum solution[[2]](#2). However, it depends on how you define "optimum". I mean, single-linkage clustering provides an optimum solution in terms of the distribution of data, but not necessarily the solution that achieves the highest correlation with metadata. Hence, there are plenty of options within the clustering procedure, like which approach to follow (Agglomerative or Divisive) and which linkage method and distance metric to use. Actually, there is no best option, it depends on data. Here, this "optimization" part is done by me, but genearally, you might have to search over all possible choices to detect which fits better on your data.

### Coding time
First of all, let's check for linear dependencies, based on correlation matrix:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Creating correlation matrix
correlation_matrix = X_normalized.corr(method='pearson').round(2)

# annot = True to print the values inside the square
plt.figure(figsize=(10,10))
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()
```

<p align="center">
  <img width="720" height="720" src="images/corr_matrix_e_05.png">
</p>

Features `total_phenols` and `flavanoids` seem to be highly dependent. So, let's drop out `total_phenols`. (I also tried to exclude `flavanoids` instead, but final results were not as desired. On the other hand, we talk about only one correlation out of 13 features in total, so we don't expect to make that difference if we finally keep it in the data set. Anyway.)

```python
# Exclude flavanoids
X_normalized.drop(columns=['total_phenols'], inplace=True)
```

Now, we are going to apply hierarchical clustering, using `AgglomerativeClustering()` function from `sklearn.cluster` library. Also, we'll evaluate clustering using the average silhouette score. The first argument passed to `AgglomerativeClustering()` function is `n_clusters`, where you specify in how many clusters you wish to separate your data, so the model keeps the most robust ones, based on linkage method. Evidently, linkage method is specified in `linkage` argument, where we specify that we'll apply complete linkage. The `affinity` argument determines the metric to calculate distances (here we have numerical normalized data, hence euclidean seems to be the best choice) and `compute_full_tree` argument determines whether to stop early the construction of the tree at `n_clusters`. This is useful to decrease computation time if the number of clusters is not small compared to the number of samples. So let's check the output.

```python
# Calculation of silhouette score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

number_of_clusters = list(range(2,8))
silhouette_scores = []

for num_of_cluster in number_of_clusters:
    cluster = AgglomerativeClustering(n_clusters=num_of_cluster, affinity='euclidean', linkage='complete', compute_full_tree = True)
    cluster.fit_predict(X_normalized)
    silhouette_scores.append(silhouette_score(X_normalized, cluster.labels_))

# Plotting
plt.figure(figsize=(8,6))
plt.plot(number_of_clusters, silhouette_scores)
plt.xlabel('Num of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score vs number of clusters')
plt.show()
```

<p align="center">
  <img width="576" height="432" src="images/silhouette_e_05.png">
</p>

Silhouette finds a peak at `n_clusters = 3`, which is the number of different labels we have (remember that we use the `y` vector as metadata and what we've done up to this point is totally unsupervised). So we need to check how correlated these clusters are with the information from metadata. We'll use a visualized approach for a better interpretation and then we'll calculate the correlation mathematically.

First of all, we need to plot the dendrogram to get a sense of its structure and next to define the clusters that data can be splitted to. We will use the `scipy` package to create the dendrograms for our dataset. More specifically, we'll use `dendrogram()` function from `scipy.cluster.hierarchy` library. This function takes the output of the `linkage()` function of the same library and plots the desired dendrogram. Evidently, we specify the argument `method = complete`, as we previously used the complete linkage in Agglomerative clustering. So, let's execute the following cell:

```python
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(20, 15))
plt.title("Wine types dendrogram")
ddata = shc.dendrogram(shc.linkage(X_normalized, method='complete'), labels=list(y.values))
plt.show()
```

<p align="center">
  <img width="1440" height="1080 " src="images/dendrogram_no_cutoff_e_05.png">
</p>

The different colors correspond to different clusters that occur from the default value of `color_threshold` argument in `dendrogram()` function. Basically, we haven't given any specific value to this argument, so it takes the default value, which is 0.7*{max value at y axis}. In our case, the max value at y axis is around 2, which means color threshold is around 1.4, so if we draw a horizontal line at `y = 1.4` and cut the main tree into subtrees based on this line, all the subtrees that occur correspond to clusters and are given different colors. In our case, silhouette metric finds a peak at `n_clusters = 3`, which means that we need to select the value of `color_threshold` argument properly, so as our dendrogramm to be splitted into three subtrees. Starting from the top and going down, if we `color_threshold = 1.9`, the tree is splitted into two subtrees; but if we set `color_threshold = 1.75` we achieve the desired number of three subtrees, and so we do. In the following code, apart from setting the `color_threshold`, we also draw a horizontal line to indicate how we split the tree into subtrees. Finally, we assign colors to the leaves of the trees, whoose labels (metadata) are in x axis. The colors are assigned properly, so as to match the colors of the corresponding clusters. Let's check the output:

```python
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(20, 20))
plt.title("Wine types")
ddata = shc.dendrogram(shc.linkage(X_normalized, method='complete'), labels=list(y.values), color_threshold=1.75)

# Assignment of colors to labels, selected properly to match the dendrogram colors.
label_colors = {'0': 'red', '1': 'orange', '2': 'green'}

ax = plt.gca()
ax.tick_params(axis='x', which='major', labelsize=20)
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    lbl.set_color(label_colors[lbl.get_text()])

plt.hlines(y = 1.75, xmin=0, xmax = 2000)
plt.show()
```

<p align="center">
  <img width="1440" height="1080 " src="images/dendrogram_cutoff_e_05.png">
</p>

Clearly, clusters and labels are highly correlated! For a better understanding of the correlation, we'll construct a generalized confusion matrix, using the following code. To explain further, each row corresponds to a cluster; we isolate all the elements of the corresponding cluster and check how they are distributed into the three different classes :)

```python
import numpy as np

# Initialization - zero matrix
conf_matrix = pd.DataFrame(np.zeros((3,3)), columns= ['Class 0', 'Class 1', 'Class 2'], index = ['Cluster 0','Cluster 1','Cluster 2'])

# Generalized confusion matrix
for i in range(3):
    for j in range(3):
        this_cluster_indices = [x for x in range(len(clusters)) if clusters[x] == i]
        conf_matrix.iloc[i,j] = list(y[this_cluster_indices].values).count(j)

print(conf_matrix)
```

~~~
           Class 0  Class 1  Class 2
Cluster 0     57.0      4.0      0.0
Cluster 1      2.0     63.0      0.0
Cluster 2      0.0      4.0     48.0
~~~



## References

<a id="1">[1]</a> 
https://en.wikipedia.org/wiki/Hierarchical_clustering

<a id="2">[2]</a> 
https://en.wikipedia.org/wiki/Hierarchical_clustering

<a id="3">[3]</a> 
Jason Brownlee (2020)
How to Perform Feature Selection for Regression Data
Machine Learning Mastery, [Link](https://machinelearningmastery.com/feature-selection-for-regression-data/)

