# Machine Learning Lesson Design

## Target Audience

The main audience of this lesson is early career life-science researchers (i.e. older students and young researchers). Established PIs or younger students might find the content of this course either too introductory, or too advanced.

## Learner Profiles

### Person A

_to be fleshed out_

### Person B

_to be fleshed out_


## Required Pre-Knowledge:

In terms of background experience, they should have a basic knowledge of data analysis. In other words, they should have a good enough knowledge of how data are produced (ideally big data), but not sufficient knowledge in how to analyze them, nor insistutional support for ML / data analysis. This background is further interpreted as:
- In terms of programming fluency; they are at the level of "copy-paster". They can adapt recipes from literature, but they are not a proficient function writer / full on dev.
- In terms of statistical knowledge; they are novices. They have a working knowledge of what a p-value is, but they are not full statisticians.
- In terms of ML knowledge: they have no prior knowledge.

Specifically, before taking this tutorial learners should have basic knowledge of the following concepts:
- basic statistical and programming knowledge
  - have used "recipes" for stats analysis in published papers
  - have attended an undergraduate course
  - know how to run a script (in R/Python)
  - know how to load data (in R/Python)
  - know how to clean data wrangling (denoise, missing values)
  - know how to use "recipes" (in R/Python)
- awareness about data dredging: data fishing, data snooping, data butchery, and p-hacking
  - are aware of these issues - not a clear understanding
- no prior knowledge / experience in ML

## Pain points

- They don't have a clear understanding of the difference between statistics and machine learning (and where the overlap is)
- They don't know how ML works, and how to select the appropriate method for the data.
- They don't know how to do efficient pre-processing (e.g. creating the appropriate train/test sets).
- They don't know how ML can be tuned, and how different parameters can lead to completely different results. They need better insights on how ML tuning works.
- They work alone, they don't know where to begin, and they want to quickly have some useful skills for their own data.

## Learning Goals

1. basic understanding of ML pipelines and steps (both in theory and in practice)
    - being able to independently explore other ML methods
    - know where to look for more information, how to ask for help
    - when to apply each method and why
    - what to do and what not to do
2. know the measures for ML model evaluation
3. feature engineering, visualization (raw data / models)
4. more confidence in their data (understand their data) and research output


## Episodes

### Episode 1: **Introduction to Machine Learning**

_Learning Outcomes_
1. Mention the components of machine learning (data, features and models)
2. Argue what the role of statistics is in AI/DL/ML 
3. Explain the difference between supervised and unsupervised methods
    - _Internal Note_: Mention also other types such as Reinforcement learning, Deep learning
4. Be able to state the different categories of ML techniques
    - list some ML techniques: Linear Regression, Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Naive Bayes, Decision Tree, Random Forest, K-means Clustering (including how to decide on the number of clusters)
5. Explain the difference between classification and regression
6. Explain the difference between clustering and dimensionality reduction
7. Explain the difference between continuous and discrete space
8. Explain the difference between linear and non-linear methods
9. Explain the difference between structured vs. unstructured data


### Episode 2: **Components and Evaluation of a Machine Learning Pipeline**
1. Be able to explain the different steps in an ML pipeline
    - _Internal Note_: Pre-processing / Feature Selection / Training /Testing / Evaluation / Validation
2. Be able to list the criteria for evaluating an ML model
    - _Internal Note_: Referring to F-measure, accuracy, recall, specificity, sensitivity, silhouette, confusion matrix, etc
3. Be able to evaluate and compare ML models for a particular dataset;  what could you do to improve your model ?
    - _Internal Note_: Referring to training, testing, validation/cross-validation, dangers of overfitting/underfitting


### Episode 3: **Optimizing a model**

1. Understand assumptions pertaining to the associated model of method A, its applicability to data
2. Be able to choose one or more appropriate ML methods (taught in the course) for the research question / dataset in question
3. For each appropriate ML method, being able to identify the optimal parameter set (hyperparameter optimization)
    - _Internal Note_: Also define what is hyperpatameter optimization


### Episode 4: **Optimizing the features**

1. Be able to choose optimal features using feature selection
2. Be able to define different methods for data preprocessing, and apply them to the given dataset.
    - _Internal Note_: Data Imputation, Data Normalisation and Data Restructuring 



### Episode 5: **Putting it all together**

1. Be able to visualize the results of an ML method, and provide an interpretation
2. Execute the appropriate software packages/libraries to apply a given method to data;
3. Examine how to combine the appropriate software packages/libraries to automate the analysis
4. Apply method A to own data (having access to relevant code snippets/function/library)
    - _Internal Note_: Define method A/B in the final lesson
5. Be able to locate alternative methods B, not taught in the course (using resources shared in course), appropriate for research questions and data



