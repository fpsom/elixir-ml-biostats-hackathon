# ELIXIR Machine Learning & Biostatistics lesson Hackathon
## Day #2 - Monday November 2nd



### Sign-In

Name / Location (Timezone) / GitHub (gh) username / Twitter (tw) handle

- Fotis Psomopoulos / Thessaloniki, Greece (UTC+3) / github handle: fpsom / twitter id: fopsom 
- Tom Lenaerts / Brussels, Belgium (UTC+1) / github handle: tlenaert / twitter id: tlenaert (will be dropping in at intervals due to other meetings/teaching)
- Bengt Sennblad / NBIS, Uppsala, Sweden /github handle: bsennblad / twitter id: - 
- Van Du Tran / Lausanne, Switzerland (UTC+1) / github handle: vanduttran
- Claudio Mirabello / Linköping, Sweden (UTC+1) / gh: clami66
- Veronica Codoni / Luxembourg, Luxembourg (UTC+1) / gh: vero1166
- Anmol Kiran / Blantyre, Malawi (UTC+2) / gh:codemeleon / twitter: AnmolKiran
- Krzysztof Poterlowicz/ Bradford, UK (UTC-1)/ gh:kpoterlowicz
- Katarzyna Kamieniecka/ Manchester, UK (UTC-1)/ gh:kpbioteam / tw:katemurat (Poterlowicz Lab)
- Shakuntala Baichoo / Mauritius, (UTC+4) /gh:ShakunBaichoo / tw:ShakunBaichoo
- Pedro Fernandes / Oeiras, PT (UTC-1) / gh:pfern /tw:pfern
- Khaled Jum'ah/ Zarqa / Jordan (UTC+2)/ github handle: khaledzuj(Poterlowicz Lab)
- Florian Huber / Amsterdam (UTC+1) / github handle: florian-huber / twitter id: me_datapoint (can only join at intervals today, mostly to follow the process)
- Alexandros Dimopoulos / Athens, Greece (UTC +3) / github handle: alex_dem
- Alireza Khanteymoori / Freiburg, Germany (UTC+1) / github handle: khanteymoori / twitter id: khanteymoori




### Logistics

- Zoom URL: https://us02web.zoom.us/j/81025149308?pwd=cDNWYkxlSjlSMTAycEpNRVhCaHY2UT09
- Lesson repository: https://github.com/fpsom/elixir-ml-biostats-hackathon



### Schedule

- 08:45 CET (zoom session starts)
- __09:00 CET 2020-11-02__: Start
- 09:15: Recap of Day 1
- 09:45: Review of the Lesson Design structure
- 10:30: Review of the concept maps
- 11:30: Breakout sessions for exercises developement
- 12:15: Report out & discussion
- 12:30: [Lunch] Break (individual)
- 13:30: Breakout sessions for exercises developement
- 15:30: Report out & discussion
- 16:00: Putting it all together
- 16:30: Wrap-up time and road ahead
- 17:00: End of hackathon 


### Relevant Links:

- Markdown cheatsheet: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
- The Carpentries Instructor Training curriculum: https://carpentries.github.io/instructor-training/



## Lesson Design structure (day #1 recap)


https://github.com/fpsom/elixir-ml-biostats-hackathon/blob/master/ML-Lesson-Design.md




## Learning Goals

1. basic understanding of ML pipelines and steps (both in theory and in practice)
    - being able to independently explore other ML methods
    - know where to look for more information, how to ask for help
    - when to apply each method and why
    - what to do and what not to do
2. know the measures for ML model evaluation
3. feature engineering, visualization (raw data / models)
4. execute the appropriate software packages/libraries to apply a ML pipeline to a given dataset
5. more confidence in their data (understand their data) and research output


## Episodes

### Episode 1: **Introduction to Machine Learning**

_Learning Outcomes_
1. Mention the components of machine learning (data, features and models)
2. Argue what the role of statistics is in AI/DL/ML :+1: :+1: :+1:
3. Explain the difference between supervised and unsupervised methods
    - _Internal Note_: Mention also other types such as Reinforcement learning, Deep learning & semi-supervised learning
4. State the different categories of ML techniques
    - list some ML techniques: Linear Regression, Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Naive Bayes, Decision Tree, Random Forest, K-means Clustering 
5. Explain the difference between classification and regression
6. Explain the difference between clustering and dimensionality reduction
7. Explain the difference between continuous and discrete space
8. Explain the difference between linear and non-linear methods
9. Explain the difference between structured vs. unstructured data
    - This should be part of 1st point :+1: 


We should be able to think of an exercise that addresses one or more of the LO. If there is no such exercise, we should re-assess the LO. :+1:

Have an over-arching theme / context: e.g. based on an RNA-Seq dataset, create a classification / clustering model?


### Episode 2: **Components and Evaluation of a Machine Learning Pipeline**

_Learning Outcomes_
1. Explain the different steps in an ML pipeline :+1: :+1: 
    - _Internal Note_: Pre-processing / Feature Selection / Training /Testing / Evaluation / Validation
2. Explain and critically argue about the criteria for evaluating an ML model
    - _Internal Note_: Referring to F-measure, accuracy, recall, specificity, sensitivity, silhouette, confusion matrix, etc
3. Evaluate and compare ML models for a particular dataset;  what could you do to improve your model ?
    - _Internal Note_: Referring to training, testing, validation/cross-validation, dangers of overfitting/underfitting
:+1: 

### Episode 3: **Optimizing a model**

_Learning Outcomes_ :+1: :+1: 
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

### Episode 4: **Selecting the features** :+1: 

_Learning Outcomes_
1. Identify the need for feature selection
2. Apply feature selection to choose optimal features
3. Define different methods for data preprocessing, and apply them to the given dataset.
    - _Internal Note_: Data Imputation, Data Normalisation and Data Restructuring 



### Episode 5: **Putting it all together**

_Learning Outcomes_
1. Visualize the results of an ML method, and provide an interpretation
2. Sketch an ML pipeline (strategy) by combining the appropriate software packages/libraries to new data :+1: 
    - _Internal Note_: Define method A/B in the final lesson
3. Identify alternative methods B, appropriate for a new research question / new data
    - _Internal Note_: method B is not taught in the course, to be performed using resources shared in course (stretch goal, as an optional aspect)
:+1: :+1: :+1: :+1: :+1: :+1: :+1:


## Key concepts

_to be filled in_


### Concept Maps

https://carpentries.github.io/instructor-training/05-memory/index.html

Agreed concepts maps here:
https://docs.google.com/presentation/d/1mMj6KtEHk56soVgSqf-Qa0UmyJUMIuWEZrz-r-fAt30/edit#slide=id.g9e3c699220_0_0


Mindmap
https://github.com/dformoso/machine-learning-mindmap


### Exercises


#### Group #1

Episode: ### Episode 2: **Components and Evaluation of a Machine Learning Pipeline**
Participants: Pedro, Anmol, Khaled, Gasper

[comment] Before Exercise #1 have:
- Illustrated example of building an ML model
- 

Exercise #1: Please, re-order the following steps in the correct way (addressing LO1)

Question: Reorder followings steps for Machine Learning model creation.
- Model performance evaluation using test datase based on Cross-validation, F-Measure, accuracy, recall, specificity, sensitivity etc
- Data QC and Preprocessing(Visualisation, Imputation,Transformation (categorical to numerical),Normalisation)
- Various model development using training sets
- Training and testing data separation
- Model optimisations
- Acquiring a dataset
- Model deployment for real world data




[comment] Before Exercise #2 have:
- 

Exercise #2: If you have generated 3 models from your data, and you want to decide which model to use. #Calculate the confusion matrix for all models and pick the one with the better performance:

Model 1   Model 2    Model 3
50 | 50   60 | 68    40 | 30       TP | FN
50 | 50   32 | 40    70 | 60       FP | TN




[Hint ]

Confusion Matrix

|  | Yes   | No |
| -------- | -------- | -------- |
| Yes     | a    | b     |
| No     | c     | d     |


Offer multiple models, each with their confusion matrix and the accuracy score, and rank them from best to worst.




Exercise #3: How to improve the model 






#### Group #2

Episode: ### Episode 3: **Optimizing a model**
Participants: 
 - Wandrille Duchemin
 - Bengt Sennblad
 - Katarzyna Murat (Kamieniecka)
 - Alireza Khanteymoori
 

Exercises:

1. (LO1 + LO3?)
    -[comment] Before Exercise #1 have:
        - Tutorial on some ML methods for classification and for regression
    - Exercise
        - Data set: X continuous/categorical/ordinal
        - Y continuous. 
        - Question: predict Y. 
        - Model: e.g., SVM
        - Task: Why does this not work? Propose an alternative model. (or same thing linear model/ non-linear data).

2. (LO1--LO5 potentially)
    - Before exercise have done:
        - After most of the lecture so that the listed concepts have all been seen.
    - Exercise
        - Task: Connect problems/issues to the right solutions/concepts
            - regularization -- overfitting (bias/variance balance)
            - regression -- continous outcome
            - imbalanced data -- oversampling
            - inappropriate parameters for the model?
            - etc.
3. LO 4 + LO3 (+ LO2 somewhat)
    - Before exercise have done:
        - KNN tutorial (+ some other models, with hyperparameters that affect bias-variance balance)
    - Exercise
        - Task: Select one of the following models (KNN, etc) and indicate a parameters that can be used to address the balance between bias and variance (_k_, the neighborhood size) [as well as how : higher _k_ leads to higher variance or higher bias ?]

4. LO5 (possibly also LO2)
    - Before exercise have done:
        - pre-processing
        - methods to address imbalanced data
    - Exercise
        -  Data: Very imbalanced 
        -  Model: an appropriate model, but one that is sensistive to data imbalance (e.g.,regression)
        -  Question: appropriate for data and model
        -  show that result is bad
        -  task: Improve analysis [should use over or undersampling + possibly better hyperparameters]
5. LO2
    - Before exercise have done:
        - Exercise on searcing parameters
        - hands on Decision trees (possibly random forest????)
    - Exercise
        - Model: Decision tree
        - (maybe limit the available hyperparameters)
        - Task: 
            1. perform grid search optimization on a set of $n$ parameters. ($n$ = 3 or 4 ? a manageable but not too low number; possibly also limits on parameter values) + (explicit limits on some parameters)
            2. comment on the limit of grid-search optimization and potential alternatives.
            - potential methods:
                - exhaustive grid search 
                - random search (hill-climbing/MonteCarlo)
                - (Bayesian optimization)


#### Group #3

Episode 4: **Selecting the features**

Participants: 
- Claudio Mirabello
- Van Du Tran
- Shakuntala Baichoo


_Learning Outcomes_
1. Identify the need for feature selection
2. Apply feature selection to choose optimal features
3. Define different methods for data preprocessing, and apply them to the given dataset.
    - _Internal Note_: Data Imputation, Data Normalisation and Data Restructuring 



Before Exercise #1 have:
- Introduce/load the dataset that will be used throughout this episode
- Show the data in tables to do some exploration (that type of data? Is it complete? Are there NAs? Are they count? Are there categorical variables?)
    - Show the data visually through plotting (scatterplots, etc)
- Show the "outcome" column (labels), which value means sick/healthy? How many classes are there?
- Show if there is a class imbalance between outcomes
- Train a classifier on the dataset as it is

Exercise #1
- [outcome 3] Given the dataset shown before in exercise #1 (e.g. metagenomics) where:
    - Some features are not present (NA) in some of the samples
    - What to do for samples where some features are missing?
        - Try removing such samples/features altogether
        - Try imputing features by replacing NA with mean or median for that feature (dataset-wise)
        - Try using k-NN to find most similar samples and using features from those samples
        - How do these approaches compare? Which is best?

Before Exercise #2 have:
- Explain why some ML methods are sensitive to input/output scaling
- Code showing how different types of normalization/rescaling can be performed

Exercise #2
- [outcome 3] Given the same dataset where the features are counts (from 0 to a high integer):
    - Try to apply different methods for data scaling (Z-score, min-max) to the data
    - Try to visualize the data again in plots, what has changed? What looks the same?
    - Try to train again the predictor on the re-scaled data, does this improve the performance? Why do you think that is?

Before Exercise #3 have:
- Explain/refresh difference between categorical and numerical features
- Load clinical dataset (addition to previously introduced dataset)
- Show that the dataset have the same number of samples, but different types of features (categorical)
- Show how to convert categorical features to one-hot encoding or to perform other relevant types of encoding

[Outcome 3] Exercise #3
- Given an additional part of the dataset including clinical features:
    - Some other clinical features are present that are categorical (i.e. sex, smoker or not etc.)
    - How to include clinical features in the dataset?
        - Try encoding categorical variables in different ways (ordinal, one-hot)
        - Do clinical features help with the predction? 
        - Which encoding works best?

[comment] Before Exercise #4 have:
- Show how to calculate or plot/visualize pairwise correlation of features
- Show how to perform a PCA (or other kind of dimensionality reduction) plot features and/or samples to see how they group, give an idea of how many features might be informative in the end

- [Outcome 1 & 2] Exercise #4 (fleshed out)
    - Prepare a dataset where some features are highly correlated, others have very low variance. In general, a small subset of features should be enough to perform a classification task (e.g. a metagenomics dataset where the presence of a handful of bacteria species is enough to predict some kind of illness)
    - Test a classifier that has been covered beforehand (e.g. SVM etc) on the full dataset (no features removed), evaluate the performance (at this stage the performance should not be great)
    - Perform all-against-all correlation of features and plot them in appropriate ways, find out which features are very correlated
    - Remove a subset of features that have been found to be highly correlated
    - Test again the same classifier and see what kind of impact removing the features had on the performance
    - Select an even smaller set of features based on variance
    - Test again as previously done
    - Select an even smaller set of features based on feature importance as calculated by e.g. Random Forest
    - Test again the classifier on this set of features, find out which subset of features gives the best performance






### Road ahead


- Flesh out the material
	- each episode to an individual MD file
	- first full draft expected around Feb 2021 (to be published to Zenodo for a DOI)

- Decide on programming language (put :+1: to your choice)
	- Python :+1: :+1: :+1: :+1: :+1: :+1:   (6)
	- R :+1: :+1: :+1: :+1:   (4)
	- ???
	- Decision: go for python 3

- For people active in ELIXIR Nodes
	- Let me know if you are interested in hosting a workshop based on this content (earliest would be ~Feb 2021, in order to ensure a first complete draft of the lesson) [Pedro, March 2021]
	- [Tom; there may be an interest in Brussels. Need to check. how can we announce this?]
	- [Bengt: NBIS may want to use (parts of) the exercises in their ML biostatistics course]
	- [Alireza, June 2021]
	- [Poterlowicz's Lab; may want to use the exercises during the Bioinformatics course]

- For all participants in this hackathon; if interested to help deliver the course (as an Instructor), please enter your name below:
	- Shakun (but it depends on the timing)
	- Tom (also relative to timing) 
	- Kate (also relative to timing)
	- Khaled 
    - Anmol
    - Alireza
    - Wandrille 
    - Pedro
    - Van Du (depends on SIB and timing)



Link to the ELIXIR feedback survey about the hackathon: https://elixir.mf.uni-lj.si/mod/feedback/view.php?id=3007
The credentials for the e-learning platform that hosts the survey were emailed to you on October 13.

**ToDo**: Define the ideal profile of an Instructor for this course (re abilities, experiences, etc). Pairing Instructors to address this.

What need to be said before each exercise.
