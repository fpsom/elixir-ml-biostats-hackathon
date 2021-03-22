# Episode 3 - Exercises

## Exercise 1
This is a regression exercise. The dataset that we'll use is Medical Cost Personal Datasets which is publicly available with full details [here](https://www.kaggle.com/mirichoi0218/insurance). Each row corresponds to a single patient and consists of 7 columns: `age`, `sex`, `bmi` (Body mass index), `children` (Number of children covered by health insurance), `smoker`, `region` and `charges`(Individual medical costs). We've already stored this dataset in our Github repository as `insurance.csv` The question is whether we can predict insurance costs by utilizing the rest of the features.

### Theory: SVM for regression or SVR (Support Vector Regression)
Support Vector Regression (SVR) uses the same principle as SVM, but for regression problems. The problem of regression is to find a function that approximates mapping from an input domain to real numbers on the basis of a training sample. 

The idea behind SVR is shown on the following figure. Consider these two red lines as the decision boundary and the green line as the hyperplane (in regular SVM). Our objective, when we are moving on with SVR, is to basically consider the points that are within the decision boundary line. Our best fit line is the hyperplane that has a maximum number of points. But what is actually this decision boundary? Consider these lines as being at any distance, say `epsilon`, from the hyperplane. So, these are the lines that we draw at distance `+ epsilon` and `- epsilon` from the hyperplane, and that's basicallly the decision boundary.

<p align="center">
  <img width="973" height="569" src="exercises_images/svr_ex_3.png">
</p>

Evidently the SVR model can be generalized to non-linear curves (polynomials of second degree and above) and planes.

### Solution
First of all we import the dataset and distinguish X and y matrices:

```python
import pandas as pd

# Loading file
df = pd.read_csv('insurance.csv')
X = df.iloc[:,0:6]
y = df['charges']
y = pd.DataFrame(y, columns = ['charges'])
```

A general overview of the dataset is the following:

Features             |  Targets
:-------------------------:|:-------------------------:
![](exercises_images/insurance_dataset_X.png)  |  ![](exercises_images/insurance_dataset_y.png)

We can also extract useful information regarding the dataset, such as the class of features and whether there are null values, by using the `.info()` attribute function of `pandas.DataFrane` objects:

```python
df.info()
```

~~~
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object 
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object 
 5   region    1338 non-null   object 
 6   charges   1338 non-null   float64
dtypes: float64(2), int64(2), object(3)
memory usage: 73.3+ KB
~~~
