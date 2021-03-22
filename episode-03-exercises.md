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

The point of this exercise is to apply a regression model to data (SVR), explain why it might not fit the data properly and apply suitable changes in order to increase the performance. For this reason, we're going to skip validation method, because it's meaningless to repeat the total ML pipeline at every exercise; it makes more sense to focus on specific parts each time. The data matrix consists of three categorical values: `sex`, `smoker` and `region` (maybe we could consider children as categorical feature too, because its values are limited, but it has already numerical values, we leave  it as it is). Let's print the distributions of our categorical features:

```python
print("----------------------------")
print("Feature: Smoker")
print(X['smoker'].value_counts())
print("----------------------------")
print("Feature: Sex")
print(X['sex'].value_counts())
print("----------------------------")
print("Feature: Region")
print(X['region'].value_counts())
print("----------------------------")
```

~~~
----------------------------
Feature: Smoker
no     1064
yes     274
Name: smoker, dtype: int64
----------------------------
Feature: Sex
male      676
female    662
Name: sex, dtype: int64
----------------------------
Feature: Region
southeast    364
southwest    325
northwest    325
northeast    324
Name: region, dtype: int64
----------------------------
~~~

We'll now use the dictionary-based assignment method to technically convert features `sex` and `smoker` to numerical:

```python
# replacing sex and smoker with numerical data
sex_and_smoker_replace = {"sex":{"female":1, "male":0} , "smoker":{"yes":1, "no":0}}
X = X.replace(sex_and_smoker_replace)
```

And the label-encoding assignment method to convert feature `region`:

```python
# Replacing region with numerical data
X["region"] = X["region"].astype('category')
X["region"] = X["region"].cat.codes
```

Hence, our new data matrix has the following structure:
	
<p align="center">
  <img width="365" height="399" src="exercises_images/new_data_matrix_ex1_ep3.png">
</p>

So, now, let's normalize our dataset, split it into training and test set, apply SVR and calculate the evaluation parameters (R^2 and MSE). The SVR algorithm is implemented in Python's `sklearn.svm` library.

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# feature names
feature_names = X.columns

# scaling
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns = feature_names)

# train - test splot
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=0)

# model
model = SVR(kernel='linear')
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)

#MSE
mse = mean_squared_error(y_test, y_pred)
print('MSE: ' + str(mse))

# R square
r_sq = model.score(X_test, y_test)
print('R^2: ' + str(r_sq))
```

~~~
MSE: 171709778.8315822
R^2: -0.0767564731816972
~~~

Results are obviously awful. In fact, the value of R^2 metric is negative, which means that the model is worse than the baseline! But why?

The first parameter that we typically tune is the `C` parameter, which has a default value `C=1`. The tuning process is usually done using trial and error method, which is characterized by repeated varied attempts which are continued until success. But before beggining to randomly testing values, let's think whether we should increase or decrease C value. C value is the reqularization parameter and it's purpose is to keep the regression coefficients as low as possible, so as to avoid overifitting. The lower the value of C, the lower the coefficients. However, our input values (values of X matrix) lie within [0,1] interval (because of normalization) and the output values are of order 10^3 or 10^4. So, we need large values in coefficients and, thus, we need to increase C value. After testing, we concluded that `C=1e3` is an appropriate value.

```python
# model
model_2 = SVR(kernel='linear', C=1e3)
model_2.fit(X_train, y_train.values.ravel())
y_pred = model_2.predict(X_test)

#MSE
mse = mean_squared_error(y_test, y_pred)
print('MSE: ' + str(mse))

# R square
r_sq = model_2.score(X_test, y_test)
print('R^2: ' + str(r_sq))
```

~~~
MSE: 58402132.49471051
R^2: 0.6337723183897759
~~~

Much much better! Now let's think what we can also do to increase the performace. A good idea would be to drop out `region` and `sex`, because logically speaking it doesn't seem to play any important role in our model. We could possible vertify that by checking the corresponding coefficients:

```python
import matplotlib.pyplot as plt

# plot feature importance
plt.figure(figsize=(10,8))
plt.bar(list(feature_names), list(model_2.coef_[0]))
plt.xticks(rotation = 'vertical')
plt.title('Feature scores')
plt.show()
```

<p align="center">
  <img width="720" height="576" src="exercises_images/ex1_fs_ep3.png">
</p>

As we expected, `sex` and `region` play the less singificant role in the final result, so let's drop them out:

```python
# as data frames
X_train = pd.DataFrame(X_train, columns = feature_names)
X_test = pd.DataFrame(X_test, columns = feature_names)

# Drop out sex and region
X_train.drop(columns = ['region', 'sex'], inplace=True)
X_test.drop(columns=['region', 'sex'], inplace=True)

# model
model = SVR(kernel='linear',  C=1e3)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)

#MSE
mse = mean_squared_error(y_test, y_pred)
print('MSE: ' + str(mse))

# R square
r_sq = model.score(X_test, y_test)
print('R^2: ' + str(r_sq))
```

~~~
MSE: 58139554.23566454
R^2: 0.6354188922894595
~~~

Unfortunately, the effect in the final result was tiny. Our last attempt is focused on tuning the `epsilon` parameter (check the theory above for what this parameter is). The default value in `epsilon` parameter is 0.1 and, at the same time, the output values are of order 10^3 or 10^4. This does not make much sense, we have to increase the value. After some tuning, we end up setting `epsilon = 4e3`.

```python
# model
model_3 = SVR(kernel='linear' ,  C=1e3, epsilon = 4e3)
model_3.fit(X_train, y_train.values.ravel())
y_pred = model_3.predict(X_test)

#MSE
mse = mean_squared_error(y_test, y_pred)
print('MSE: ' + str(mse))

# R square
r_sq = model_3.score(X_test, y_test)
print('R^2: ' + str(r_sq))
```

~~~
MSE: 44557039.09459874
R^2: 0.7205920326880368
~~~

The performance is much better! We've begun with a totally unreliable model and by fixing parameters we end up having a decent one.
