# Apache Flink

- spark and flink intermediate data is kept in memory instead of writing to disk.

  ![image-20200813144402113](/home/omer/image-20200813144402113.png)

## Spark vs Flink

- spark is not a true real time processing. It is near real time. Flink is a true stream processing framework.
- Spark streaming computation model is based on micro-batching (breaking the stream into small chunks). with flink the data is treated as pure stream based on windowing and check pointing.
- spark does not have an effivient memory manager. Flink has its own efficient automatic memory. No out of memory errors  with flink.
- Spark Uses DAG as it's execution engine, while flink uses controlled cyclic dependency graph.
- speed wise flink and storm perform the same for data rate wile spark performed 3 times slower.

## Flink

- every operation in flink will create a new dataset.

## DataStream API

- 



# Example of machine learning model

### Description

- I got the opportunity to create a model to predict the hours required to complete maintenance jobs

### Problem:

- the customer provides a gui that plugs in to the customer database





# Machine Learning Fundamentals

### What are the steps for data wrangling and data cleaning

- Data profiling: understand the dataset by calling the describe and shape on dataset
- Data visualizations: histograms, box plots to understand the relationship between variables
- syntax error; no white space, typos
-  standardization or normalization 
- handling null values

### How to Deal with unbalanced binary classification

- consider the metrics to use: accuracy is gonna be a bad metric most likely, think about precision, recall, f1 and also using a confusion matrix
- you can also increase the penalty of misclassifying the minority class
- oversampling the minority class or under-sampling the majority class

### Describe the difference between regularization methods eg L1, L2

- both L1 and L2 are used to reduce the over-fitting of training data
- The key difference between these two is the penalty term

### precision and recall

- **Recall** attempts to answer “What proportion of actual positives was identified correctly

  Recall = True Positive / (True Positive + False Negative)

- **Precision** attempts to answer “What proportion of positive identifications was actually correct

### Difference between Supervised and unsupervised learning

- supervised learning involves learning a function that maps an input to an output based on example input-output pairs
- unsupervised learning is used to draw inferences and find patterns from input data without references to labeled outcomes. a common use of unsupervised learning is grouping customers by purchasing behavior to find target markets

### Bias Variance Trade off ?

- Bias is error introduced in your model due to over simplification of machine learning algorithm. It can lead to under-fitting
- Variance is error introduced in your model due to complex machine learning algorithm, your model learns noise also from the training dataset and performs bad on test dataset. It can lead high sensitivity and over-fitting
- Increasing the bias will decrease the variance. Increasing the variance will decrease the bias

### Reinforcement Learning

- develop a system(agent) that improves its performance based on interactions with the environment
- the feedback is not the ground truth it is the measure of how well the action was measured by a reward function

### Machine Learning Work-flow

1. data preprocessing
   - feature selection
   - feature extraction
   - dimensionality reduction
   - sampling (80/20)
2. Learning 
   - Model selection
   - cross-validation
   - performance metrics
   - hyper parameter optimization
3. Evaluation
4. Prediction

## Data Preprocessing

### Dealing with missing data

- you can eliminate samples or features with missing values
- imputing missing values

### Handling categorical data

#### Ordinal features

- categorical features that can be sorted or ordered

#### Nominal features

- features that can not be ordered for example colors

### Training Test split

- comparing the model performance on the test set is the unbiased evaluation of the model

### Feature scaling

- Decision trees and random forest classifiers are the two methods that don't require feature scaling, they are feature invariant
- most machine learning algorithms require mature scaling or else the algorithm will be busy optimizing the weights according to the features that have larger scales and not taking much account of smaller scale features

### Reducing Generalisation Error

- generalization error is caused by over fitting the model to the training data so the model does not generalize well to data that is not included in the dataset 
- types of ways to reduce generalization error are:
  1. collect more training data
  2. introduce a penalty for complexity via regularization
  3. choose a simpler model with fewer parameters
  4. reduce the dimensionality of the data

### Regularization

- Can modify the cost function to add preference for certain parameter values
- regularization term is independent of the data: paying more attention reduces our variance
- the cost function now includes both penalty and cost function to be minimized

#### Ridge Regularization (L2)

- reduce the complexity of the model by penalizing large individual weights

#### Lasso Regularization (L1)

- the cost function to be minimized that includes the cost and penalty term encourages sparsity of the weights because of the diamond shape of the penalty term which is the sum of the absolute weight coefficients 

### Dimensionality reduction

- Dimensionality reduction is the process of reducing the number of features in a dataset. This is important mainly in the case when you want to reduce variance in your model (over fitting)
- advantages of dimensionality reduction
  - It reduces time and storage space required
  - removal of multi-collinearity improves the interpretation of the parameters of the machine learning model
  - It becomes easier to visualize the data when reduced to very low dimensions such as 2D or 3D
  - it avoids the curse of dimensionality
- Types of dimensionality reduction fall under 2 types feature selection techniques and feature extraction techniques

#### Feature Selection techniques

- sequential backward selection (SBS)

#### Feature Extraction 

##### principal component analysis

- PCA helps identify patterns in data based on the correlation between features
- PCA finds the direction of maximum variance in high dimensional data and projects that onto a new subspace with equal or lesser dimensions

##### Linear discriminant analysis

- find the feature subspace that optimizes class separability

## Models

### Steps for training machine learning model

1. selecting features and collecting training samples
2. choose a performance metric
3. choosing a classifier and optimization algorithm
4. evaluating the performace of the model
5. tuning the algorithm

### Perceptron

#### notes

- Linear classifier that is guarnteed to converge if the data is linearly seperable

#### implementation

```python
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)

print('Misclassification samples: %d' % (y_test != y_pred).sum())
Misclassification samples: 1
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
Accuracy: 0.978
```

![img](https://learning.oreilly.com/library/view/python-machine-learning/9781789955750/Images/B13208_03_01.png)

### Logistic regression 

#### notes

- classification model performs very well on linearly separable classes

- $$
  \begin{align*}
  ln(p(y=1|x))=(w~0~x0+ w1x1+...+wmxm = \sum wixi = wtx
  \end{align*}
  $$

![img](https://learning.oreilly.com/library/view/python-machine-learning/9781789955750/Images/B13208_03_03.png)

#### implementation

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, random_state=1,
                       solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)
lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)

array([2, 0, 0])
```

#### Applying varying regularization parameter

- below we will vary the C parameter which is the inverse of the regularization parameter between 10^-5 to 10^5. When we decrease C we increase the regularization

```python
weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, random_state=1,
                           solver='lbfgs', multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0], label='patel length')
plt.plot(params, weights[:, 1], label='patel width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
```

![img](https://learning.oreilly.com/library/view/python-machine-learning/9781789955750/Images/B13208_03_08.png)

### Support Vector Machines

#### notes

- can be used for both regression and classification
- an extension of the Perceptron, in the perceptron we minimize misclassification error but in the svm we maximize the margin which is the distance between the decision boundary
- attempts to plot in n-dimensional space n features the value of each feature is a particular coordinate
- uses hyper planes to separate out different classes based on the provided kernel function
- kernel types:
  - linear
  - radial
  - polynomial
  - sigmoid

![img](https://learning.oreilly.com/library/view/python-machine-learning/9781789955750/Images/B13208_03_09.png)

#### Implementation

```python

```



### Decision Tree

- s

