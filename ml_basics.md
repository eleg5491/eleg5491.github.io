---
layout: page
permalink: /ml_basics/
---

<!-- ## Note on Machine Learning Basics -->

Initial release: Jan 24, 2017. Hongyang Li.
Update: Feb 16, 2017. Hongyang Li and Kai Kang. 

Table of Contents:

- [Introduction](#intro)
- [Cross Entropy, Log-likelihood](#concept)
- [Generalization](#generalize)
  - [Relation to cross-validation](#cross)
- [Overfit and underfit](#overfit)
- [Summary](#summary)

<a name='intro'></a>

### Introduction

Last week we talked about the fundamental concepts in machine learning, from generative/discriminative model, overfit/underfit, optimization and different types of losses. In this post we point out some similar conceptions that are easy to be confusing for beginners. In general, the tasks of machine learning are broadly divided into supervised and unsupervised learning, where the former is trained with labelled data (called **supervision**) to minimize the loss and thus achieve better test performance, and the latter invesitigates the underlying data pattern itself without the help of supervision. In the real-world, we have millions of billions data without any or with limited label (see [YouTube-8M dataset](https://research.google.com/youtube8m/) from Google). Also we have semi-supervised learning where the supervision is partially included in the learning process and reinforcement learning where the target value in the loss is formulated via a careful design of reward and policy. 
<!-- We will dig into details in the following lectures. -->

<div class="fig figcenter fighighlight">
  <img src="/assets/ml/ml_task.png" height="250">
  <div class="figcaption">
    Machine learning tasks in broad categories. In this course, we mainly focus on the supervised learning where data are annotated with label and thus a training loss can be applied. With proper optimization method, we can obtain a set of deep learning features that can best represent the data pattern.
  </div>
</div>


<a name='concept'></a>

### Cross Entropy and Log-likelihood

<a name='generalize'></a>

### Generalization

The generalization ability of a machine learning algorithm describes how accurately an algorithm is able to predict outcome values for previously *unseen* data. The generalization error can be minimized by avoiding overfitting in the learning algorithm. A model is said to generalize well if its performance on the test set is high. The dataset for a standard procedure of supervised learning should contain *training*, *validation* and *test* set, where the validation set is for pruning hyperparameters and the test set is for verifying a model's performance and generalization ability.

- An example using training, validation and test sets: [Faster RCNN](https://arxiv.org/abs/1506.01497)  (Sec. 4.2)

- Another aspect to describe a model's generalization is to train a network on one dataset and evaluate it directly to another test set which has different classes: see [a paper here](https://arxiv.org/pdf/1606.04446v1.pdf) (Sec. 3.2)

<a name='cross'></a>

#### Relation to cross-validation

To be precise, there is a connection between generalization and stability (via cross-validation) of a learning algorithm.
If an algorithm is symmetric (the order of inputs does not affect the result), has bounded loss and meets [two stability conditions](https://en.wikipedia.org/wiki/Generalization_error#Relation_to_stability), it will generalize. For details, please refer to [wiki](https://en.wikipedia.org/wiki/Generalization_error). Here we want to point out that in some cases, the generalization of a model could be also reflected by conducting cross-valiation. There are two common types: (a) Leave-\\(p\\)-out cross-validation, it involves using \\(p\\) observations as the validation set and the remaining observations as the training set. This is repeated on all ways to cut the original sample on a validation set and a training set. 
(b) \\(k\\)-fold cross-validation, the original samples are *randomly* partitioned into \\(k\\) equal sized subsamples. Of the \\(k\\) subsamples, a single subsample is retained as the validation data for testing the model, and the remaining \\(k − 1\\) subsamples are used as training data. 
The cross-validation process is then repeated \\(k\\) times (called *folds*). The \\(k\\) results from the folds can then be averaged to produce a single estimation. The advantage of this method is that all observations are used for both training and validation, and each observation is used for validation exactly once. 
When \\(k = n\\) (the number of observations), the \\(k\\)-fold cross-validation is exactly the leave-one-out cross-validation.



<a name='overfit'></a>

### Overfit and underfit
The overfitting and underfitting problem is a common issue when training deep models. A brief and illustrative example is [here](http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html). **Underfiiting** means the model cannot find a solution that fits the training samples well and thus is incapable of capturing the true pattern of data. **Overfitting** refers to the case where the learner fits the training data too well, aka, has larger model capacity; it also captures the data noise and loses the ability to generalize well on test data.

<div class="fig figcenter fighighlight">
  <img src="/assets/ml/overfit_example.png" height="250">
  <div class="figcaption">
    A 1-D polinomial regression problem. Underfit (left), proper fit (middle) and overfit (right). We calculate the mean squared error (MSE) on the validation set. The higher of MSE, the less likely the model generalizes correctly from the training data.
  </div>
</div>

Below we sum up some reasons and their (possible) solutions to tackle overfit and underfit.

Reasons for underfit:

- Model capacity is not large enough: increase layers, add more neurons, from AlexNet to ResNet, etc.
- Hard to find global optimum or easy to get stuck at local minimum: try another initial point (adjust learning rate, momentum, etc.) or change learning policy (SGD, Adam, RMSProp, etc).
- Improper training logistics: longer iteration, diversity samples of different classes in one iteration.

Reasons for overfit:

- The number of candidate functions to describe the model is too large. Without sufficient data, the learner cannot distinguish which one is the most appropriate one: [increase training data](https://deeplearningmania.quora.com/The-Power-of-Data-Augmentation-2).

- Data is contaminated by noise and the model intends to be complicates in the parameter space: sparsify the network by adding penalty for complexity (known as **regularization**).
	- An example for linear regression (L2 norm): 
	$$
		L(w, x, y) = \frac{1}{N}  \sum_{i} \big(  w x_i^{(train)} - y_i^{(train)} \big)^2 + \lambda \| w \| ^2
	$$ 
	- There are many other forms of regularization, for example, [Dropout or DropConnect](http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf).

<a name='summary'></a>

### Summary
The following figure shows a rough splitup of the feature learning methods, where CNN and RNN in the supervised domain will be explicitly explored in the following lectures. In essence, deep learning models are just another workaround to provide more powerful representation of features than the traditional counterparts (such as HOG, SIFT, etc.) and based on the expressive and automatic learned features, data patterns (class clustering, for example) can be better distinguished in higher dimensional space. 
<div class="fig figcenter fighighlight">
  <img src="/assets/ml/feature_learning.png" height="250">
  <div class="figcaption">
    Taxonomy of feature learning methods. 
    <!-- Deep neural networks are of the main interest in this course.  -->
    Credit from <a href="https://sites.google.com/site/deeplearningcvpr2014/">Honglak Lee's Tutorial</a>.
  </div>
</div>

At last, the unsupervised methods will be briefly introduced in the upcoming lecture and you can find a good starting tutorial [here](http://www.uoguelph.ca/~gwtaylor/outbox/gwt_unsupervised_learning.pdf). The general knowledge discussed above are useful throughout this course and must be reflected when we publish professional research papers.
