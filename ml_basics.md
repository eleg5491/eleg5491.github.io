---
layout: page
permalink: /ml_basics/
---

<!-- ## Note on Machine Learning Basics -->

Initial release: Jan 24, 2017. Hongyang Li.

Update: Feb 16, 2017. Hongyang Li and Kai Kang. 

Table of Contents:

- [Introduction](#intro)
- [Cross Entropy](#concept)
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

### Cross Entropy

One of the most important thing in deep learning is to minimize the loss from the learned features and its corresponding label. Let \\( x_i \in \mathbb{R}^d \\) be the \\(d\\)-dimension feature of \\(i\\)-th sample in one mini-batch. The weight matrix in the classification layer is denoted as \\( W \in \mathbb{R}^{n \times d} \\), where \\( y_i = Wx_i+b\\). The label of sample \\( x_i \\) is \\( l_i \in \{ 0, 1, \cdots, n-1 \}\\), where \\( n\\) is the number of categories. The [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) loss, in plain English, measures the difference between two probability distributions \\(p\\) and \\(q\\) over a given set:

$$
H(p, q) = -\sum_x p(x) \log q(x).
$$

Putting the cross entropy loss into our notation defined above, it goes like:

$$
L(W, x_i, l_i) = - \sum_{j=1}^n (t_i)_j \log (\hat{y}_i)_j,
$$

where \\(\hat{y}_i \in \mathbb{R}^n\\) is the normalized probability of the output in the classification layer. \\((\cdot)_j\\) denotes the \\( j\\)-th element in a vector. Here \\(t_i \in \mathbb{R}^n\\) is a vectorized target mapping from a single scalar \\(l_i\\). For instance, in the example shown below, sample \\(x_i\\) belongs to class \\(l_i =2\\), thus its target vector is \\(t_i = [0, 0, 1]^T\\). The total loss over a batch \\(\mathcal{B}\\) is just the summation over all samples: \\(\sum_i L(W, x_i, l_i)\\).
The gradient of \\(L\\) w.r.t. input \\(\hat{y}_i\\) is:

$$
 \frac{\partial L}{\partial \hat{y}_j } = -\frac{t_j}{\hat{y}_j}. 
$$


 It is in an element-wise form and we remove the subscript \\(i\\) for brevity. Hope you are not lost with the notations here. :) In a general sense, \\(t_j\\) can be continuous; in softmax loss, the target has only one element to be 1 and all the others as zero. That's why you often see the softmax (cross-entropy) loss over a batch is written as:

 $$
 L(W, \mathcal{B}) = - \sum_{i=1} \log \hat{y}_{i , l_i}.
 $$

The notation \\(\hat{y}_{i , l_i}\\) denotes the normalized probability output (scalar) corresponding to the \\(l_i\\)-th dimension in \\(\hat{y}_i\\), a.k.a, its  label index.
In Caffe, the [softmax loss](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1SoftmaxWithLossLayer.html) is known as `SoftmaxWithLoss`.

<div class="fig figcenter fighighlight">
  <img src="/assets/ml/loss.png" height="220">
  <div class="figcaption">
    The loss expression via a detailed example. Adaption from <a href="http://cs231n.github.io/linear-classify/#svmvssoftmax">a blog</a>. Taking the feature 
    <b>x</b> as input, we have a raw probability output <b>y</b>; the loss descends from a pair (<b>y</b>, label) and could be in various forms (Euclidean, hingeloss, softmax loss, etc.), depending on the input <b>y</b> and label.
  </div>
</div>

A very similar loss that often confuses beginners is the cross-entropy loss of independent multi-classes. Take the example above again, in softmax loss, the summation of \\(
\hat{y}_i
\\) is 1, indicating that the sample can belong to one class *only* (the input image is a dog, not a person/cat/desk, etc). It is a **one-of-many** classification problem. In multi-class loss (so I call it!), each element in the probability vector is independent and ranges from 0 to 1 via some mapping: \\(  
\hat{y}_i = \sigma(y_i) \in \mathbb{R}^n
\\), where \\(\sigma(\cdot)\\) is a sigmoid function, for instance. Therefore, \\(\hat{y}_i\\) becomes \\( \hat{y}_i = [0.055, 0.703, 0.569]^T\\); each element could mean whether the sample has person or not, whether the scene is indoor or outdoor, whether the person is laughing or not, etc. It is a **multi-binary** classification problem. Using the notation defined above, the loss and gradient of multi-class are as follows (removing the sample index \\(i\\) for brevity):

$$
L(W, x, l) = - \sum_{j=1}^n 
l_j \log \hat{y}_j + (1 - l_j ) \log ( 1- \hat{y}_j ),  \\
\frac{\partial L}{\partial y_j } = \frac{\partial L}{\partial \hat{y}_j } \frac{\partial \hat{y}_j}{\partial y_j } = \hat{y}_j - l_j,
$$

where the label is now in a vector form: \\(l_i \in \mathbb{R}^n\\), with each element \\(l_j\\) being 0 or 1. The vector form of the gradient regarding sample \\(i\\) is \\( \frac{\partial L}{\partial y_i } = (\hat{y}_i - l_i) \in \mathbb{R}^n\\). It is a little bit tedious in the derivations above, but we want students to be crystal clear about the gradient flow in each element of data in the network. In Caffe, the [multi-class loss](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1SigmoidCrossEntropyLossLayer.html#details) is known as `SigmoidCrossEntropyLoss`.



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


<div>
  <div id="disqus_thread"></div>
  <script>
  /**
  * RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
  * LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
  */
  // var disqus_config = function () {
  // this.page.url = PAGE_URL; // Replace PAGE_URL with your page's canonical URL variable
  // this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
  // };
  (function() { // DON'T EDIT BELOW THIS LINE
  var d = document, s = d.createElement('script');

  s.src = '//hongyangblog.disqus.com/embed.js';

  s.setAttribute('data-timestamp', +new Date());
  (d.head || d.body).appendChild(s);
  })();
  </script>
  <noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>
</div>