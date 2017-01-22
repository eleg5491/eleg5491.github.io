---
layout: page
permalink: /ml_basics/
---

<!-- ## Note on Machine Learning Basics -->

Initial release: Jan 24, 2017. Hongyang Li.

Table of Contents:

- [Introduction](#intro)
- [Cross Entropy, Log-likelihood](#concept)
- [Generalization](#generalize)
  - [Relation to cross-validation](#cross)
- [Overfit and underfit](#overfit)
- [Summary](#summary)

<a name='intro'></a>

### Introduction

Last week we talked about the fundamental concepts in machine learning, covering a wide span from generative/discriminative model, overfit/underfit, optimization and different types of losses. In this post we point out some similar conceptions that are easy to be confusing for beginners. In general, the tasks of machine learning are broadly divided into supervised and unsupervised learning, where the former is trained with labelled data (called *supervision*) to minimize the loss and thus achieve better test performance, and the latter invesitigates the underlying data pattern itself without the help of supervision. In the real-world, we have millions of billions data without any or with limited label (see [YouTube-8M dataset](https://research.google.com/youtube8m/) from Google).

<div class="fig figcenter fighighlight">
  <img src="/assets/ml/ml_task.png" height="250">
  <div class="figcaption">
    Machine learning tasks in broad categories. In this course, we mainly focus on the supervised learning where data are annotated with label and thus a training loss can be applied. With proper optimization method, we can obtain a set of deep learning features that can best represent the data pattern.
  </div>
</div>


<a name='concept'></a>

### Cross Entropy and Log-likelihood

