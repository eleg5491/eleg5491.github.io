---
layout: page
permalink: /backprop/
---

Initial release: Feb 14, 2017. Hongyang Li.



### Optimization

### Computing Gradient

\\( x = \{x_1, \dots, x_i, \dots, x_m\} \in  \mathcal{R} ^m \\)  

W \in \mathcal{R}^{n \times m}


z = f(x, w) = Wx, z \in \mathcal{R}^n

\frac{\partial L}{ \partial z} \\
\frac{\partial L}{ \partial x} = \frac{\partial L}{ \partial z} \frac{\partial z}{ \partial x}

\frac{\partial L}{ \partial W} = \frac{\partial L}{ \partial z} \frac{\partial z}{ \partial W}

x = \{x_1, \cdots, x_i, \cdots, x_m\}, m =4 \\
z = \{z_1, \cdots, z_j, \cdots, z_n\}, n =3 \\

\frac{\partial L }{\partial z_j } = (f - y)_j

\frac{\partial L}{ \partial (W_j)_i} = \frac{\partial L}{ \partial z_j} \frac{\partial z_j}{ \partial (W_j)_i}

 \nabla_{ w^a} L  = \frac{\partial L }{ \partial a}  \frac{a }{ \partial w^a } 


$$
\min_{w^*} \frac{1}{2} \big \| f(x, w) - y \big \| ^2 \\
w = w - \alpha \frac{ \partial L}{ \partial w}
$$

### Design Customized Layers

### BP Theory on the way