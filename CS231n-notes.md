# CS231n

### p2

Object Recongnition

ImageNet 

2012 CNN

### p3

Image classification

object detection

CNN 

2012 AlexNet: 7-8 layers

1998 LeCun CNN born

Why?

1. Computation->large network
2. Better data: high quality labeled datasets

 True, deep understanding of a picture, a long way to go

piazza

### p4 

Image in computer: a giant grid of numbers ->the problem: Semantic Gap

challenges: Viewpoint variation/ Illumination/ Deformation/ Occlusion/ Background Clutter/ Intraclass variation

An image classifier? Unable to write an algorithm.

Maybe find edges, then find corners...

#### Data-Driven Approach

1. Collect a dataset of images and labels
2. Use ML to train a classifier
3. Evaluate the classifier on new images

* Two funcs:

   ```python
  def train(images, labels):
    # Machine Learning!
    return model
  ```

  ```python
  def predict(model, test_images):
    #use model to predict labels
    return test_labels
  ```

  

##### Nearest Neighbor

Dataset: **CIFAR10**

**Distance Metric** to compare images

* **L1 distance:**  $d_1(I_1,I_2)=\sum_p\left|I_1^p-I_2^p\right|$

```python
import numpy as np

class NearestNeighbor:
  def __init__(self):
    pass
  
  def train(self, X, y):
    ‘’‘ X is N x D where each row is an example. Yis 1-d of size N ’‘’
    # the nearest neighbor classifier simply remebers all the tarining data
    self.Xtr = X
    self.ytr = y
    
  def predict(self X):
    ’‘’ X is N x D where each row is an example wish to predict label for ‘’‘
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
    
    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i*th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) # using broadcast
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
      
    return Ypred
```

* Tarin O(1); predict O(N)
* We want **fast at prediction; slow for training is ok**

### p5

#### K-Nearest Neighbors

* Take **majority vote** from **K** closest points

* Chosing the **Distance Metric**:

  L1(Manhattan) distance

  L2(Euclidean) distance :$d_2(I_1,I_2)=\sqrt{\sum_p(I^p_1-I^p_2)^2}$

  * L1 distance depends on ur choice of coordinate frame, but L2 does not
  * That means if the data has some specific meaning, it is better to use L1 distance

Such K and distance are **hyperparameters**: cannot learn from data, depend on the problem

* It is wrong to choose hyperparameters that work best on the data/ the test data: **Evaluate on the test set only a single time, at the very end**
* Split data into **train**, **validation**, and **test** sets; choose hyperparameters on validation set and evaluate on test
* The splits people tend to use is between 50%-90% of the training data for training and rest for validation.
* **Cross-Validation**: computational expensive, and not usually in deep learning.
* Typical number of folds you can see in practice would be 3-fold, 5-fold or 10-fold cross-validation.
* If the number of hyperparameters is large you may prefer to use bigger validation splits. If the number of examples in the validation set is small (perhaps only a few hundred or so), it is safer to use cross-validation.

```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```

Problems:

* Vary slow at test time: we often care about the test time efficiency much more than the efficiency at training time
* Distance metrics on pixels are not informative
* Curse of dimensionality
  * images are high-dimensional objects (i.e. they often contain many pixels), and distances over high-dimensional spaces can be very counter-intuitive
  * the pixel-wise distance does not correspond at all to perceptual or semantic similarity.

### p6

#### Linear Classification

**Parametric Approach** :

**f(x,W) = Wx + b**: x means input data; W means parameters or weights; b means bias vector

* Shape: $x = 32*32*3 = 3072 * 1; W = 10 * 3072; b = 10 * 1 ;\  f(x,W)= 10 * 1$

* Problem: it can only learn one catagory, and produce one template; hard cases: multimodal data; odd and even problem

* We can interpret linear classifier as template matching: each row of **W** correspond to a template

* A commonly used trick is to combine the two sets of parameters **W,b** into a single matrix that holds both of them by extending the vector x with one additional dimension that always holds the constant 1- a default *bias dimension*. With the extra dimension, the new score function will simplify to a single matrix multiply: **f(x,W) = Wx**

* **Image data preprocessing**: change the raw pixel values from [0,255] to [-1,1]

### p7

#### **Loss function**

* Quantify how good the model is

* Loss over the dataset: $L=\frac{1}{N}\sum_iL_i(f(x_i,W),y_i)$

* **Multiclass SVM loss**:
  * Given an example $(x_i,y_i)$ where $x_i$ is the image and $y_i$ is the (integer) label
  
  * using the shorthand for the scores vector: $s_i=f(x_i,W)$
  
  * SVM loss: $$L_i=\sum_{j\not=y_i}\left\{\begin{array}{l} 0 &if\quad s_{y_i}\ge s_j+1 \\s_j-s_{y_i}+1 &otherwise\end{array}\right.\\ \quad =\sum_{j\not=y_i}max(0,s_i-s_{y_i}+1)$$
  
  * hinge loss(->squared hinge loss)
  
  * The SVM loss function wants the score of the correct class $y_i$ to be larger than the incorrect class scores by at least by Δ (delta). If this is not the case, we will accumulate loss.
  
  * Here we set the safety margin as 1, when we start off training, the scores are approximately equal to 0, so the expected loss equals (numbers of classes - 1)
  
  * There is no difference between using sum or mean (scaling does not matter)
  
  * W is not unique->2W, 3W...
  
    ```python
    def L_i_vectorized(x, y, W):
    	scores = W.dot(x)
    	margins = np.maximum(0, scores - scores[y] + 1)
      margins[y] = 0
      loss_i = np.sum(margins)
    return loss_i
    ```
  
* Adding **regularization loss**: L= $\frac{1}{N}\sum_iL_i(f(x_i,W),y_i)+\lambda R(W)$
  * Model should be simple, no over-fitting(not just fitting the training data): the more complex the model is, the greater the penalty is
  * Occam's razor
  * $\lambda$ is an important hyperparameter: regularization strength
  * When combining with gradient descent, it becomes **Weight Decay**
  * Why regularize?
    * Express preferences over weights
    * Make the model simple so it works on test data
    * Improve optimization by adding curvature
  * In common use:
    * **L2 regularization**: $R(W) = \sum_k\sum _l W^2_{k,l}$ 
      * penalizing the euclidean norm of the weight vector;
      * spreading the influence across all x, not depending only on some certain elements (overfitting)
    * **L1 regularization**: $R(W) = \sum_k\sum _l |W_{k,l}|$  
      * encouraging sparsity in the matrix W
    * Elastic net(L1 + L2): $R(W) = \sum_k\sum _l \beta W_{k,l}^2+ |W_{k,l}|$
  
* **Softmax Classifier** (Multinomial Logistic Regression) 
  
  * Probability distribution: $P(Y=k|X=x_i)=\frac{e^{s_k}}{\sum_je^{s_j}}$
  
  * Minimizing the negtive log likelihood of the correct class -- *Maximum Likelihood Estimation*: $L_i=-logP(Y=y_i|X=x_i)=-s_i+log\sum_je^s_j$
  
  * We can now also interpret the regularization term $R(W)$ in the full loss function as coming from a Gaussian prior over the weight matrix $W$, where instead of MLE we are performing the *Maximum a posteriori* (MAP) estimation.
  
  * Min loss is 0; max loss is infinity (actually we will never reach them)
  
  * when we start off training, the scores are approximately equal to 0, so the expected loss equals $log(C)$
  
  * Practical issues: **Numerical stability**:
  
    * the intermediate terms $e^{s_i}, \sum_je^{s_i}$ can be very large, and dividing large numbers can be numerically unstable.
  
    * So we use a normalization trick:$\frac{e^{s_k}}{\sum_je^{s_j}}=\frac{Ce^{s_k}}{C\sum_je^{s_j}}=\frac{e^{s_k+logC}}{\sum_je^{s_j+logC}}$
  
    * A common choice for $C$ is to set $logC=-max_js_j$ (shifting the values inside the vector $s$ so that the highest value is zero.)
  
    * ```python
      f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
      p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup
      
      # instead: first shift the values of f so that the highest number is 0:
      f -= np.max(f) # f becomes [-666, -333, 0]
      p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
      ```
  
  * **Cross-entropy loss**:
    * The *cross-entropy* between a “true” distribution **p** and an estimated distribution **q** is defined as: $H(p,q) = -\sum_xp(x)log[q(x)]$
    * The Softmax classifier is hence minimizing the cross-entropy between the estimated class probabilities ($q=\frac{e^{s_k}}{\sum_je^{s_j}}$) and the “true” distribution, which in this interpretation is the distribution where all probability mass is on the correct class (i.e. $p=[0,…1,…,0]$ contains a single 1 at the $y_i$ position.). 
  
* suprvised learning

#### **optimazation**

* Convex optimazation: https://stanford.edu/~boyd/cvxbook/

* Radom search

* Gradient: the direction of steepest descent is the negative gradient
  * Finite differences approximation (numerical gradient): super slow
  
    ```python
  def eval_numerical_gradient(f, x):
      """
      a naive implementation of numerical gradient of f at x
      - f should be a function that takes a single argument
      - x is the point (numpy array) to evaluate the gradient at
      """
    
      fx = f(x) # evaluate function value at original point
      grad = np.zeros(x.shape)
      h = 0.00001
    
      # iterate over all indexes in x
      it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
      while not it.finished:
    
        # evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h # increment by h
        fxh = f(x) # evalute f(x + h)
        x[ix] = old_value # restore to previous value (very important!)
    
        # compute the partial derivative
        grad[ix] = (fxh - fx) / h # the slope
        it.iternext() # step to next dimension
    
      return grad
    ```
  
    
  
  * Analytic gradient: exact, fast, error-prone
  
  * Numerical gradient: centered formula $\frac{f(x)}{dx}=\frac{f(x+h)-f(x-h)}{2h}$
  
  * **Gradient check**: 
  
    * Using analytic gradient, but check implementation with numerical gradient
    * Relative error: $\frac{\left |f'_a-f'_n \right|}{max(\left |f'_a\right|,\left|f'_n \right|)}$ (max or add)
    * relative error > 1e-2 usually means the gradient is probably wrong
    * 1e-2 > relative error > 1e-4 should make you feel uncomfortable
    * 1e-4 > relative error is usually okay for objectives with kinks. But if there are no kinks (e.g. use of tanh nonlinearities and softmax), then 1e-4 is too high
    * 1e-7 and less you should be happy
    * The deeper the network, the higher the relative errors will be
    * Using double but not float for precision, and if the gradients or step size h are too small (<1e-10) there will be some numerical issues
    * One source of inaccuracy to be aware of during gradient checking is the problem of *kinks*. Kinks refer to non-differentiable parts of an objective function, introduced by functions such as ReLU (max(0,x)max(0,x)), or the SVM loss, Maxout neurons, etc
    * Using only few datapoints to solve the above problem
    * It is best to use a short **burn-in** time during which the network is allowed to learn and perform the gradient check after the loss starts to go down
    * Regularization loss may overwhelm the data loss, in which case the gradients will be primarily coming from the regularization term. Therefore, it is recommended to turn off regularization and check the data loss alone first, and then the regularization term second and independently
    * Remember to turn off dropout/augmentations
    * Make sure to gradient check a few dimensions for every separate parameter
  
* Gradient descent:

  * $L(W)=\frac{1}{N}\sum^N_{i=1}\nabla W L_i(x_i,y_i,W)+\lambda \nabla_W R(W)$

  ```python
  # Vanilla Gradient Descent
  
  while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += - step_size * weights_grad # perform parameter update
  ```

  * Step size or learning rate is an important hyper-parameter
  * In practice, the $N$ can be quite large, so we often use a minibatch of examples (32/64/128)
  * Stochastic Gradient Descent (SGD)

  ```python
  # Vanilla Minibatch Gradient Descent
  
  while True:
    data_batch = sample_training_data(data, 256) # sample 256 examples
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += - step_size * weights_grad # perform parameter update
  ```

#### Image Features:

* Using proper feature representation is much better than putting in raw pixels in to the classifier
* Color histogram
* Histogram of Oriented Gradients
* Bag of Words
* ConvNets: learn the features from the data

### P8

#### Computational graphs

* Back Propagation: recursively use of the chain rule

* At each node, we just compute the local gradient and multiply it with receiving numerical values of gradients coming from upstream
* We can group any nodes to make any complex nodes
* **Gradients add up at forks**. The forward expression involves the variables **x,y** multiple times, so when we perform backpropagation we must be careful to use `+=` instead of `=` to accumulate the gradient on these variables (otherwise we would overwrite it). This follows the *multivariable chain rule* in Calculus, which states that if a variable branches out to different parts of the circuit, then the gradients that flow back to it will add.
* $\sigma(x)=\frac{1}{1+e^{-x}}$ , called sigmoid function, and $\frac{d\sigma(x) }{dx}=(1-\sigma(x))\sigma(x)$
* patterns in backward flow:
  * Add gate: gradient distributor
  * Max gate: gradient router
  * Mul gate: gradient switcher
  * Copy gate: gradient adder
* Gradients add at branches
* Gradients for vectorized code: Jacobian matrix
* In practice, the Jacobian matrix can be vary large; and sometimes they are diagnoal matrix (element-wise)
* The gradient with respect to a variable should have the same shape as the variable

* $Y=WX$: $\frac{\partial L}{\partial x}=\frac{\partial L}{\partial Y}W^T$, $\frac{\partial L}{\partial W}=X^T\frac{\partial L}{\partial Y}$

Modularized implementation: forward / backward API

```python
class ComputationalGraph(object):
  #...
  def forward(inputs):
    # 1. pass inputs to input gates...
    # 2. forward the computational graph:
    for gate in self.graph.nodes_topologically_sorted():
      gate.forward()
     return loss # the final gate in the graph outputs the loss
  def backward():
for gate in reversed(self.graph.nodes_topologically_sorted()):
  gate.backward()
  return inputs_gradients
```

### P9

#### Neural Network

* Neural network is stacking some simple functions in a hierachical way in order to make up a more complex non-leaner fuction
* $s=W_2max(0,W_1x)$
* Non-linear layers are important: ReLU....  
* Each row of $W1$ is a template, and $W2$ is the weighted sum of all these intermediate scores ($h$ which is after the non-linearity) 
* The abstraction of a layer has the nice property that it allows us to use efficient vectorized code (e.g. matrix multiplies)
* The forward pass of a fully-connected layer corresponds to one matrix multiplication followed by a bias offset and an activation function.
* The neural network can approximate any continuous function

### P10

#### History of CNN

### P11

#### Convolutional Neural Network

* Convolution layer: preseve spatial structure

* Filters always extend the depth of the input volume, and each filter produce an activation map which means the depth equals the number of filters
* Convolution: taking a dot product between the filter and a small chunk of the image
* Actually, we stretch the filter out into a long vector to do the dot product
* Filters at the earlier layers: low-level features like edges; Filters at the mid-level layers: more complex features like corners and blobs; high-level features....
* Output size: **(N - F) / stride + 1**
* In practice: Common to zero pad the border to maintain the same size we have before: **(N - F + 2P) / stride + 1**
* Don't forget the bias term
* number of params: **F  * F * D * K** weights and **K** biases

### P12

* 5x5 filter = 5x5 receptive field

* Pooling layer: 

  * Makes the representations smaller and more manageable
  * Spatially downsampling (without depth)
  * In practice, we often set the stride to avoid overlapping (one value to represent one reign)
  * Max pooling: more intuitive
  * Same equation: **(N - F) / stride + 1**
  * People don't use padding in pooling
  * The depth dimension remains unchanged

* Fully connected layer

* Typical architectures:

  **[(CONV - RELU)\*N - POOL]\*M - (FC - RELU)*K, SOFTMAX**

  * N is usually up yo 5, M is large, 0<=K<=2

### P13

#### **Activation Functions**

* Sigmoid: 

  * $\sigma(x)=1/(1+e^x)$
  * Squashes numbers to range [0,1]
  * Can be interpreted as a saturating firing rate of a neuron
  * Problems:
    1. "kills" the gradients (when x is away from 0);
    2.  its outputs are not zero-centered. When the input to a neuron is always positive,the gradients on $w$ are all positive or negative, causing inefficient gradients update
    3. Exp() is compute expensive

* Tanh(x)

  * Squashes numbers to [-1,1]
  * Tanh is a scaled sigmoid: $tanh(x)=2\sigma(2x)-1$ 
  * Zero centered
  * kills gradients when saturated

* ReLU (Rectified Linear Unit) **BEST**!

  * $f(x)=max(0,x)$
  * Don't saturate (in + region)
  * Very computationally effiecieny
  * Greatly accelerate the convergence of stochastic gradient descent compared to sigmoid/tanh in practice
  * More biologically plausible than sigmoid
  * Problems:
    1. Not zero-centered output
    2. Kills the gradients when x<=0: dead ReLU. So people like to initialize ReLU neurons with slightly positive biases.
  * Be careful with the learning-rate!

* Leaky ReLU (Good)

  * $f(x)=max(0.1x, x)$

  * Will not die

* Parametric ReLU

  * $f(x)=max(\alpha x,x)$
  * Backprop into $\alpha$

* ELU (Exponential Linear Units)

  * $f(x)= \left\{\begin{array}{lrc}x&if&x>0\\\alpha(exp(x)_1)&if&x\leq0\end{array}\right.$

* All benefits of ReLU
    * Closer to zero mean outputs
    * Negative saturation regime compared with Leaky ReLU, adds some robustness to noise
* SELU (Scaled Exponential Linear Units)
* Maxout Neuron (Good)
  * $f=max(w_1^Tx+b_1,w_2^T+b_2)$
  * Generalizes ReLU and Leaky ReLU
  * Linear Regime: does not saturate or die
  * Problem: double the number of parameters

#### Data Preprocessing

* **Zero-centered data**: ``X -= np.mean(x, axis = 0)``

  * Subtract the mean image: AlexNet
  * Subtract per-channel mean: VGGNet

* Normalized data: ``X /= np.std(X, axis = 0)``

* Any preprocessing statistics (e.g. the data mean) must only be computed on the training data, and then applied to the validation / test data

* PCA and Whitening

  * ```python
    # Assume input data matrix X of size [N x D]
    X -= np.mean(X, axis = 0) # zero-center the data (important)
    cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
    U,S,V = np.linalg.svd(cov) # SVD factorization
    Xrot = np.dot(X, U) # decorrelate the data
    """
    Using PCA: 
    Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced becomes [N x 100]
    """
    # whiten the data:
    # divide by the eigenvalues (which are square roots of the singular values)
    Xwhite = Xrot / np.sqrt(S + 1e-5)
    ```

* **Weight Initialization**
  * If we initialize all weights using the same value, their gradients will be all same and act the same (no symmetry breaking)
  * Initialization too small: Activation go to zero, gradients also zero, No learning
  * Initialization too big: Activations saturate, gradients zero, No learning
  * Initialization just right: Nice distribution of activations at all layers, Learning proceeds nicely
  * Small random numbers (but does not fit for deep network)
  * Xavier initialization: ``W = np.random.randn(Din, Dout) / np.sqrt(Din)``
  * Kaiming initializaion: Due to the reason that ReLU kills half of the neurons, ``W = np.random.randn(Din, Dout) / np.sqrt(Din / 2)``
  * It is possible and common to initialize the biases to be zero, since the asymmetry breaking is provided by the small random numbers in the weights. 

### P14

#### **Batch Normalization**

* Forcing the inputs to be unit Gaussian 

* Compute the empirical mean and variance (for each mini-batch) independently for each dimension (element-wise)
  * Normalization: $\hat{x}^{(k)}=\frac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}$
  * Scaling and shifting: $\hat{y}=\gamma ^{(k)}\hat{x}^{(k)}+\beta ^{(k)}$ ,the network can learn:$\gamma ^{(k)}=\sqrt{Var[x^{(k)}]}\ \ \ \  \beta^{(k)}=E[x^{(k)}]$
  * Insert after FC or Conv layers and before nonlinearity
  * At test time, using average of values seen during training
  * Benefits:
    * Improves gradient flow through the network
    * Allows higher learning rates. Faster convergence
    * Reduces the strong dependence on initialization
    * Acts as a form of regularization

#### **Babysitting the Learning Process**

* Preprocess the data

* Choose the architecture

* Before optimization:

  * **Look for correct loss at chance performance.** Make sure you’re getting the loss you expect when you initialize with small parameters. It’s best to first check the data loss alone (so set regularization strength to zero)
  * As a second sanity check, increasing the regularization strength should increase the loss
  * **Overfit a tiny subset of data**. Lastly and most importantly, before training on the full dataset try to train on a tiny portion (e.g. 20 examples) of your data and make sure you can achieve zero cost. For this experiment it’s also best to set regularization to zero, otherwise this can prevent you from getting zero cost

* Start training

  * Quantities that should be monitored: loss, train/val accuracy, ratio of the update magnitudes, the first-layer weights in ConvNets

* Start with small regularization and find learning rate that makes the loss go down: [1e-3, 1e-5]
    * If the loss not going down, that means the learning rate is too low; if the loss is NaN, that means a high learning rate
    * If there is a big gap between training accuracy and validation accuracy, that means overfitting (increase regularization or get more data); if there is no gap, that means you can increase your model capacity
    * If accuracy still going up, you need to train longer
* Early stop: when validation loss starts going up
    
* **Hyperparameter Optimization**

  * Learning rate, regularization, learning rate decay, model size
  * **Prefer one validation fold to cross-validation**

  * Cross-validation in stages:

    * Coarse to fine search:
      1. First stage: only a few epochs to get rough idea of what params work (wide range; 1-5 epochs)
      2. Second stage: longer running time, fine search (repeat as necessary; 10-20 epochs)

    * It is best to optimize in log space: 

      ```python
      learning_rate = 10 ** uniform(-6, 1)
      ```

    * But some parameters are instead searched in the original scale:

      ```python
      dropout = uniform(0,1)
      ```

    * Random Search is sometimes better than Grid Search

    * Make sure that the best value is not at the edge of ur searching interval

  * Track the ratio of weight updates / weight magnitudes: around 0.001

### P15

#### Optimization

* Problems of SGD:
  *  The maximum ratio between the largest and the smallest gradient can be quite large, especially in high-dimensional problems, causing zig-zag path in SGD
  * Local minima / saddle point (much more common in high dimension): zero gradient, so gradient descent get stuck
  * Our gradients come from mini batches so they can be noisy

* SGD + Momentum

  * $v_{(t+1)}=\rho v_t+\nabla f(x_t)$          $x_{t+1}=x_t-\alpha v_{t+1}$

  * ```python
    vx = 0
    while True:
    	dx = compute_gradient(x)
    	vx = rho * vx + dx
    	x -= learning_rate * vx
    ```

  * Velocity is like a running mean of gradients, rho gives " the coefficient of friction" ( [0.5, 0.9, 0.95, 0.99])

  * Velocity and gradient weighted sum (vector composition)

  * Perfect solving all those problems above

  * **SGD + Nesterov Momentum**:   

    *  $v_{t+1}=\rho v_t-\alpha \nabla f(x_t+\rho v_t)$     $x_{t+1}=x_t+v_{t+1}$

    *  $v_{t+1}=\rho v_t-\alpha \nabla f(\widetilde x_t)$                $\widetilde x_{t+1}=\widetilde x_t+v_{t+1}+\rho (v_{t+1}-v_t)$ , $\widetilde x_t=x_t+\rho v_t$

    * ```python
      dx = compute_gradient(x)
      old_v = v
      v = rho * v - learning_rate * dx
      x += -rho * old_v + (1 + rho) * v
      ```

* AdaGrad

  * ```python
    grad_squared = 0
    while True:
    	dx = compute_gradient(x)
    	grad_squared += dx * dx
    	x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
    ```

  * Adding **element-wise scaling** of the gradient based on the historical sum of squares in each dimension

  * Step-size will get smaller and smaller: good at convex problems

  * RMSProp

    * ```python
      grad_squared = 0
      while True:
      	dx = compute_gradient(x)
      	grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
      	x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
      ```

* **Adam**

  * ```python
    first_moment = 0
    second_moment = 0
    for t in range(num_iterations):
    	dx = compute_gradient(x)
      first_moment = beta1 * first_moment + (1 - beta1) * dx # Momentum
    	second_moment = beta2 * second_moment + (1 - beta2) * dx * dx # RMSProp
      first_unbias = first_moment / (1 - beta1 ** t) # t is your iteration counter going from 1 to infinity
      second_unbias = second_moment / (1 - beta2 ** t) # bias correction
    	x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-8) # AdaGrad/RMSProp
    ```

  * Adam with beta1 = 0.9, beta2 = 0.999, and learning_rate = 1e-3 or 5e-4 is a great starting point for many models

  * Adam uses **exponential weighted average (EWA) of the squared gradients and momentum**

* Learning rate
  * Learning rate decay over time:
    * Step decay: Decay learning rate by half every few epochs. One heuristic you may see in practice is to watch the validation error while training with a fixed learning rate, and reduce the learning rate by a constant (e.g. 0.5) whenever the validation error stops improving.
    * Exponential decay: $\alpha = \alpha_0e^{-kt}$, where $\alpha _0 , k$ are hyperparameters and $t$ is the iteration number (but you can also use units of epochs)
    * Cosine: $\alpha_t=\frac{1}{2}\alpha_0(1+cos(t\pi /T))$
    * Linear: $\alpha_t=\alpha_0(1-t/T)$
  * Inverse sqrt: $\alpha_t=\alpha_0/\sqrt t$
    * 1/t decay: $\alpha = \alpha_0/(1+kt)$
    * Start woth no decay, and observe the loss curve to find where need decay
  * Warm up: High initial learning rates can make loss explode; linearly increasing learning rate from 0 over the first ~5000 iterations can prevent this
  * If increase the batch size by N, also scale the initial learning rate by N
  
* First-Order Optimization: use gradient form linear approximation, and step to minimize the approximation
* Second-Order Optimization: use gradient and Hessian to form quadratic approximation, and step to the minima of the approximation
  * $\theta^* = \theta_0-H^{-1}\nabla_{\theta}J(\theta_0)$
  * No learning rate
  * Computing the Hessian is very costly
  * Quasi-Newton methods (BGFS most popular) that seek to approximate the inverse Hessian
  * L-BFGS: dose not form/store the full inverse Hessian and works well in full batch (less stochastic data)
* Model Ensembles
  * Training multiple independent models and average their results at test time
    * Same model, different initializations
    * Top models discovered during cross-validation
  * Using multiple snapshots of a single model during training
  * Using the moving average of the parameter vector at test time

### P16

#### Regularization

* L1, L2, Maxnorm

* Dropout

  * In each forward pass, randomly set some neurons to zero

  * Probability of dropping is a hyperparameter: 0.5 is common

  * ```python
    p = 0.5
    
    def train_step(X):
    
    	# forward pass for 3-layer neural network
    	H1 = np.maximum(0, np.dot(W1, X) + b1)
      U1 = np.random.rand(*H1.shape) < p # first dropout mask
      H1 *= U1 # drop!
      H2 = np.maximum(0, np.dot(W2, H1) + b2)
      U2 = np.random.rand(*H2.shape) < p # first dropout mask
      H2 *= U2 # drop!
      out = np.dot(W3, H2) + b3
      
      
    def predict(X):
      # ensembke forward pass
      H1 = np.maximum(0, no.dot(W1, X) + b1) * p
      H2 = np.maximum(0, no.dot(W2, X) + b2) * p # scale at test-time
      out = np.dot(W3, H2) + b3
    ```

  * Can be interpret as doing model ensemble within a single model: each binary mask is one model ($2^N$ models)

  * Dropout makes output random (**Training**): $y=f_W(x,z)$

  * We want to average out the randomness **at test-time**: 

    * $y=f(x)=E_z[f(x,z)]=\int p(z)f(x,z)dz$
    * Want to approximate the integral: at test time, multiply by dropout probability

* Inverted dropout

  * ```python
    p = 0.5
    
    def train_step(X):
    
    	# forward pass for 3-layer neural network
    	H1 = np.maximum(0, np.dot(W1, X) + b1)
      U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask
      H1 *= U1 # drop!
      H2 = np.maximum(0, np.dot(W2, H1) + b2)
      U2 = (np.random.rand(*H2.shape) < p) / p # first dropout mask
      H2 *= U2 # drop!
      out = np.dot(W3, H2) + b3
      
      
    def predict(X):
      # ensembke forward pass
      H1 = np.maximum(0, no.dot(W1, X) + b1) * p
      H2 = np.maximum(0, no.dot(W2, X) + b2) * p # scale at test-time
      out = np.dot(W3, H2) + b3
    ```

  * to make test-time faster

* Batch Normalization act the same: **add noise or stochasticity at training time, and average it out at test time**

* Data Augmentation (transform the data without changing its label)
  * Random crops and scales
    * Training: sample random crops / scales
    * Testing: average a fixed set of crops
  * Color jitter (randomsize contrast and brightness)
  * Random mix/combinations of: translation, rotation, streching, shearing, len distortions, ...

* DropConnect
  
* Zero out some of the valuses of the weight matrix
  
* Fractional Max Pooling

* Stochastic Depth

* Cutout/ Mixup for small classification datasets

### P17

#### Transfer Learning

* One reason of overfitting is there is not enough data

|                         | **Very similar dataset**           | **Very different dataset**                               |
| ----------------------- | ---------------------------------- | -------------------------------------------------------- |
| **Very little data**    | Use Linear Classifier on top layer | Difficult... Try linear classifier from different stages |
| **Quite a lot of data** | Finetune a few layers              | Finetune a larger number of layer                        |

### P18

#### Hardware and software 

##### GPU

* Compared to CPU: more cores, but each core is much slower, and good at doing parallel tasks: Matrix multiplication
* CUDA
* Streaming multiprocessors: FP32 cores, Tensor Core (4x4 matrix)

##### Pytorch

* Tensor

* Autograd

  * ```python
    Requires_grad = True
    ```

  * ```python
    Loss.backward()
    ```

  * Gradients are **accumulated** into w1.grad and w2.grad and the graph is destroyed

  * Set gradients to zero: 

    ```python
    w1.grad.zero_()
    w2.grad.zero_()
    ```

  * Tell pytorch not to build a graph for these operations:

    ```python
    with torch.no_grad():
      # gradient descent...
      for param in model.parameters():
        param -= learning_rate * param.grad
    ```

  * Use optimizer to update params and zero gradients:

    ```python
    optimizer.step()
    optimizer.zero_grad()
    ```

  * Can define new functions, but pytorch still creates computation graphs step by step (numerical unstable)

  * Define new autograd operators by subclassing Function, define forward and backward

    ```python
    class Sigmoid(torch.autograd.Function):
      @staticmethod
      def forward():
      
      @staticmethod
      def backward():
        
    def sigmoid(x):
      return Sigmoid.apply(x)
    ```

    * Only adds one node to the graph

* Module

  * Define Modules as a torch.nn.Module subclass

    ```python
    class TwoLayerNet(torch.nn.Module):
      def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__():
        # ...
        
      def forward(self, x):
        # ...
        
    	# no need to define backward - autograd will handle it
    ```

  * Stack multiple instances of the component in a sequential:

    ```python
    model = torch.nn.Sequential()
    ```

* Pretrain Models: torchvision

* tensorboard

* Dynamic Computation Graphs: 

  * Building the graph and computing the graph happen at the same time

  * let u use regular Python control flow during the forward pass
  * Applications: model structure that depends on the input (**RNN**)

* Static Computation Graph: 

  1. Build computational graph describing our computation
  2. Reuse the same graph on every iteration
  
  ```python
  @torch.jit.script # python function compiled to a graph when it is defined
  ```

|               | Static                                                       | Dynamic                                                      |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Optimization  | Framework can optimize the graph before it runs              | None                                                         |
| Serialization | Once the graph is built, can serialize it and run it without the code (tarin model in Python, deploy in C++) | Graph building and execution are intertwined, so always need to keep code around (RNN) |
| Debugging     | Lots of indirection - can be hard to debug, benchmark, etc   | The code u write is the code that runs. Easy to reason about, debug |

* Data parallel
  * nn.DataParallel
  * nn.DistributedDataParallel

##### TensorFlow

* TF 1.0 (Static Graphs)
  * Define computational graph
  * run rhe graph many times
* TF 2.0 (Dynamic Graphs)
  * Static: @tf.function
* Keras

### P19

#### Net Architecture

* Some indexes:

  * Memory: KB = (num of output elements) * (bytes per element) / 1024

    ​                       = (C * H' * W') * 4 [for 32-bit floating point] / 1024

  * Params: number of weights = (weight shape) + (bias shape)

    ​                                                   = C' * C * H * W + C'

  * Number of floating point operations (flop: multiply + add) 

    ​	= (number of output elements) * (ops per outpit element)

    ​	= (C' * H' * W') * (C * H * W)

##### AlexNet (2012) 

* Model split over two GPUs
* Most of the memory usage is in the early convolution layers
* Nearly all of parameters are in the fc layers
* Most floating-point ops occurs in the convolution layers

##### ZFNet (2013)

* Both the architecture of AlexNet and ZFNet are set by trials and errors (hand-design)

##### VGG (2014)

* VGG Design rules:
  * All conv are 3x3 stride 1 pad 1
  * All max pool are 2x2 stride 2
  * After pool, double #channels
* Have 5 conv stages:
  1. conv-conv-pool
  2. conv-conv-pool
  3. conv-conv-pool
  4. conv-conv-conv-[conv]-pool
  5. conv-conv-conv-[conv]-pool (VGG19 have 4 conv in stage 4 and 5)

* Two 3x3 conv has same receptive field as a single 5x5 conv, but has fewer parameters and takes less computation. Plus, two 3x3 conv can have more non-linearity by inserting ReLU
* Conv layers at each spatial resolution take the same amount of computation by doubling channels and halfing its spatial resolution

##### GoogLeNet (2014)

* Innovations for efficiency
* Stem network at the start aggressively downsamples input

* Inception module: local unit with parallel branches which is repeated many times throughout the network
  * Try all size of kernels instead of setting  it as a hyperparameter
  * Uses 1x1 bottleneck layers to reduce channel dim before expensive conv

* Use Global Average Pooling to collapse spatial dimensions, and one linear layer to produce class scores

* Auxiliary Classifiers: no longer need to use this after BN

##### ResNet (2015)

* Problem: Deeper networks perform worse than the shallow one
* Hypothesis: deeper networks are harder to optimize, and don't learn identity functions to emulate shallow models
* Solution: just copying the learned layers from the shallower model and setting additional layers to identity mapping

* Change the network so learning identity functions with extra layers is easy

* Residual block:
  * Basic block: two 3x3 conv
  * Bottleneck block: 1x1 conv + 3x3 conv + 1x1 conv
  * Pre-action block: ReLU inside residual -> can learn true identity function by setting Conv weights to zero
* Network is divided into stages like VGG: the first block of each stage halves the resolution (with stride-2 conv) and doubles the number of channels

* Uses the same aggressive stem as GoogLeNet to downsample the inout 4x before applying residual blocks
* Uses Globle Average Pooling
* Improving ResNets: ResNeXt (group comvolution)

##### Module Ensembles (2016)

##### Others 

SENet (2017)

Densely Connected Neural Networks

MoblieNets, ShuffleNet

Neural Architecture Search



### RNN: Process Sequences

* RNN on both sequntial and non-sequential data
* RNNs have an internal state that is updated as a sequence is processed
* $h_t=f_W(h_{t-1},x_t)$
* Sharing weights: using the same weight matrix W at every timestamp

#### Vanilla RNN

* $h_t=tanh(W_{hh}h_{t-1}+W_{xh}x_t)=tanh(W(\substack{h_{t-1}\\x_t}))$
* $y_t=W_{hy}h_t$
* $h_0$ is either set to all 0, or learn it

* Language Modeling
  * Encode inputs as one-hot-vector
  * Embedding layer
  
* GPT-2
  
* Forward the entire sequence to compute the loss, the backward through entire sequence to compute gradient

* Truncated Backproppagation Through Time

* Image Captioning: $h = tanh(W_{xh}*x+W_{hh}*h+W_{ih}*v)$

* Vanilla RNN Gradient Flow

  * Computing gradient to $h_0$ incolces many factors of W and repeated tanh (almost always < 1): Exploding / Vanishing gradients

  * Explodin Gradients: Gradient clipping -> scale gradient if its norm is too big

    ```python
    grad_norm = np.sum(grad * grad)
    if grad_norm > threshold:
      grad *= (threshold / grad_norm)
    ```

  *  Vanishing Gradients: change RNN architecture

#### LSTM

* $\left(\substack{i\\f\\o\\g}\right)=\left(\substack{\sigma\\\sigma\\\sigma\\tanh}\right)(W\left(\substack{h_{t-1}\\x_t}\right)+b_h)$

* $c_t=f\odot c_{t-1}+i\odot g$

  $h_t=o\odot tanh(c_t)$

* i: input gate, whether to write to cell

* f: forget gate, whether to erase cell
  
  * o: output gate, how much to reveal cell
  
  * g: gate gate, how much to write to cell
  
  * Gradient Flow: 
    * Backpropagation from $c_t$ to $c_{t-1}$ only elementwise multiplication by $f$ (between [-1,1])
    * Uninterrupted gradient flow (Similar to ResNet)
    * Allow better control of gradients values, using suitable parameter updates of the forget gate
    * Add through the f, i, g, o gates: better balancing of gradient values
    * LSTM makes it easier tfor RNN to preserve information over many time steps (if f = 1, i = 0, then the information of that cell is preserved indefinitely)
    * LSTM doesn't guarantee that there is no vanishing/exploding gradient, but it does provide an easier way for the model to learn long-distance dependencies

Multilayer RNNs

Gated Recurent Unit (GRU)

#### Sequence to Sequence: 

* Many to one (Encoder) : $h_t=f_W(x_t,h_{t-1})$
  * Initial decoder state $s_0$
  * Context vector $c$ (often $c=h_t$)
* One to many (Decoder): $s_t=g_U(y_{t-1},h_{t-1},c)$
* Input sequence bottlenecked through fixed-sized vector

#### Attention

* Alignment scores: $e_{t,i}=f_{att}(s_{t-1},h_i)$      ($f_{att}$ is an MLP)
* normalize alignment scores to get attention weights (using Softmax) : $\sum_ia_{t,i}=1, 0<a_{t,i}<1$
* Compute context vector as linear combination of hidden states: $c_t=\sum_ia_{t,i}h_i$
* Use context vector on decoders: $s_t=g_U(y_{t-1},c_t,s_{t-1})$
* Image captioning with RNNs and Attention: Each tilmestep of decoder uses a different context vector that looks at different parts of the input image
* The decoder doesn't use the fact that $h_i$ form an ordered sequence - it just treats them as an unordered set $\{h_i\}$
* X, Attend, and Y: Show, attend, and tell/read
* Attention Layer:
  * Inputs:
    1. Query vector: $Q$  (Shape: $N_Q \times D_Q$) (Multiple query vectors)
    2. Input vectors: $X$ (Shape: $N_X\times D_X$)
    3. Similarity function: scaled dot product instead of $f_{att}$ (large similarities will cause softmax to saturate and give vanishing gradients)
  * Computation:
    1. Similarities: $E=QX^T$ (Shape: $N_Q\times N_X$) $E_{i,j}=Q_i\cdot X_j/sqrt(D_Q)$
    2. Attention weights: $A=softmax(E, dim=1)$ (Shape: $N_Q\times N_X$)
    3. Output vector: $Y=AX$ (Shape: $N_Q\times D_X$) $Y_i=\sum_jA_{i,j}X_j$
  * Changes: Separate key and value
    - Key matrix: $W_K$ (Shape: $D_X\times D_Q$)
    - Value matrix: $W_V$ (Shape: $D_X \times D_V$)
    - Key vectors: $K=XW_K$ (Shape: $N_X\times D_Q$)
    - Value vectors: $V=XW_V$ (Shape: $N_X\times D_V$)
    - Similarities: $E=QK^T$ (Shape: $N_Q\times N_X$) $E_{i,j}=Q_i\cdot K_j/sqrt(D_Q)$
    - Output vector: $Y=AV$ (Shape: $N_Q\times D_X$) $Y_i=\sum_jA_{i,j}V_j$
* Self-Attention Layer:
  * one query per input vector
  * Inputs:
    1. Input vectors: $X$ (Shape: $N_X\times D_X$)
    2. Key matrix: $W_K$ (Shape: $D_X\times D_Q$)
    3. Value matrix: $W_V$ (Shape: $D_X \times D_V$)
    4. Query matrix: $W_Q$  (Shape: $D_X \times D_Q$) 
  * Computation:
    1. Query vectors: $Q=XW_Q$
    2. Key vectors: $K=XW_K$ (Shape: $N_X\times D_Q$)
    3. Value vectors: $V=XW_V$ (Shape: $N_X\times D_V$)
    4. Similarities: $E=QK^T$ (Shape: $N_Q\times N_X$) $E_{i,j}=Q_i\cdot K_j/sqrt(D_Q)$
    5. Attention weights: $A=softmax(E, dim=1)$ (Shape: $N_Q\times N_X$)
    6. Output vector: $Y=AV$ (Shape: $N_Q\times D_X$) $Y_i=\sum_jA_{i,j}V_j$
  * Self-Attention layer is Permutation Equivariant $f(s(x))=s(f(x))$
  * Self-Attention layer works on sets of vectors, compared with each others, and output sets of vectors (does not care about the order if the vector it is processing) 
  * In order to make processing position-aware, concatenate input with positional encoding
* Masked Self-Attention Layer:
  * Don't let vectors look ahead in the sequence
  * Language modeling (predict next word)
  * setting $E_{i,j}$ to -inf, where $i>j$
* Multihead Self-Attention Layer:
  * use $H$ independent Attention Heads in parallel
  * Hyperparameters: Query dimension $D_Q$, Number of heads $H$
* "Self-Attention Generative Adversarial Networks"

* Three Ways of Processing Sequences
  1. RNN
     * Works on Ordered Sequences
     * Good at long sequences: afyer one RNN layer, $h_T$ sees the whole sequence
     * Not parallelizable: need to compute hidden states sequntially
  2. 1D Convolution
     * Works on Multidimensional Grids
     * Highly parallel
     * Bad at long sequences: need to stack many conv layers for outputs to see the whole sequnce
  3. Self-Attention
     * Works on Sets of Vectors
     * Good at long sequences: after one self-attention layer, each output sees all inputs
     * Highly parallel: each output can be computed in parallel
     * Very memory intensive
* The Transformer
  * "Attention is all you need"
  * Transformer Block: 
    * input + Self-Attetion layer + residual connection + layer normalization + MLP (FC layer) independently on each vector +  residual connection + layer normalization = output
  * Self-attention is the only interaction between vectors
  * A Transformer is a sequence of transformer blocks
  * Transfer Learning:
    * Pretrain a giant Transformer model for language modeling
    * Finetune the Transformer on your own task
  * Scaling up Transformers

### Adversarial Machine Learning

* Machine learning models can be attacked by an adversary
* an example attack: Input perturbation
  * Adersary goal: find a small perturbation $\delta_i$ such that $f(x_i+\delta_i)\neq f(x_i)$ (non-targeted)
  * The perturbation is small enough that may not be noticeable to human eyes

* Adversarial input attacks
  * White-box attacks
  * Black-box attacks: Adverary can query the model, but no access its internals (weights etc)
  * For targeted attack, w]we want to reduce the distance to target label and move opposite to the gradient
* Data poisoning attacks during training
* Evasion attacks

### Visualizing and Understanding

* First Layer: edges, colors...
* Last Layer
  * Nearest neighbor
  * Dimensionality Reduction: PCA, t-SNE
* Visualizing Activations: Maximally Activating Patches
* Salient Map
  * Saliency map tells us the degree to which each pixel in the image affects the classification score for that image
  * Saliency via Occlusion: mask part of the image before feeding to CNN, check how much predicted probabilities change
  * Saliency via Backprop
* Intermediate Features via (guided) Backprop
* Class Activation Mapping
* Gradient Ascent
  * Generate a synthetic image that maximally activates a neuron
  * $I^*=argmax_If(I)+R(I)$
  * Simple regularizer: $R(I)=\lambda\parallel I\parallel^2_2$ to make images more natural
* Feature inversion
  * Given a CNN feature vector for an image, find a new image that
    1. Match the give feature veactor
    2. Look natural (image prior regularization)
* DeepDream: Amplify Existing Features
  * Choose and image and a layer in a CNN; repeat:
    1. Forward: compute activations at chosen layer
    2. Set gradient of chosen layer equal to its activation
    3. Backward: compute gradient on tmage
    4. Update image: : $I^*=argmax_I\sum_If_i(I)^2$
* Neural Texture Synthesis: Gram Matrix
* Neural Style Transfer: Feature (high-level) + Gram Reconstruction (low-level)

### Object Detection

##### Task Definition

* Input: Single RGB Image
* Output: A set of detected objects; for each object predict:
  * Category label (from fixed, known set of categories)
  * Bouding box (four numbers: x, y, width, height)

##### Challenges

* Multiple outputs
* Multiple types of output: "what" and "where"
* Large images: higher resolution for detection

##### Detecting a single object:

* Correct label: Softmax Loss
* Correct box: L2 Loss
* Multitask Loss: Weighted Sum

##### Detecting multiple object: Sliding Window

* Object or background
* Region Proposals: find a small set of boxes that are likely to cover all objects

##### R-CNN: Region-Based CNN

* Reigions of interest from a proposal method
  * Categorize each region proposal as positive, negative, or neutral based on overlap with ground-truth boxes 
  * Using IoU
* Warped image reigions (224*224)
* Forward each reigion through ConvNet:
  1. Classify each reigion: 
     * Positive boxes predict class and box offset
     * Negative boxes just predict background class
  2. Bounding box regression: predict "transform" to correct the Reigions of interest
     * Input: ($p_x, p_y, p_h, p_w$)
     * Transform: ($t_x, t_y, t_h, t_w$)
     * Output: ($b_x, b_y, b_h, p_w$)
     * Translate relative to box size: $b_x=p_x+p_wt_x$     $b_y=p_y+p_ht_y$
     * Log-space scale transform: $b_w=p_wexp(t_w)$      $b_h=p_hexp(t_h)$
* Use scores to select a subset of region proposals to output (threshold on background / per-category / take top K peoposals per image)
* Compare with ground-truth boxes
  * Intersection over Union (IoU)
  * $\frac{Area\ of\ Intersection}{Area\ of\ Union}$
  * IoU > 0.5 is decent; IoU > 0.7 is pretty good; IoU > 0.9 is almost perfect
  * Overlapping boxes: Non-Max Suppression (NMS)
    * Problem: Object detectors often output many overlapping detections
    * Solution: post-process raw detections using NMS
      1. Select next highest-scoring box
      2. Eliminate lower-scoring boxes with IoU > threshold (e.g. 0.7)
      3. If any boxes remain, GOTO 1
* Evaluating Object Detectors: Mean Average Precision (mAP)
  * Run object detector on all test images (with NMS)
  * For each category, compute Average Precision
    * For each detection (highest score to lowest score)
      1. If it matches some GT box with IoU > 0.5, mark it as positive and eliminate the GT
      2. Otherwise mark it as negtive
      3. Plot a point on PR Curve
    * Average Precision  = area under Precision vs Recall Curve
  * mAP = average of AP for each category
  * For "COCO mAP": Compute mAP@thresh for each IoU threshold (0.5, 0.55, 0.6, ..., 0.95) and take average

##### Fast R-CNN

* Add backbone network before region proposals
* Cropping and resizing features: RoI Pool / RoI Align (bilinear interpolation)
  * Must be differentiable for backprop
* Per-Reign Network is relatively lightweight 
* Problem: Runtime dominated by region proposals

##### Faster R-CNN

* Insert Region Proposal Network to predict proposals from features
  * Imagine an anchor box of fixed size at each point in the feature map
  * At each point, predict whether the corresponding anchor contains an object (per-cell logistic regression, predict scires with conv layer)
  * For positive boxes, also predict a box transform to regress from anchor box to object box
  * Problem: anchor box may have the wrong size / shape
  * Solution: use K different anchor boxes at each point
  
* Jointly train with 4 losses:
  * RPN classification
  * RPN regression
  * Object classification
  * Object regression
  
* Ignore overlapping proposals with non-max suppression

* Two-stage object detector
  * First stage: Run once per image
    * Backbone network
    * Region proposal network
  * Second stage: Run once per region
    * Crop features: RoI pool / align
    * Predict object class
    * Predict Bbox offset
  
* Single-stage object dection
  
  * Classify each object as one of C categories or background
  
  * YOLO / SSD / RetinaNet
  
* Feature Pyramid Networks

### Segmentation

##### Semantic Segmentation

Fully Convolutional Network

* Problems
  1. Effective receptive field size is linear in number of conv layers: with 3x3 conv layers, receptive field is 1 + 2L
  2. Convolution on high resolution images is expensive
* Downsampling and upsampling
  * In-Network Upsampling: "Unpooling"
    * Bed od Nails: padding with 0, copy value to up-left corner
    * Nearest Neighbor: padding with input value
    * Bilinear Interpolation
    * Bicubic Interpolation
    * Max Unpooling: place max value into remembered positions, padding with 0 
  * Learnable Upsampling: Transposed Convolution
    * Weight filter by input value and copy to output
    * Stride gives ratio between movement in output and input
    * Sum overlaps

##### Instance Segmentation

Mask R-CNN

##### Panoptic segmentation, Pose Estimation, Video Captioning

### 3D Vision

#### 3D Shape Representations

##### Depth Map:

* For each pixel, depth map gives distance from the carema to the objec in the world at that pixel

* RGB image + Depth image = RGB-D image (2.5D)
* Predicting Depth Maps using FCN
  * Fatal problem in 3D vision: scale / depth ambiguity
  * scale invariant loss

##### Surface Normals

* For each pixel, surface normals give a vector giving the normal vector to the object in the world for that pixel
* Predicting Normals
* Per-pixel Loss: $(x\cdot y)/(|x|\ |y|)$

##### Voxel Grid

* Need high spatial resolution to capture fine structures
* Processing Voxels input: 3D Convolution
* Generating Voxel Shapes
  * 3D Convolution
  * Voxel Tubes
* Voxel problems: computational expensive and large memory usage
* Oct-Trees, Nested Shape Layers

##### Implicit Surface

* Learn a function to classify arbitrary 3D points as inside / outside the shape: $R^3\rightarrow\{0,1\}$
* The surface of the 3D object is the level set $\{x:o(x)=0.5\}$
* signed distance function

##### Point Cloud

* Processing Pointcloud Inputs: PointNet
* Order of points doesn't matter (set)
* Generatong Pointcloud Outputs
  * Predicting Point Clouds: Loss Function: Chamfer distance

##### Triangle Mesh

* Predicting Meshes: Pixel2Mesh
  * Iterative mesh refinement
  * Graph Convolution: $f'_i=W_0f_i+\sum_{j\in N(i)}W_1f_j$
  * Vertex-Aligned Features
  * Loss Function: Chamfer distance between predicted samples and ground-truth samples

#### Shape Comparison Metrics

* Voxel IoU
* Chamfer Distance
* F1 Score:
  * Precision@t = fraction of predicted points within t of some ground-truth point
  * Recall@t = fraction of ground-truth points with t of some predicted point
  * F1@t = $2*\frac{precision@t*Recall@t}{Precision@t+Recall@t}$
  * Robust to outliers

#### Camera Systems

* Canonical Coordinates vs View Coordinates

#### Datasets

* ShapeNet: 3D CAD models
* Pix3D: 3D models of IKEA furniture

#### 3D Shape Prediction

Mesh R-CNN

* Input image -> 2D object recognition -> 3D object voxels -> 3D object meshes
* Chamber Loss + L2 regularization

### Video

#### Dataset

* Video Classification: UCF101, Sports-1M, YouTubw 8M
* Atomic Actions: Charades, Atomic Visual Actions (AVA), Moments in Time (MIT)
* Video Retrival (Movie Querying): M-VAD, MPII-MD, LSMDC

#### Challenges

* Computaionally expensive (size of video >> image datasets, redundancy in adjacent frames)
* Lower quality (Resolution, motion blur, occlusion)
* Requires a lot of training data

#### Video framework

* Sequence modeling
* Temporal reasoning (receptive filed)
* Focus on action recognition (representative task for video understanding)

#### Pre-Deep Learning

Features:

* Local features (hand-crafted): HOF + HOF
* Trajectory-based:
  * Motion Boundary Histogram (MBH)
  * (impoved) dense trajectories: good performance, but computationally intensive 

Ways to aggregate features:

* Bag of Visual Words
* Fisher vectors

Measuring Motion: Optical Flow

* Highlights local motion

* give a displacement field F between images $I_t$ and $I_{t+1}$ : $F(x,y)=(dx,dy)$
* Trajectory stacking

#### Deep Learning

##### Main models:

* CNN + RNN: video understanding as sequence modeling
* 3D Convolution: embed temporal dimension to CNN
* Two-stream networks: explicit model of motion

Problem: raw videos are big

Solution: 

* Train on short clips: low fps and low spatial resolution
* Run model on different clips, average predictions

##### Single-Frame CNN

* Train normal 2D CNN to classify video frames independently
* Average prediction at test-time
* Strong baseline for video classification

##### Late Fusion (with FC layers)

* Intuition: Get high-level appearance of each frame, and combine them
* Run 2D CNN on each frame, concatenate features and feed to MLP

##### Late Fusion (with pooling)

* Intuition: Get high-level appearance of each frame, and combine them
* Run 2D CNN on each frame, pool features and feed to Linear
* Problem: hard to compare low-level motion between frames

##### Early Fusion

* Intuition: Compare frames with vary first Conv Layer, after that normal 2D CNN
* Input: T x 3 x H x W
* Reshape: 3T x H x W
* First Conv Layer: D x H x W (collapse all temporal information)
* Problem: 
  * One layer of temporal processing may not be enough
  * No temporal shift-invariance -> needs to learn separate filters for the same motion at different times in the clip

##### Slow Fusion (3D CNN)

* Intuition: Use 3D convolution and pooling to slowly fuse temporal information over the course of the network
* Build slowly in space and time
* Temporal shift-invariant since each filter slides over time
* C3D: The VGG of 3D CNNs
* I3D: Inflating 2D networks to 3D
  * Input: 2D conv / pool kernel $K_h\times K_w$
  * Output: 3D conv / pool kernel $K_t\times K_h\times K_w$ by copying $K_t$ times in space and divide by $K_t$

##### Two stream

* Video = Appearance + Motion (Complementary)
* Separate motion (multi-frame -> optical flow) from static appearance (single frame)
* Motion: external + camera -> mean subtraction to compensate camera motion
* Problem:
  * The appearance and motion are not aligned -> spatial fusion
  * Lacking modeling of temporal evolution -> temporal fusion

##### CNN + RNN

Modeling long-term temporal structure

* Extract feature with CNN (2D or 3D) : 
  * Each value a function of a fixed temporal window (local temporal strcuture)

* process local features using RNN (LSTM)
  * Each vector is a function of all previous vectors (global temporal structure)
* Sometimes don't backdrop to CNN to save memory, pretrain and use CNN as a feature extractor
* Multi-layer RNN
  * Features from layer $L$, timestep $t-1$
  * Features from layer $L-1$, timestep $t$
  * Use different weights at each layer, share weights across time
* Recurrent CNN: replace dot product in RNN with convolution

##### Spatio-Temporal Self-Attention (Nonlocal block)

* Interpret as a set of THW vectors of dim C
* Trick: initialize last conv to 0, then entire block computes identity, so the nonlocal block can be inserted into existing 3D CNNs

##### SlowFast Network

* Treating time and space differently

### Generative Models

#### Supervised vs Unsupervised Leaning

* Supervised learning: learning a function to map x -> y (classification, regression, object detection, semantic segmentation, image captioning)

* Unsupervised learning: learn some underlying hidden  structure of the data (clustering, dimensionality reduction, feature learning, density estimation)

#### Discriminative vs Generative Models

* Discriminative Model: 
  * Learn a probability distribution $p(y|x)$
  * The possible labels for each input compete for probability mass
  * Can not handle unreasonable inputs, it must give label distributions for all images
  * Applications:
    * Assign labels to data
    * Feature learning (with labels)
* Generative Model: 
  * Learn a probability distribution $p(x)$
  * All possible images compete with each other for probability mass
  * Model can reject unreasonable inputs by assigning them small values
  * Applications:
    * Detect outliers
    * Feature learning (without labels)
    * Sample to generate new data
* Conditional Generative Model: learn $p(x|y)$
  * Each possible label induces a competition among all images
  * $P(x|y)=\frac{P(y|x)}{P(y)}P(x)$
  * Applications:
    * Assign labels, while rejecting outliers
    * Generate new data conditioned on input labels

#### Generative Modeling

* Given training data, generate new sample from same distribution
* Formulate as density estimation problems:
  * Explicit density estimation: explicitly define and solve for $p_{model}(x)$
  * Implict density estimation: learn model that can sample from $p_{model}(x)$ without explicitly defining it
* https://deepgenerativemodels.github.io/notes/

#### Autoregressive models (explicit density estimation)

* Goal: write down an explict function for $p(x)=f(x,W)$
* Fully visible belief network
* Tarin by maximize probability of training data: 

  * $W^*=argmax_W\prod_ip(x^{(i)})$ (Maximum likelihood estimation)
  * $W^*=argmax_W\sum_ilog\ p(x^{(i)})$ (Log trick to exchange product for sum)
  * $W^*=argmax_W\sum_ilog\ f(x^{(i)},W)$ (Train with gradient descent)
* $p(x)=\prod^T_{t=1}p(x_t|x_i, ...,x_{t-1})$ -> RNN
* PixelRNN

  * Generate image pixels one at a time, strating at the upper left corner

  * $h_{x,y}=f(h_{x-1,y},h_{x,y-1},W)$
  * Drawback: sequential generation is slow in both training and inference
* PixelCNN

#### Variational Autoencoders

* Intractable density: optimize lower bound
* Autoencoders (regular)
  * Use the features to reconstruct the input data with a decoder
  * Loss function: $||\hat x-x||^2_2$
  * Features need to be lower dimensional than the input data: to catch meaningful factors of variation in data
  * Encoder can be used to initialize a supervised model
  * No way to sample new data from learned ones 
* Probabilistic spin
  1. Learn latent features z from raw data
  2. Sample from the model to generate new data
     * Assume simple prior p(z), e.g. Gaussian
     * Represent p(x|z) with a neural network (similar to decoder)
* Independet assumption: pixels are conditional independent given the latent features
* Train
  * Maximize likelihood of data $p_\theta(x)=\frac{p_\theta(x|z)p_\theta(z)}{p_\theta(z|x)}$
  * Train decoder that inputs latent code z, gives distribution over data x $p_\theta(x|z)=N(\mu_{x|z},\sum_{x|z})$
  * Train another network (encoder) that inputs data x, gives distribution over latent codes z $q_\phi(z|x)=N(\mu_{z|x},\sum_{z|x})$
  * If we can ensure that $q_\phi(z|x)\approx p_\theta(z|x)$, the we can approximate $p_\theta(x)\approx\frac{p_\theta(x|z)p_\theta(z)}{q_\phi(z|x)}$
  * Jointly train both encoder and decoder to maximize the variational lower bound on data likelihood $log\ p_\theta(x)\geq E_{Z\sim q_\phi(z|x)}[log\ p_\theta(x|z)]-D_{KL}(q_\phi(z|x),p(z))$
    1. Run input data through encoder to get a distribution over latent codes
    2. Encoder output should match the prior p(z)
    3. Sample code z from encoder output
    4. run sampled code through decoder to get a distribution over data samples
    5. Original input data should be likely under the distribution output from (4)
* Generating data
  1. Sample z from prior p(z)
  2. run sampled z through decoder to get distribution over data x
  3. sample from distribution in (2) to generate data
* Edit images: modify some dimensions of sampled z
* Combining VAE + Autoregressive: VQ-VAE2

#### Generative Adversarial Network

* Idea: introduce a latent variable $z$ with simple prior $p(z)$, sample $z \sim p(z)$ and pass to a Generator Network $x = G(z)$, then $x$ is a sample from the Generator distribution $p_G$, we want $p_G=p_{data}$
  * Train Generator Network G to convert z into fake data x sampled from $p_G$ by fooling the discriminator D
  * Train Discriminator Network D to classify data as real or fake (0/1)

* jointly train generator G and discriminator D with a minimax game:

  $\ \ \ \ min_Gmax_D(E_{x\sim p_{data}}[log\ D(x)]+E_{z\sim p(z)}[log\ (1-D(G(z)))])\\=min_Gmax_DV(G,D)$

  * Using alternative gradient updates

    For t in 1, ..., T:

    1. Update D: $D=D+\alpha_D\frac{\partial V}{\partial D}$ (Gradient ascent)
    2. Update G: $G=G-\alpha_G\frac{\partial V}{\partial G}$ (Gradient descent)

  * Problem: Vanishing gradients for G at the beginning

  * Solution: train G to maximize $-log\ (D(G(z)))$ (maximize likelihood of discriminator to being wrong)

  * The minimax game achieves its global minimum when $p_G=p_{data}$

### Reinforcement Learning

#### Environment and Agent

* The agent sees a **state**; may be noisy or incomplete
* The agent makes an **action** based on what it sees
* **Reward** tells the agent hwo well it is doing
* Action causes change to environment
* Agent updates

#### Reinforcement Learning vs Supervised Learning

* Stochasticity: Rewards and state transitions maybe random
* Credit assignment: Reward $r_t$ may not directly depend on action $a_t$
* Nondifferentiable: Can't backprop through world; can't compute $dr_t/da_t$ 
* Nonstationary: What the agent experiences depends on how it acts

##### Markov Decision Process (MDP)

* A tuple $(S,A,R,P,\gamma)$
  * $S$ : Set of possible states
  * $A$ : Set of possible actions
  * $R$ : Distribution of reward given (state, action) pair
  * $P$ : Transition probability: distribution over next state given (state, action)
  * $\gamma$ : Discount factor (tradeoff between future and present rewards)
* Agent executes a policy $\pi$ giving distribution of actions conditioned on states (a function from S to A that specifies what action to take in each state)
* Objective: Find policy $\pi^*$ that maximize cumulative discounted rewards: $\sum_t\gamma ^tr_t$
* Training process
  * At time step t = 0, environment samples initial state $s_0\sim p(s_0)$
  * Then, for f = 0 until done:
    * Agent selects action $a_t\sim \pi(a|s_t)$
    * Environment samples reward $r_t\sim R(r|s_t, a_t)$
    * Environment samples next state $s_{t+1}\sim P(s|s_t, a_t)$
    * Agent receives reward $r_t$ and next state $s_{t+1}$
* GridWorld
* Problem: Lots of randomness
* Solution: Maximize the expected sum of rewards $\pi^*=arg\ max_\pi E[\sum_{t\geq0}\gamma^tr_t|\pi]$

* Value Function and Q Function
  * Value Function: $V^\pi(s)=E[\sum_{t\geq0}\gamma^tr_t|s_0=s,\pi]$
  * Q Function: $Q^\pi(s,a)=E[\sum_{t\geq0}\gamma^tr_t|s_0=s,a_0=a,\pi]$

#### Q-learning

##### Bellman Equation

* Optimal Q-function: $Q^*(s,a)=max_\pi E[\sum_{t\geq0}\gamma^tr_t|s_0=s,a_0=a,\pi]$
* $Q^*$ Encodes the optimal policy: $\pi^*(s)=arg \ max_{a'}Q(s,a')$
* Bellman Equation: $Q^*(s,a)=E_{r,s'}[r+\gamma max_{a'}Q^*(s',a')]$ , where $r\sim R(s,a), s'\sim P(s,a)$

##### Value Iteration

* If we find a function $Q(s,a)$ that satisfies the Bellman Equation, then it must be $Q^*$
* Start with a random Q, and use the Bellman Equation as an update rule:
  * $Q_{i+1}(s,a)=E_{r,s'}[r+\gamma max_{a'}Q^*(s',a')]$ , where $r\sim R(s,a), s'\sim P(s,a)$
* $Q_i$ converges to $Q^*$ as $i\rightarrow\infty$
* Problem: Not scalable. Must compute $Q(s,a)$ for every state-action pair
* Solution: Use a function approximator to estimate $Q(s,a)$ -> neural network

##### Deep Q-Learning

* train a neural network (with weights $\theta$) to approximate $Q^*$ : $Q^*(s,a)\approx Q(s,a;\theta)$
* $y_{s,a,\theta}=E_{r,s'}[r+\gamma max_{a'}Q^*(s',a';\theta)]$ , where $r\sim R(s,a), s'\sim P(s,a)$
* Loss for training $Q$ : $L(s,a)=(Q(s,a;\theta)-y_{s,a;\theta})^2$

##### Policy Gradient

* Train a network $\pi_\theta(a|s)$ that takes state as input, gives distribution over which action to take in that state
* objective function: $J(\theta)=E_{r\sim p_\theta}[\sum_{t\geq0}\gamma^tr_t]$
* Find the optimal policy by maximizing $\theta^*=arg\ max_\theta J(\theta)$ (using gradient ascent)
* Problem: Nondifferentiability and suffer from high variance so requires a lot of samples
* $\frac{\partial J}{\partial \theta}=E_{x\sim p_\theta}[\sum_{t\geq 0}\frac{\partial}{\partial \theta}log\pi_\theta(a_t|s_t)]$
* SOTA: Proximal Policy Optimization

(Soft) Actor-Critic, Model-Based, Imitation Learning, Inverse Reinforcement Learning, Adveisarial Learning, Stochastic Computation Graphs

### Recap

#### Prediction

1. We will discover interesting new types of deep models
   * Neural ODE: $\frac{dh}{dt}=f(h(t),t,\theta)$
2. Deep Learning will find new applications
   * Deep Learning for Computer Science / Mathematics
3. Deep Learning will use more data and compute
   * New hardware for Deep Learning

#### Problem

1. Models are biased
2. Need new theory
3. Deep Learning needs a lot of labeled training data (self-supervised learning)
4. Deep Learning doesn't "Understand" the world