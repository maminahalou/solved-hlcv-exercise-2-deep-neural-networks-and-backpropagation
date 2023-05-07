Download Link: https://assignmentchef.com/product/solved-hlcv-exercise-2-deep-neural-networks-and-backpropagation
<br>
Deep neural networks have shown staggering performances in various learning tasks, including computer vision, natural language processing, and sound processing. They have made the model designing more flexible by enabling end-to-end training.

In this exercise, we get to have a first hands-on experience with neural network training. Many frameworks (<em>e.g. </em>PyTorch, Tensorflow, Caffe) allow easy usage of deep neural networks without precise knowledge on the inner workings of backpropagation and gradient descent algorithms. While these are very useful tools, it is important to get a good understanding of how to implement basic network training from scratch, before using this libraries to speed up the process. For this purpose we will implement a simple two-layer neural network and its training algorithm based on back-propagation using only basic matrix operations in questions 1 to 3. In question 4, we will use a popular deep learning library, PyTorch, to do the same and understand the advantanges offered in using such tools.

As a benchmark to test our models, we consider an image classification task using the widely used CIFAR-10 dataset. This dataset consists of 50000 training images of 32×32 resolution with 10 object classes, namely airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The task is to code and train a parametrised model for classifying those images. This involves

<ul>

 <li>Implementing the feedforward model (Question 1).</li>

 <li>Implementing the backpropagation algorithm (gradient computation) (Question 2).</li>

 <li>Training the model using stochastic gradient descent and improving the model training with better hyperparameters (Question 3).</li>

 <li>Using the PyTorch Library to implement the above and experiment with deeper networks (Question 4).</li>

</ul>

<h2>Question 1: Implementing the feedforward model</h2>

In this question we will implement a two-layered a neural network architecture as well as the loss function to train it. Starting from the main file ex2 FCnet.py, complete the required code in the two layernet.py to complete this question. Refer to the comments in the code to the exact places where you need to fill in the code.

<strong>Model architecture </strong>Our architecture is shown in Fig.1. It has an input layer, and two model layers – a hidden and an output layer. We start with randomly generated toy inputs of 4 dimensions and number of classes <em>K </em>= 3 to build our model in q1 and q2 and in q3 use images from CIFAR-10 dataset to test our model on a real-world task. Hence input layer is 4 dimensional for now.

In the hidden layer, there are 10 units. The input layer and the hidden layer are connected via linear weighting matrix <em>W</em><sup>(1) </sup>∈ R<sup>10×4 </sup>and the bias term <em>b</em><sup>(1) </sup>∈ R<sup>10</sup>. The parameters <em>W</em><sup>(1) </sup>and <em>b</em><sup>(1) </sup>are to be learnt later on. A linear operation is performed, <em>W</em><sup>(1)</sup><em>x </em>+ <em>b</em><sup>(1)</sup>, resulting in a 10 dimensional vector <em>z</em><sup>(2)</sup>. It is then followed by a

Figure 1: Visualisation of the two layer fully connected network, used in Q1-Q3

relu non-linear activation <em>φ</em>, applied element-wise on each unit, resulting in the activations <em>a</em><sup>(2) </sup>= <em>φ</em>(<em>z</em><sup>(2)</sup>). Relu function has the following form:

(1)

A similar linear operation is performed on <em>a</em><sup>(2)</sup>, resulting in <em>z</em><sup>(3) </sup>= <em>W</em><sup>(2)</sup><em>a</em><sup>(2) </sup>+ <em>b</em><sup>(2)</sup>, where <em>W</em><sup>(2) </sup>∈ R<sup>3×10 </sup>and <em>b</em><sup>(2) </sup>∈ R<sup>3</sup>; it is followed by the softmax activation to result in <em>a</em><sup>(3) </sup>= <em>ψ</em>(<em>z</em><sup>(3)</sup>). The softmax function is defined by:

(2)

<table width="643">

 <tbody>

  <tr>

   <td width="441">The final functional form of our model is thus defined by</td>

   <td width="202"></td>

  </tr>

  <tr>

   <td width="441"><em>a</em>(1) = <em>x</em></td>

   <td width="202">(3)</td>

  </tr>

  <tr>

   <td width="441"><em>z</em>(2) = <em>W</em>(1)<em>a</em>(1) + <em>b</em>(1)</td>

   <td width="202">(4)</td>

  </tr>

  <tr>

   <td width="441"><em>a</em>(2) = <em>φ</em>(<em>z</em>(2))</td>

   <td width="202">(5)</td>

  </tr>

  <tr>

   <td width="441"><em>z</em>(3) = <em>W</em>(2)<em>a</em>(2) + <em>b</em>(2)</td>

   <td width="202">(6)</td>

  </tr>

  <tr>

   <td width="441"><em>f</em><em>θ</em>(<em>x</em>) := <em>a</em>(3) = <em>ψ</em>(<em>z</em>(3))</td>

   <td width="202">(7)</td>

  </tr>

 </tbody>

</table>

which takes a flattened 4 dimensional vector as input and outputs a 3 dimensional vector, each entry in the output <em>f<sub>k</sub></em>(<em>x</em>) representing the probability of image <em>x </em>corresponding to the class <em>k</em>. We summarily indicate all the network parameters by <em>θ </em>= (<em>W</em><sup>(1)</sup><em>,b</em><sup>(1)</sup><em>,W</em><sup>(2)</sup><em>,b</em><sup>(2)</sup>).

<strong>Implementation      </strong>We are now ready to implement the feedforward neural network.

<ol start="7">

 <li>Implement the code in two layenet.py for the feedforward model. You are required to implement Eq.3 to 7. Verify that the scores you generate for the toy inputs match the correct scores given in the ex2 FCnet.py. (4 points)</li>

 <li>We later guide the neural network parameters <em>θ </em>= (<em>W</em><sup>(1)</sup><em>,b</em><sup>(1)</sup><em>,W</em><sup>(2)</sup><em>,b</em><sup>(2)</sup>) to fit to the given data and label pairs. We do so by minimising the loss function. A popular choice of the loss function for training neural network for multi-class classification is the cross-entropy loss. For a single input sample <em>x<sub>i</sub></em>, with label <em>y<sub>i</sub></em>, the loss function is defined as :</li>

</ol>

)                          (8)

(9)

(10)

(11)

Averaging over the whole training set, we get

(12)

where <em>K </em>is the number of classes. Note that if the model has perfectly fitted to the data (<em>i.e. f<sub>θ</sub><sup>k</sup></em>(<em>x<sub>i</sub></em>) = 1 whenever <em>x<sub>i </sub></em>belongs to class <em>k </em>and 0 otherwise), then <em>J </em>attains the minimum of 0.

Apart from trying to correctly predict the lable, we have to prevent overfitting the model to the current training data. This is done by encoding our prior belief that the correct model should be simple (Occam’s razor); we add an <em>L</em><sub>2 </sub>regularisation term over the model parameters <em>θ</em>. Specifically, the loss function is defined by:

(13)

where  is the squared <em>L</em><sub>2 </sub>norm. For example,

(14)

By changing the value of <em>λ </em>it is possible to give weights to your prior belief on the degree of simplicity (regularity) of the true model.

Implement the final loss function in two layernet.py and let it return the loss value. Verify the code by running and matching the output cost 1<em>.</em>30378789133. (4 points)

<ol>

 <li>c) To be able to train the above model on large datsets, with larger layer widths, the code has to be very efficient. To do this you should avoid using any python <em>for </em>loops in the forward pass and instead use matrix/ vector multiplication routines in the numpy library. If you have written the code of parts (a) and (b) using loops, convert it to vectorized version using numpy operations (2 points).</li>

</ol>

<h2>Question 2: Backpropagation</h2>

We train the model by solving min <em>J</em>˜(<em>θ</em>)                      (15) <em>θ</em>

via stochastic gradient descent. We therefore need an efficient computation of the gradients ∇<em><sub>θ</sub>J</em>˜(<em>θ</em>). We use backpropagation of top layer error signals to the parameters <em>θ </em>at different layers.

In this question, you will be required to implement the backpropagation algorithm yourself from a pseudocode. We will give a high-level description of what is happening at each line.

For those who are interested in the robust derivation of the algorithm, we include the optional exercise on the derivation of backpropagation algorithm. A prior knowledge on standard vector calculus including the chain rule would be helpful.

<strong>Backpropagation </strong>Backpropagation algorithm is simply a sequential application of chain rule. It is applicable to any (sub-) differentiable model that is a composition of simple building blocks. In this exercise, we focus on the architecture with stacked layers of linear transformation + relu non-linear activation.

The intuition behind backpropagation algorithm is as follows. Given a training example (<em>x,y</em>), we first run the feedforward to compute all the activations throughout the network, including the output value of the model <em>f<sub>θ</sub></em>(<em>x</em>) and the loss <em>J</em>. Then, for each parameter in the model we want to compute the effect that parameter has on the loss. This is done by computing the derivatives of the loss w.r.t each model parameter.

Backpropagation algorithm is performed from the top of the network (loss layer) towards the bottom. It sequentially computes the gradient of the loss function with respect to each layer activations and parameters.

Let’s start by deriving the gradients of the un-regularized loss function w.r.t final layer activations <em>z</em><sup>(3)</sup>. We will then use this in the chain rule to compute analytical experssions for gradients of all the model parameters.

<ol>

 <li>Verify that the loss function defined in Eq.12 has the gradient w.r.t <em>z</em><sup>(3) </sup>as below.</li>

</ol>

(16)

where ∆ is a matrix of <em>N </em>× <em>K </em>dimensions with

(

1<em>,      </em>if <em>y<sub>i </sub></em>= <em>j</em>

∆<em><sub>ij </sub></em>=                                                                     (17)

0<em>,    </em>otherwise




<ol start="2">

 <li>To compute the effect of the weight matrix <em>W</em><sup>(2) </sup>on the loss in Eq.12 incurred by the network, we compute the partial derivatives of the loss function with respect to <em>W</em><sup>2</sup>. This is done by applying the chain rule. Verify that the partial derivative of the loss w.r.t <em>W</em><sup>(2) </sup>is</li>

</ol>

(18)

(19) Similary, verify that the regularized loss in Eq.13 has the derivatives

(20)




<ol>

 <li>We can repeatedly apply chain rule as discussed above to obtain the derivatives of the loss with respect to all the parameters of the model <em>θ </em>= (<em>W</em><sup>(1)</sup><em>,b</em><sup>(1)</sup><em>,W</em><sup>(2)</sup><em>,b</em><sup>(2)</sup>). Dervive the expressions for the derivatives of the regularized loss in Eq.13 w.r.t <em>W</em><sup>(1)</sup>, <em>b</em><sup>(1)</sup>, <em>b</em><sup>(2) </sup> (<em>report</em>, 6 points)</li>

 <li>Using the expressions you obtained for the derivatives of the loss w.r.t model parameters, implement the back-propogation algorithm in the file two layernet.py. Run the ex2 FCnet.py and verify that the gradients you obtained are correct using numerical gradients (already implemented in the code). The maximum relative error between the gradients you compute and the numerical gradients should be less than 1e-8 for all parameters. (5 points)</li>

</ol>

<h2>Question 3: Stochastic gradient descent training</h2>

We have implemented the backpropagation algorithm for computing the parameter gradients and have verified that it indeed gives the correct gradient. We are now ready to train the network. We solve Eq.15 with the stochastic gradient descent.

<strong>Stochastic gradient descent (SGD) </strong>Typically neural networks are large and are trained with millions of data points. It is thus often infeasible to compute the gradient ∇<em><sub>θ</sub>J</em>˜(<em>θ</em>) that requires the accumulation of the gradient over the entire training set. Stochastic gradient descent addresses this problem by simply accumulating the gradient over a small random subset of the training samples (minibatch) at each iteration. Specifically, the algorithm is as follows

<table width="642">

 <tbody>

  <tr>

   <td width="642"><strong>Data: </strong>Training data {(<em>x<sub>i</sub>,y<sub>i</sub></em>)}<em><sub>i</sub></em><sub>=1<em>,</em>···<em>,N</em></sub>, initial network parameter <em>θ</em><sup>(0)</sup>, regularisation hyperparameter <em>λ</em>, learning rate <em>α</em>, batch size <em>B</em>, iteration limit <em>T </em><strong>Result: </strong>Trained parameter <em>θ</em><sup>(<em>T</em>) </sup><strong>for </strong><em>t </em>= 1<em>,</em>··· <em>,T </em><strong>do</strong>a random subset of the original training set                           ;);<em>θ</em>(<em>t</em>) ← <em>θ</em>(<em>t</em>−1) + <em>v</em>; <strong>end</strong></td>

  </tr>

 </tbody>

</table>

<strong>1</strong>

<strong>2</strong>

<strong>3</strong>

<strong>4</strong>

<strong>5</strong>

<strong>Algorithm 1: </strong>Stochastic gradient descent with momentum

where the gradient) is computed only on the current randomly sampled batch.

Intuitively, <em>v </em>= −∇<em><sub>θ</sub>J</em><sup>˜</sup>(<em>θ</em><sup>(<em>t</em>−1)</sup>) gives the direction to which the loss <em>J</em><sup>˜ </sup>decreases the most (locally), and therefore we follow that direction by updating the parameters towards that direction <em>θ</em><sup>(<em>t</em>) </sup>= <em>θ</em><sup>(<em>t</em>−1) </sup>+ <em>v</em>.

<ol>

 <li>Implement the stochastic gradient descent algorithm in two layernet.py and run the training on the toy data. Your model should be able to obtain loss ≤ 0.02 on the training set and the training curve should look similar to the one shown in figure 2. (3 points)</li>

</ol>

Figure 2: Example training curve on the toy dataset.

Figure 3: Example images from the CIFAR-10 dataset

<ol start="3072">

 <li>We are now ready to train our model on real image dataset. For this we will use the CIFAR-10 dataset. Since the images are of size 32×32 pixels with 3 color channels, this gives us 3072 input layer units, represented by a vector <em>x </em>∈ R<sup>3072</sup>. See figure 3 for example images from the dataset. The code to load the data and train the model is provided with some default hyper-parameters in ex2 FCnet.py. With default hyperparametres, if previous questions have been done correctly, you should get validation set accuracy of about 29%. This is very poor. Your task is to debug the model training and come up with beter hyper-parameters to improve the performance on the validation set. Visualize the training and validation performance curves to help with this analysis. There are several pointers provided in the comments in the ex2 FCnet.py to help you understand why the network might be underperforming (Line 224-250). Once you have tuned your hyper parameters, and get validation accuracy greater than 48% run your best model on the test set once and report the performance.</li>

</ol>

<h2>Question 4: Implement multi-layer perceptron using PyTorch library</h2>

So far we have implemented a two-layer network by explicitly writing down the expressions for forward, backward computations and training algorithms using simple matrix multiplication primitives from numpy library.

However there are many libraries available designed make experimenting with neural networks faster, by abstracting away the details into re-usable modules. One such popular opensource library is PyTorch (https://pytorch.org/). In this final question we will use PyTorch library to implement the same two-layer network we did before and

train it on the Cifar-10 dataset. However, extending a two-layer network to a three or four layered one is a matter of changing two-three lines of code using PyTorch. We will take advantage of this to experiment with deeper networks to improve the performance on the CIFAR-10 classification. This question is based on the file ex2 pytorch.py

To install the pytorch library follow the instruction in https://pytorch.org/get-started/locally/ . If you have access to a Graphics Processing Unit (GPU), you can install the gpu verison and run the exercise on GPU for faster run times. If not, you can install the cpu version (select cuda version None) and run on the cpu. Having gpu access is not necessary to complete the exercise. There are good tutorials for getting started with pytorch on their website (https://pytorch.org/tutorials/).

<ol>

 <li>Complete the code to implement a multi-layer perceptron network in the class <em>MultiLayerPerceptron </em>in ex2 pytorch.py. This includes instantiating the required layers from nn and writing the code for forward pass. Intially you should write the code for the same two-layer network we have seen before.</li>

 <li>Complete the code to train the network. Make use of the loss function nn.CrossEntropyLoss to compute the loss and loss.backward() to compute the gradients. Once gradients are computed, optimizer.step() can be invoked to update the model. Your should be able to achieve similar performance (<em>&gt; </em>48% accuracy on the validation set) as in Q3. Report the final validation accuracy you achieve with a two-layer network.</li>

</ol>

Now that you can train the two layer network to achieve reasonable performance, try increasing the network depth to see if you can improve the performance. Experiment with networks of atleast 2, 3, 4, and 5 layers, of your chosen configuration. Report the training and validataion accuracies for these models and discuss your observations. Run the evaluation on the test set with your best model and report the test accuracy.