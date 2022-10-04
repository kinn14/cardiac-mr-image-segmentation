# cardiac-mr-image-segmentation
Semantic segmentation of magnetic resonance (MR) images of heart is performed using deep learning.


# Introduction
The task we are attempting is to segment cardiovascular magnetic resonance (CMR) images into four regions: the background, left ventricle, myocardium and right ventricle. We’ll be using semantic segmentation to do this, which assigns a label to all the pixels in an image, where pixels that share characteristics will have the same label. To perform this task we will be constructing a convolutional neural network (CNN). The dataset we are using is a modified version from the ACDC1 challenge. It contains 200 CMR images and corresponding segmented masks. The dataset will be split 50%, 10% and 40% for training, validation and testing respectively. During this experiment, we will to explore a variety of CNN architectures and hyperparameters in order to implement a neural network which can accurately segment cardiovascular MR images.

# Implementation  <sub>2</sub>

The model currently employed is the U-net architecture. U-net is a fully convolutional neural network with a symmetric architecture, hence the U-shape. It consists of two major parts — the left part is called contracting path, which is constituted by the general convolutional process; the right part is an expansive path, which is constituted by transposed 2D convolutional layers. In the contracting path, the network learns important features and in the expansive path tries to enhance the resolution of the image to bring it to the original size of the image. In the contracting path U-net uses max pooling layers to decrease the size of the image and in the expansive path it replaces the max pooling layers with deconvolution layers to amplify small features. The same idea is also adopted in DeepLab to enhance the resolution <sub>3</sub>. The fully connected layers are also removed so that small changes in regions where the main object occurs does not affect the prediction.
The main reason for going forward with this architecture was due to the fact that U-net has a major role in medical image segmentation, which coincides with our task. The only difference in our proposed architecture is that we add padding in every convolution layer unlike the original network where the borders are cropped.
In our network we apply two 3 x 3 convolutions (padding =1) each succeeded by a ReLU. Then we apply 2 x 2 maxpooling operation with a stride of 2. After every maxpooling the feature channels are doubled. The above combination is applied 4 times until we reach an image of dimensions 6 x 6 x 1024 after which we apply a convolution + ReLU twice. In the expansive path we upsample the data and we concatenated with the corresponding feature map. Then again, we apply two 3 x 3 convolutions each succeeded by ReLU, hence obtaining a symmetric architecture. The feature maps are halved every time we upsample the data. The expansive step is applied until we reach a feature map of size 96 x 96 x 64. Finally, we apply a 1 x 1 convolution that maps each feature vector to the 4 classes. The below figure illustrates our architecture:





<figure>
<center>
![image](https://user-images.githubusercontent.com/10370198/193909590-bb2c2754-57e3-427e-8e10-6402c795a314.png)
</figure>

Initially we research a variety of CNNs, looking at both academic papers and online articles, in order to get a broad understanding of them and how their different architectures may impact the result of the model. From our research and implementations, we found that architectures which have many layers such as DeconvNet and Deeplab both produced bad results primarily due to the fact we have a small dataset, which leads to overfitting while training the model. Despite initial concerns about lack of computational power, we found that SegNet produced good results. However, we decided to use UNet as it produced the best results from our initial experiments, and our research found that it is the best architecture to use when working with low volumes of data <sub>4</sub>.

##Loss functions considered:
The main idea of all loss functions and optimizers is to determine the point where the function is the minimum. However, most algorithms take the assumption that the function is convex. Below we detail the different loss functions we considered using for training. 
###Lovasz Softmax:
Lovasz Softmax is based on the idea of Lovasz hinge and this method assumes that we are dealing with a convex function. As a result, if the function is not convex, we would not be able to find the global minima and may get stuck in a local minimum.
###Dice Loss:
Dice loss uses the Dice coefficent to calculate the overlap between two images. It is computed using the following formula:

$Dice (A, B)= \frac{2|A \cap B|}{|A| + |B|} $

*where A and B are images.*
###Focal Loss:
Focal loss builds on standard cross-entropy loss, such that loss can be calculated using:
$FL(p$<sub>t</sub>$)=(1-p)$<sup>$γ$</sup>$log(p$<sub>t</sub>$)$

Adding the factor $(1-p)$<sup>$γ$</sup> reduces the weighting of well classified examples. This means that misclassfied examples have a greater impact of the overall loss.

###Boundary Loss:
Boundary loss uses distance metric on space contours rather than segmentation regions that other losses such as Dice or Cross-entropy use. Using integrals over the interface of regions enables it to work well on highly unbalanced problems.

##Loss function implemented:
###Cross-Entropy:
Cross entropy is based on the concept of entropy in information theory and is different from KL Divergence (i.e Kullback Leibler Divergence), which calculates the relative entropy between two probability distributions. Cross-entropy loss deals with finding the total entropy between the distributions. This is represented by taking the negative log-likelihood where the increased negative value indicates a higher loss. We have intuitively understood that in order to really evaluate which classes (3 channels) of the image data are correctly or not correctly predicted, a loss function that deals with categorical data would be the best choice. 

##Optimizers considered: 

###Stochastic Gradient Descent:
SGD randomly picks one data point from the whole data set at each iteration to reduce the computations enormously.
It is also common to sample a small number of data points instead of just one point at each step and that is called “mini-batch” gradient descent. Mini-batch tries to strike a balance between the goodness of gradient descent and speed of SGD.

### AdaGrad:
Adagrad is an adaptive learning rate method. In Adagrad we adapt the learning rate to the parameters. We perform larger updates for infrequent parameters and smaller updates for frequent parameters. It is well suited when we have sparse data as in large scale neural networks. In Adagrad, we use different learning rates for every set of parameters for each time step.

##Optimizer implemented : 
###Adam:
Adam is another method that calculates the individual adaptive learning rate for each parameter from estimates of first and second moments of the gradients.It also reduces the radically diminishing learning rates of Adagrad. Adam can be viewed as a combination of Adagrad, which works well on sparse gradients and RMSprop which works well in online and nonstationary settings. It implements the exponential moving average of the gradients to scale the learning rate instead of a simple average as in Adagrad. It keeps an exponentially decaying average of past gradients. Adam is computationally efficient and has very little memory requirement. Adam optimizer is one of the most popular gradient descent optimization algorithms.
Hyper-parameters β1, β2 ∈ [0, 1] control the exponential decay rates of these first and second moments.




