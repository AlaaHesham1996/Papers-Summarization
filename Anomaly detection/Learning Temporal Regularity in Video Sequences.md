# Learning Temporal Regularity in Video Sequences

In this document I will be sharing the summary of the following paper : Learning Temporal Regularity in Video Sequences [[1]](https://arxiv.org/abs/1604.04574)

## The summary covers :

* Their final goal/ what they are trying to achieve
* The main two approaches they used as their methodlogy
* Datasets used for training and testing
* Capabilites of their learned model
* Discussing the two approaches used in the methodology in more details namely:
  * Input Preprocessing
  * Model Architecture
  * Activation
  * Objective Function and its Optimization , Network Initialization.
* Training Specifications
* Regularity Score as a metric for Quantitive Analysis
* Experiments
* Paper Contribution:
  * Model Generalizability
  * Synthesizing the regular frame from the irregular one
  * Predicting the regular past and the future

* Anomalous Event detection as an application

* Results

* Conclusion

## Their final goal/ what they are trying to achieve

Their goal is to learn regular patterns inside videos and use this in various applications such as Anomalous Event Detection. As opposed to learning irregular events as the definition of such events is ill i.e., visually unbounded.

## Methodology

They have approached their problem in two ways :
First, leveraging the conventional handcrafted spatio-temporal local features and learning a  fully connected autoencoder on them.

Second, building end-to-end learning  framework that is based on fully conventional feed-forward autoencoder. This approach has yielded better results than the first one. 

## Capabilites of their learned model

1. Detecting  different temporally regular patterns in videos and hence identifing  the irregular ones

2. Synthesizing the most regular frame from a video
3. Delineating objects involved in irregular motions
4. Predicting the past and the future regular motions from a single frame.
5. Learning the low level motion features for their proposed method using a fully convolutional autoencoder.

## Discussing the two approaches used in the methodology in more details namely

* ### First approach details

  * Input
  
    As an input to the autoencoder , they used handcrafted motion features that consist of Histogram of  Oriented Gradients (HOG)  and Histograms of Optical Flows (HOF)  with improved trajectory features . The 204 dimensional HOG+HOF feature are fed to the encoder and decoder sequentially.

  * Model Architecture  

    The model has four hidden layers with 2,000, 1,000, 500, and 30 neurons respectively, whereas the decoder has three hidden layers with 500, 1,000 and 2,000 neurons respectively.

    ![i1](/Anomaly%20detection/imgs/LTRVS/1.JPG)

    **[Justification Alert]** The small-sized middle layers are for learning compact semantics as well as reducing noisy information.

  * Activation

    Since the magnitude of the input and reconstructed signals should fall in the range of 0 and 1. They have used sigmoid or hyperbolic tangent (tanh) as the activation function instead of the rectified linear unit (ReLU). **[Justification Alert]** ReLU is not suitable for a network that has large receptive fields for each neuron as the sum of the inputs to a neuron can become very large.

  * Network Initialization

    To prevent the large input problem, they used the sparse weight initialization technique for the large receptive field.  

  * Objective Function

    The Euclidean loss of input feature (x_i) and the reconstructed feature (f_W (x_i)) with an L2 regularization term. The goal is to minimize the overall reconstruction cost.

    ![i1](/Anomaly%20detection/imgs/LTRVS/2.JPG)

    Where N is the mini batch size, γ is a hyper-parameter to balance the loss and the regularization and fW a neural network associated with its weights W and works as a nonlinear classifier.

* ### Second approach details

  As mentioned earlier, they have built an end-to-end learning framework that is based on fully conventional feed-forward autoencoder.**[Justification Alert]** They used fully convolutional network as fully connected layers loses spatial information which is vital for reconstructing the input frames.

  * Input

    It takes short video clips in a temporal sliding window as the input.  This raises a question which is  how big this sliding window could be .

  * Input Preprocessing
    * Size of sliding window

      As the time increases, the training loss takes more iterations to converge. The main reason the input size increases, it is more likely to meet more irregularity and hence making the process to learn the regularity harder. The main advantage is that the learned model is capable of distinguishing better between the regular and irregular events.

    ![i1](/Anomaly%20detection/imgs/LTRVS/4.JPG)

    * Data Augmentation in the Temporal Dimension

      Due to the large number of Autoencoder parameters, we need to perform data augmentation in the time dimension. This means concatenating frames with various skipping stride cuboids to construct T-sized input cuboid. Whereas in stride-2 and stride-3 cuboids, we skip one and two frames, respectively. The stride used for sampling cuboids is two frames.

  * Model Architecture

    The following figure represents the architecture of fully convolutional autoencoder.

    ![i1](/Anomaly%20detection/imgs/LTRVS/3.JPG)

    They used three convolutional layers and two pooling layers on the encoder side and three deconvolutional layers and two unpooling layers on the decoder side by considering.
  
  * Activation
  
    There are two types of pooling operations : max’ and ‘average’.**[Justification Alert]** They used ‘max’ for translation invariance. It is usually helpful to classify images by making convolutional filter output to be spatially invariant. The downside is that the spatial information is lost, which is important for location specific regularity.  They designed the unpooling layers in the deconvolution network to do the reverse operation of pooling and reconstruct the original size of activations.

  * Objective Function and its Optimization

    It is the same as the one used in the first approach except that F_W (•) is a non-linear classifier - a fully convolutional deconvolutional neural network with its weights W.

    ![i1](/Anomaly%20detection/imgs/LTRVS/2.JPG)

    To optimize the loss function, they used a stochastic gradient descent with Adagrad which is an adaptive subgradient method. **[Justification Alert]** Adagrad determines learning rate in a dimension-wise way that is adaptive to the rate of gradients by a function of all previous updates on each dimension.  Adagrad also guarantees convergance . They also tested Adam and RMSProp but AdaGrad performed better.

    ![i1](/Anomaly%20detection/imgs/LTRVS/5.JPG)

  * Network Initialization

    Regarding the initialization, they used the Xavier over the Gaussian one. **[Justification Alert]** As, unlike the Gaussian, the Xavier initialization adapts the initialization scale based on the number of input and output neurons which ensures keeping the signal in a reasonable range. This is important as if the signal is too small or too large, then the signal shrinks or become too large to be useful respectively.


## Training Specifications

  For the training, they started with a learning rate of 0.001 and reduced it when the training loss stops decreasing. For the first approach, Autoencoder on the improved trajectory features, they use mini batches of size 1024 and weight decay of 0.0005. For the fully convolutional autoencoder, they use mini batch size of 32 and choose 0.01 to be the start value of the learning rate. 

## Regularity Score as a metric for Quantitive Analysis

  For every pixel intensity value in the frame, calculate the reconstruction error as the following formula:

  ![i1](/Anomaly%20detection/imgs/LTRVS/6.JPG)

  Where f_w is the learned model. Then, sum up all the reconstruction errors of the pixels to calculate the construction error for the whole frame. Finally, determine the value of the regularity score s(t) of a frame t by the following formula :
  
  ![i1](/Anomaly%20detection/imgs/LTRVS/7.JPG)

## Experiments

They trained the model using multiple datasets. The overall videos’ duration is 1h, 15 min. **All experiments are on NVIDIA Tesla K80 GPUs**.For qualitative analysis, they produce the most regular frame from a video and use heatmaps to visualize the pixel-level irregularity. For quantitative analysis, they temporally segment the anomalous events and calculate the correct detection /False Alarm.

## Paper Contribution

* Model Generalizability

  They have tested the generalizability of the model by testing its performance in different training scenarios in terms of regularity score attained by the model.
  For each target dataset, the model is tested on it but trained in three different ways: once on the target dataset itself, another time on all datasets, finally on all datasets except for the target dataset. The results of these different training paradigms are marked in blue, red, and yellow respectively in the following figure. 

  The results are similar to each other despite different training setups .That shows the ability of the model to generalize and how it is balanced between overfitting and underfitting.
  
  ![i1](/Anomaly%20detection/imgs/LTRVS/8.JPG)

* Synthesizing the regular frame from the irregular one

    They synthesized the regular frame from the irregular one using the temporal dimension. For each pixel, they look for the time instant where its reconstruction cost was minimum. In the following figure, the left image represents the irregular motion. The middle image indicates the regular one and the right image refers to regularity score. Blue represents high regularity score and red represents low.
    They also synthesize the most regular frame in a similar manner.

    ![i1](/Anomaly%20detection/imgs/LTRVS/9.JPG)

* Predicting the regular past and the future

    They formed a temporal cube that contains only a single irregular image that is also a center frame,  then zero-padded all other frames, and finally passed it through the trained model to extrapolate the near past and future of that frame . The result of this experiment was that objects involved in an irregular motion start appearing from the past and disappearing in the future indicating that the network can predict the regular motion of the objects.

## Applications of the network

It is possible to use the local minima existed in the regularity scores to indicate the presence of anomalous events. However, not all of these minima are meaningful. To filter the meaningful ones, they used persistence 1D algorithm along with a fixed window of 50 frames in the temporal dimension to group nearby expanded local minima regions. If two local minima are separated by less than 50 frames , then they belong to the same abnormal event .

## Results

Their model outperforms or performs comparably to the state-of-the-art abnormal event detection methods but with a few more false alarms.

The following figure Shows Regularity score of each frame of three video sequences. (Top) Subway Exit, (Bottom-Left) Avenue, and (Bottom-Right) Subway Enter datasets. Green and red colors represent regular and irregular frames respectively.

![i1](/Anomaly%20detection/imgs/LTRVS/11.JPG)

The results are in a quantitative way with its comparison to the state-of-the-art abnormal event detection methods but with a few more false alarms :

![i1](/Anomaly%20detection/imgs/LTRVS/10.JPG)

## Conclusion

The paper aims at learning the regular patterns in video sequences. In a single learning framework , they built an end to end conventional autoencoder-based model  . The model shows generalizablility across multiple datasets . They analysized the results in a qualitiative ways such as : Synthesizing a regular frame from the irregular one ,predicting the regular past and the future from a single image. To perform quantitative analysis, they compared the ability of their model to detect anomalous events with state-of-the-art methods and performed comparably to them.

















    
    


    
    




 

    
