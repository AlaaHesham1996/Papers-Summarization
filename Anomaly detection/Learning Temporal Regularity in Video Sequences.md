# Learning Temporal Regularity in Video Sequences

In this document I will be sharing the summary of the following paper : Learning Temporal Regularity in Video Sequences [[1]](https://arxiv.org/abs/1604.04574)

The summary covers :

* Their final goal/ what they are trying to achieve
* The main two approaches they used as their methodlogy
* Datasets used for training and testing
* Capabilites of their learned model
* Discussing the two approaches used in the methodology in more details namely:
  * Input Preprocessing
  * Model Architecture
  * Activation
  * Objective Function and its Optimization , Network Initialization.
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

    ![i1](/imgs/LTRVS/1.JPG)
    

* 

*









    
    


    
    




 

    
