# **Traffic Sign Recognition** 

## Writeup 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualizations/exploratory.png "exploratory"
[image2]: ./visualizations/unprocessed_distributions.png "unprocessed_distribution of all traffic signs"
[image3]: ./visualizations/all_traffic_signs.png "all_traffic_signs"
[image4]: ./visualizations/augmented_images.png "sample of augmented_images"
[image5]: ./visualizations/processed_distributions.png "processed_distributions of all traffic signs"
[image6]: ./visualizations/comfirmed_new_dataset.png "make sure all in the right places"
[image7_1]: ./visualizations/lenet_1_doe1.png 
[image7_2]: ./visualizations/lenet_1_doe2.png 
[image7_3]: ./visualizations/lenet_1_doe3.png 

[image8_1]: ./visualizations/lenet_2_doe1.png 
[image8_2]: ./visualizations/lenet_2_doe2.png 
[image8_3]: ./visualizations/lenet_2_doe3.png 
[image8_4]: ./visualizations/lenet_2_doe4.png 

[image9_1]: ./visualizations/lenet_3_doe1_new.png 
[image9_2]: ./visualizations/lenet_3_doe2_new.png
[image9_3]: ./visualizations/lenet_3_doe3_new.png 
[image9_4]: ./visualizations/lenet_3_doe4_new.png 
[image9_5]: ./visualizations/lenet_3_doe5.png 

[image8]: ./visualizations/12_traffic_signs.png "Traffic Sign 12"
[image9]: ./visualizations/placeholder.png "Traffic Sign 4"
[image10]: ./visualizations/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/demigecko/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used basic python commands to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?  
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

![alt text][image1]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the dataset. It is a bar chart showing how the data distributed. One way to improve accuracy and avoid having overfitting is to use data augmentation to increase the dataset first. The max count is 2010 and the minimum count is 180.

![alt text][image2]

I also prepared a dictionary to see all the signs v.s. images in grayscale. This graph will come handy when we get to the end of the problem set. 

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

From the last session, the number of some traffic signs is not sufficient (i.e. max: 2010 and min: 180). Moreover, after many times of playing with the LeNet 5-layer architecture, I noticed the following: the result can easily fall into the overfitting situation that the *validation accuracy* is constantly lower than the *training accuracy*. therefore I took three approaches: (1) to enlarge the dataset by augmenting the existing images. Having a large dataset is always the best option to overcome the overfitting, (2) to use grayscale images instead of color ones because the RGB images as input were proven to not as good as grayscale images (explained in the later session), and (3) to introduce the dropout in the layers. All these actions are to reduce the overfitting issue and obtain high validation/testing accuracy.  

#### Increase the database: Data Augmention

From the last session, the number of some traffic sign images is not sufficient (i.e. max: 2010 and min: 180). Moreover, after many times of playing with the LeNet 5-layer Architecture, I noticed the following: the result can easily fall into the overfitting situation that the *validation accuracy* is constantly lower than the *training accuracy*. Therefore I took three approaches: (1) to enlarge the dataset by augmenting the existing images. Having a large dataset is always the best option to overcome the overfitting, (2) to use grayscale images instead of color ones because using the RGB images as input is proven to be worse than using the grayscale images (explained in the later session), and (3) to introduce the dropout within layers. All these actions are to reduce the overfitting issue and obtain high validation/testing accuracy.  

The image below is the outcome randomly generated from my code. Depends on the number of existing images of each class (traffic sign), my code will generate the images up to 2010.   
![alt text][image4] 
Another look after data augmention.
![alt text][image5] 
To examine if my code performs properly, I picked one category of the traffic signs and randomly produced 15 images for confirmation. 
![alt text][image6]

#### A. RGB image (32, 32, 3)


After (1) dataset boost by augmentation, I decided to combine (1) image normalization, and (3) introduction of dropout. I chose these three as my *Design of Experiments* (DOEs). Intuitively,  image normalization would help to reduce the fitting bias and the dropout to reduce the variance, in other words, it suppresses those small weights to be zero. Here is the dropout condition is set the *keep_prob*  be either 1 or 0.5.

To simplfy my DOE (design of experiments). Here is my table:  

| DOE     |  image      | Dataset Boost     | Normalized     | Dropout     | Training Accuracy     | Validation Accuracy     | Overfitting?     |
|:---:    |:-------:    |:-------------:    |:----------:    |:-------:    |:-----------------:    |:-------------------:    |:------------:    |
|  1      |  32x32x3    |       v           |      x         |    x        |        0.991        |  0.916            |          Yes                |
|  2      | 32x32x3     |       v           |      v         |    x        |        0.981        | 0.746            |          Yes                 |
|  3      | 32x32x3     |       v           |      v         |    v        |        0.926          | 0.772          |          Yes                 |

Graphs below are the accuracy converging plots for each DOE condition. 

![alt text][image7_1]
![alt text][image7_2]
![alt text][image7_3]

With this simple trials, here are my observations: 
(1) *Dataset boost* shows the smallest overfitting  (the value between Training Accuracy and Validation Accuracy) among all three DoEs. The Training Accuracy reaching 99.1%. This implies the boost of the dataset is effective. However, (2) the normalized images showed the worse *Validation Accuracy* while having the good *Training Accuracy*, which is the question I have. Last, (3) Dropout can help to close the gap between Training Accuracy and Validation Accuracy. 

#### B. Use R G B + gray (32, 32, 4) 
The intention to add one more layer of a grayscale image as the input data is to ensure all the information is fully utilized. Therefore, one individual image will have 4 channels. I would like to see if this approach would boost performance or not. I have a DOE plan shown below. 
| DOE     |  image      | Dataset Boost     | Normalized     | Dropout     | Training Accuracy     | Validation Accuracy     | Overfitting?     |
|:---:    |:-------:    |:-------------:    |:----------:    |:-------:    |:-----------------:    |:-------------------:    |:------------:    |
|  1      | 32x32x4     |       v           |      x         |    x        |       0.983           |        0.881            |      No          |
|  2      | 32x32x4     |       v           |      v         |    x        |       0.983           |        0.741            |      Yes         |
|  3      | 32x32x4     |       v           |      v         |    v        |       0.927           |        0.791            |      Yes         |
|  4      | 32x32x4     |       v           |      x         |    v        |       0.966           |        0.930            |      No          |

![alt text][image8_1]
![alt text][image8_2]
![alt text][image8_3]
![alt text][image8_4]

Among all 4 DOSs, the best performance is the one *without image normalization, and with dropout* (**keep_prob=0.5**). In this case, the training accuracy keeps increasing, therefore it will perform better if I increase the size of EPOCHS.  Moreover, I like the fact overfitting is dramatically reduced.  It is one of the potential candidates for my final approaches. On the other hand, the same observation is made that the *normalized RGB input* and *normalized RGB+ gray input* are not the best option to start with. To be clarified

#### C.  Use Gray image only (32, 32, 1)


On the contrary to the 4 channel case, I would like to know if we squeeze the information into one layer, how it performs. 
Below is my DOE table.
| DOE     |  image      | Dataset Boost     | Normalized     | Dropout     | Training Accuracy     | Validation Accuracy     | Overfitting?     |
|:---:    |:-------:    |:-------------:    |:----------:    |:-------:    |:-----------------:    |:-------------------:    |:------------:    |
|  1      | 32x32x1     |       v           |      x         |    x        |       0.994                |         0.938               |       a little          |
|  2      | 32x32x1     |       v           |      v         |    x        |       0.990                |          0.791              |        Yes          |
|  3      | 32x32x1     |       v           |      x         |    v        |       0.976                |          0.923              |        No          |
|  4      | 32x32x1     |       v           |      v         |    v        |        0.919               |          0.799              |        Yes          |
|  5      | 32x32x1     |       x           |      x         |    v        |        0.989               |          0.921              |        a little         |

 ![alt text][image9_1]
 ![alt text][image9_2]
 ![alt text][image9_3]
 ![alt text][image9_4]
 ![alt text][image9_5]

Input images without any normalization performs better than those with normalization. Garylevel with the boost of the dataset should be able to meet the requirement of this project as well as with the dropout; The performance of DOE 3 can be improved if I increase the EPOCHS and dropout probability. Surprisingly, grayscale images as the input seems to be the winner. Compare to the previous cases in sessions A and B, I found this single layer input is the most effective way to reach high *Validation Accuracy*. I speculated that the grayscale images carry the most critical information and it seems that losing some linkage of different color layers is a good thing, which implies color could bring more noise to such deep leaning. Therefore, I would like to downselect my input format to be a grayscale image without normalization. Then I will try to tune the EPOCHS and Dropout keep_prob to reach a better performance. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 (RGB), and  32x32x4 (RGB+Gray) , and 32x32x1 (Gray)  							| 
| Convolution 5x5     	| 1x1 stride , padding =valid, outputs 28x28x6 	|
| RELU					|Activation												|
| Max pooling	      	| 2x2 stride,  padding =valid, outputs 5x5x16 				|
| Flatten                 | 400 
| Fully connected	    |  1x1 stride, padding= valid, outputs 120      									|
| RELU				| Activation.        									|
| **Dropout**						| keep_prob>0.5												|
| Fully connected		    	| input= 120 Output=84												|
| RELU                | Activation.                                            |
| **Dropout**                        | keep_prob>0.5                                                |
| Fully connected                | input= 84 Output=42                    


I tried to modify a lot of parameters *in* or *between* layers, but later I realizesd that dropout is the most efficient way to deal with overfitting and achieve high accuracy 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train this model, I set some LeNet interlayer-connection parameters as global parameters, so I can output a graph of the validation accuracy directly. Here is the snapshot of how I did it. However, after some trials, I gave up on tunning those interlayer-connection parameters, because I realized the usefulness of the dropout that we don't need to change the size of layers or the number of layers. I downselected my approaches and DOEs at the very beginning of this report. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

All the discussion has been made in session A, B, and C. Please go to those sessions. In short, I downselected my input format to be a grayscale image without normalization. Then I tried to tune the EPOCHS and keep_prob (for the Dropout) to reach a better performance. 

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
All the discussion has been made in session A, B, and C. Please go to those sessions. 

Initially, I was hesitated to enlarge the database, therefore I was based on my understanding of CNN to perceive how a computer sees things. I tried the BRG image,  RGB+Gray images, grayscale-only image along with the normalization, but these cannot bring up the accuracy much. Later I tried the dropout with a grayscale image, finally, the validation accuracy is higher than 93%, but marginally meet the requirement. therefore the last step for me is to increase the database. What a long journey! 

* What were some problems with the initial architecture?
All the discussion has been made in session A, B, and C. Please go to those sessions. in short, I stated below: 
1. overfitting and unable to meet the 93% requirement 
2. image normalization is a bad choice for sure.  
3. the noise coming from color images. (my speculation)  

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

This concept I get it clearly, my report is all about how to overcome the overfittng. 

* Which parameters were tuned? How were they adjusted and why?

grayscale image without normalization. 
I add two dropout layers in LeNet 5. Because LeNet is the only CNN I know of, so I only do what is available so far. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Because CNN is good for searching features, and it doesn't matter where the location of the features in an image. In LeNet there are 90 or 120 interlayer parameters, too many parameters to induce the overfitting, so the concept of dropout layer is like adopting the L1 (Lasso) and L2 (Ridge) in Regression to me. 

If a well known architecture was chosen:
* What architecture was chosen?

I primarily only use LeNet and added two dropout layer in the full-connection layer 

* Why did you believe it would be relevant to the traffic sign application?

Because the traffic signs have very similar shape, i.e. triangular or circular. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


