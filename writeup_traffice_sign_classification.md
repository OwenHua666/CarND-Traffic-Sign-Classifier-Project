## Traffic Sign Recognition 

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

[image1]: ./Writeup_Images/image1.JPG "Visualization1"
[image2]: ./Writeup_Images/image2.JPG "Visualization2"
[image3]: ./Writeup_Images/image3.JPG "Visualization3"
[image4]: ./Writeup_Images/image4.JPG "Visualization4"
[image5]: ./Writeup_Images/image5.JPG "Visualization5"
[image6]: ./examples/grayscale.jpg "Grayscaling"
[image7]: ./Writeup_Images/image7.JPG "Visualization5"
[image8]: ./Writeup_Images/image8.JPG "Visualization5"
[image9]: ./Writeup_Images/image9.JPG "Visualization5"
[image10]: ./Writeup_Images/image10.JPG "Visualization5"
[image11]: ./Writeup_Images/image11.JPG "Visualization5"


### Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/OwenHua666/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 by 32 pixels
* The number of unique classes/labels in the data set is 43

The example of each class is displayed here:

![alt text][image1]
![alt text][image2]

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the number of traffic sign image per class.

![alt text][image3]
![alt text][image4]
![alt text][image5]

The distributions of the three datasets are approximately the same. There are classes which have more data samples than the other.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to try to convert the images to grayscale because it requires less training compute. However, the grayscale image has less feature for the neural network to learn. After tested both methods, the color image behave slightly better than the grayscale image.

As a second step, I used gaussian filter to remove the image noise in the image.
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image6]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and a processed image:

![alt text][image7]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16     					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| flatten, outputs 400     									|
| Fully connected		| flatten, outputs 120    									|
| RELU					|												|
| dropout					|												|
| Fully connected		| flatten, outputs 84     									|
| RELU					|												|
| dropout					|												|
| Softmax				| cross_entropy      									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used Adam Optimized in tensorflow. The batch size is 128. There are 15 epochs in the training process. The learning rate is 0.001. The keep probability of dropout is 70%. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.959
* test set accuracy of 0.940

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture used is the basic leNet without any dropout and L2-regularization. I chose it because it behaves well at classifying images like written numbers. I also think it is a good start point to do iterative improvement. 
* What were some problems with the initial architecture?
The biggest problem is the validation accuracy never exceeds 85%. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I believe the low accuracy is because of overfitting. So I add dropout layer to the fully connected layers and use L2 regularization to fight against overfitting.
* Which parameters were tuned? How were they adjusted and why?
I turned the epochs, learning rate, dropout keep probability, and L2 regularization factor. They make a huge role in learning speed and anti-overfitting. For the learning rate, I started with 0.005 and iteratively train the neural net with an increment of 0.0001 until it becomes too large for the model to learn. For dropout keep probability (0, 1], I started with 0.5 and use binary to adjust. For the L2 regularization factor, I stated with 0.0001 and increased it by 0.0001 for each iterative step to find the factor giving the biggest accuracy. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The convolution layer helps the neural work to see the feature beyond a single pixel. When it convolves, the neural net sees the relation of one pixel with its neighbors. The dropout layer helps a lot with fighting against overfitting, only the useful feature for classifying each image survives. 

If a well known architecture was chosen:
* What architecture was chosen?
LeNet was chosen.
* Why did you believe it would be relevant to the traffic sign application?
There are often number digits, letters, and special pattern on the traffic signs. These patterns also have a clear contrast to their background like the input images to the LeNet have.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The validation accuracy is 95.9% and test set accuracy is 94.0%, which are not bad.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8]

Some of the test images are hard to classify because the sign is tilted and distorted.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| slippery road      		| Right-of-way at the next intersection  									| 
| Stop sign     			| Stop sign 										|
| Road work					| Road work											|
| Pedestrian					| Speed limit (70km/h)											|
| 60 speed limit (distorted)	      		| Priority road				 				|
| 60 speed limit			| Speed limit (70km/h)     							|
| Pedestrian (not German Traffic Sign			| Speed limit (80km/h)	      							|
| turn right ahead			| Speed limit (80km/h)      							|


The model was able to correctly guess 2 of the 8 traffic signs, which gives an accuracy of 80%. This result is worse than the test set accuracy. To increase the accuracy, I think I should add augmented image to the training dataset because the images downloaded from web are oriented in a different way. I also need to do more image processing to clean the input images from web like rotation and scaling. After the input data becomes cleaner, the next step is to use a more powerful architecture like the architecture described in [Going Deeper with Convolutions](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

![alt text][image9]
![alt text][image10]
![alt text][image11]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


