## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I will use deep neural networks and convolutional neural networks to classify traffic signs. I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out your model on images of German traffic signs that you find on the web.

In this project I am using tensorFlow, openCV, and other python packages to train and implement the deep neural network. The trained neural network is able to identify the German Traffic Sign Dataset with an accuracy of at least 93%.

This project contain four files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown 
* saved models

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Dependencies
---
This project requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

Instruction for running the code:
---
You can retrain the neural network by running the jupyter notebook after you set the dependencies right. The hyperparameters in the neural network, L2 regularization, and dropout are free for you to change to seek further improvement. You can also change the size of the neural network. (The current neural net has two covnets and three fully connected nets.)
