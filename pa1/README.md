[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ENWZafzJ)
# PA1

* CS484, Programming Assignment 1
* Spring 2025
* Due: February 3, 2025 at 11:59PM

Note: you know everything you need to know when this assignment is assigned. You can finish this much earlier than the deadline if you want to.

## Purpose

This assignment will give you the opportunity to practice and learn
1. to modify machine learning code
2. to run experiments and interpret/graph the results
3. working with git and Python classes

## Overview 

In this assignment you will use the book's provided perceptron code to create a one vs. rest multiclass Perceptron, and train it on the provided data. You will need to slightly modify the Perceptron code, but mainly focus on designing your program to accommodate one vs. rest learning before training/testing. You will then interpret your results in the markdown file provided.

## Dataset

The dataset you will be working with is a wine dataset that has chemical analysis of different wines, with the goal of classifying what cultivar the wine was derived from. It has 13 features, and the class has 3 values. The dataset is included in your repository; you can use python to learn what the features are, and get a sense of what they look like.

Two of the features are on a vastly different scale than the others. You should drop those features as one of the first steps in your code.


## Requirements

We learned in class the idea of one vs. rest as one of the two ways to implement a multiclass classifier. Create a regular python program (e.g. not a Jupyter notebook) that uses the Perceptron class from the book to be able to do multi-class prediction on the provided dataset. Remember that you will need to put the Perceptron class in its own file and import it into your file to be able to create Perceptron objects.

After creating your new multi-class perceptron code, you will need to use this code to try to classify these three classes. Your code must do the following:

* Use the data provided
* Split data into appropriate sets using the correct sklearn function
* Train your new multi-class Perceptron using the appopriate data. Output the number of errors that occurred on each epoch for each model learned as a graph, and output the final learned bias and weights of each model in your multi-class Perceptron. 
* Test your new multi-class Perceptron using the appropriate data on the set of models that worked best based on accuracy.
* Output the final confusion matrix and the classification report for the test data.


### Programming Expectations 

* Your code should NOT be in a Jupyter notebook, but in a .py file.
* You should use pandas for initial processing of data
* Be careful about how you deal with the data for each model you need to learn. Labels should be either 0/1 or -1/1 when doing your learning.
* Code should have the book's Perceptron class in its own py file.
* Code should be well designed python code, such as using functions as appropriate. Your code should do all of the required steps.
* Code should be well commented. This includes comments at the top of the file stating your name, the assignment, and the purpose of the file; comments at the start of a function stating purpose, parameters, and return values; and comments at each step/group of the code.
* The primary python file should be able to be run once and do everything requested by the assignment. Choice of best model must be decided by the code itself, not set by the programmer. 

* Your code must run with the version of python installed on jhub, as that is where it will be graded. There are instructions on Moodle on how to run your code on the command line on jhub. Or, you can install the same versions on your own system. 


### Using git effectively
* commit and push your code as you work, not just at the end. Every time you step away from the project, or get something working, you should commit and push that code so that it's saved as a version in your repository. That way you will never lose your work, and can always return to an earlier version if you change your mind. The instructor will not have sympathy if you do not follow these steps and then lose your work.
* Write useful git commit messages, and commit as you work on your program so that you have a history of your work if you make a mistake and need to return to it. Because we have git to help with this aspect of coding, the instructor will have little sympathy for losing or overwriting your files as proper use of git should prevent that issue.
* Do not upload via the github website. Commit/push/pull from your IDE or the command line only.

### Tips

This type of problem is best solved using iterative development. For example, ignore the training of different hyperparameters at first and just get it working with one setup. Even that part can be broken up into different steps. Then once that is working, determine how to expand it. Once once the process works, think about how to use test data.

Well designed code will not need to be particularly long to solve this problem. If you find yourself copy/pasting lines of code frequently, that is a sign that you are not using good functional design. A good design will make testing your code and completing the project easier.

The book gives you a very strong hint on how to implement multi-class perceptrons. 


# Final Interpretation and sharing of results

Update the interpretation.md file to answer the questions posed there about what you did and what you learned.

