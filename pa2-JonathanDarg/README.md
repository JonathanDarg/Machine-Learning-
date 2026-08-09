[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/mZZLDsW9)
# PA2

* CS484
* Spring 2025
* Due: Friday 3/14 at 11:59PM

Note: you know everything you need to know when this assignment is assigned. You can finish this much earlier than the deadline if you want to. You are being given more than 2 weeks to account for the LOA.

## Purpose

This assignment will give you the opportunity to practice and learn
1. pre-processing data with pandas
2. Using the full machine learning process with gridsearch and pipelines
3. Using Jupyter notebooks and markdown effectively

## Overview 

In this assignment you will use a Jupyter notebook to develop an analysis of Maryland traffic accidents to try to predict the severity of an accident based on the included features. You will need to prepare the data using the types of techniques we have covered this semester, and train models using grid search. 

You will use a Juptyer notebook for all of your data preparation steps. 

You will use a regular python script for your grid search and results analysis.


## Dataset and Dataset Preparation

The dataset is provided to you, and is real data on traffic accidents in Maryland. The goal is to try to predict the severity of an accident. You can see a summary of the features in the datasummary.md file.

There are many issues in this dataset that you will need to deal with to prepare your data for machine learning; this dataset is not straightforward to train a model for prediction. 

All of your processing must be in a Jupyter notebook. Throughout your notebook you *must* use properly formatted markdown to make it clear what is being done and why. For example, you should use headers for different sections, put markdown after an analysis noting what you plan to do next because of it, and label why you are making certain decisions. You can see examples of good uses of markdown in prior homework notebooks and some in class examples. You should show all steps needed to determine what processing to do, and then show the processing. Without markdown, the highest grade you can earn on this assignment is a U, as reasoning through your pre-processing is a core aspect of this assignment; don't skip it.

After you have prepared the dataset, save the training and testing data each into their csv file. You can use panda's `to_csv` method. Note: you should set `index=False` in that method to keep from saving the index as an unnecessary column.


## Training

After pre-processing and preparing your data for machine learning, you need to write a regular python script (.py file) to train models using two different pipelines and a grid searches with k-fold cross validation. You should vary in your grid searches either how you deal with potential class imbalance, or what supervised learning algorithm you use. You must vary the hyperparameters between them, and from examples we've used in class.

You should make good decisions on the setup based on what you know so far about the machine learning process and how to try to learn a good model. Your grid searches should use a good choice of machine learning algorithm of the ones we have learned before spring break, with at least 20 parameter combinations tried in each. You should use a verbose level of at least 2.

Your final model should be the one that's best of the two best models created, and your code should figure that out automatically. You should test your final model and output the expected statistics on the test data to the screen. You should also write to the screen information about the best models from each search including how well they worked on the training data, as well as which model was chosen as the best one.


### Programming Expectations 

* You should use pandas for all initial processing of data
* Code should be well designed python code, such as using functions as appropriate. Your code should do all of the required steps.
* Code should be well commented. This includes comments at the top of the file stating your name, the assignment, and the purpose of the file; comments at the start of a function stating purpose, parameters, and return values; and comments at each step/group of the code.
* The Jupyter notebook must use markdown effectively as described above
* Your code must run with the version of python installed on jhub, as that is where it will be graded. There are instructions on Moodle on how to run your code on the command line on jhub. Or, you can install the same versions on your own system. 


### Using git effectively
* commit and push your code as you work, not just at the end. Every time you step away from the project, or get something working, you should commit and push that code so that it's saved as a version in your repository. That way you will never lose your work, and can always return to an earlier version if you change your mind. The instructor will not have sympathy if you do not follow these steps and then lose your work.
* Write useful git commit messages, and commit as you work on your program so that you have a history of your work if you make a mistake and need to return to it. Because we have git to help with this aspect of coding, the instructor will have little sympathy for losing or overwriting your files as proper use of git should prevent that issue.
* Do not upload via the github website. Commit/push/pull from your IDE or the command line only.

### Tips

This type of problem is best solved using iterative development. Especially for the data preparation, make changes, and then check to see what else you may need to fix as you work through the dataset.


## Final Interpretation and sharing of results

Update the interpretation.md file to answer the questions posed there about what you did and what you learned.

