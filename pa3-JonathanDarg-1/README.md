[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YiyFbQSs)
# PA3

* CS484
* Spring 2025
* Due: Monday April 14 at 11:59PM

Note: you know everything you need to know when this assignment is assigned. You can finish this much earlier than the deadline if you want to. Because the slow part (grid search) is separated from analysis, you can easily break this project into sub-tasks.

## Purpose

This assignment will give you the opportunity to practice:
1. Working with ensembles
2. Using the full machine learning process with gridsearch and pipelines
3. Fully understanding the role of hyperparameters
4. Using Jupyter notebooks and markdown effectively

## Overview 

In this assignment you will use a random forest to try to predict if someone will have diabetes. This dataset is highly imbalanced, but otherwise does not need much pre-processing. Instead, we'll focus on closely examining the full results of our grid searches to really see how each hyperparameter is affecting the learning.

You will use a regular python script for your data preparation and grid search.

You will use a Juptyer notebook for analysis.


## Dataset and Dataset Preparation

The dataset is provided to you, and is real data from a survey of patients. The label to predict corresponds to whether or not the patient has diabetes. There are 22 features in addition to the label.

In your python script, you should output the names of the features and check if they are already numeric, then drop the one feature that is obvious to need to drop. If there is any other pre-processing necessary, you should do that and then do your train/test split.


## Training

We are going to train our model using random forests with a grid search. We will do the same grid search twice; the only difference is that the first grid search will use `accuracy` for the scoring metric, and the second grid search will use either `f1` score or `balanced_accuracy`; your choice. The rest of the grid search parameters much be identical between the two.

You should make good choices on hyperparameters. You must include `class_weight` as one of your tested hyperparameters. Your goal is to complete enough of a parameter search that we can then fully analyze the role of the hyperparameters. You should test values for `n_estimators`, `oob_score`, and `max_features`, plus one more of your choice. 

After each search is complete, save the full results of the search as a csv file. You can access the results of a grid search from the `cv_results_` attribute of the grid search variable. Load this data into a pandas dataframe and save to a file for each grid search. Note: Pandas will interpret the `None` value for class weight as a missing value. You should replace with the string `"None"` to enable your further analysis. Recall that PA2 taught you how to save a pandas dataframe to a file correctly.

Output a classification report on the *training* data for each model.

Then output a classification report on the *testing* data for each model.

Answer the questions related to training in `interpretations.md`.

## Analysis

Now that we have completed our grid searches, we can analyze their results. The following analysis should be done using pandas, matplotlib, and seaborn in a Jupyter notebook. Be sure to use markdown in that notebook to organize what you are doing and interpret each graph to answer the questions.

When we use accuracy as our grid search score we want to answer the following questions. In each case we want to differentiate between `class_weight` values in the graphs (hint: `class_weight` as `hue` in seaborn):
1. How good do the results appear to be based on the scoring?
2. How does the number of estimators affect overall accuracy?
3. How does `max_features` relate to accuracy?
4. How does `max_features` affect the time it took to fit the model?
5. Does `oob_score` affect the goodness of the model?
6. How does the hyperparameter you chose affect the results?

Next, answer the same questions on the results from the f1 scored search with graphs and markdown in your notebook.

Finally, answer the big picture questions about your model in the `interpretations.md` file.

## Tips

If you need to create two different py files for your different searches due to the time it takes for your search to run, that is fine. Just make it clear in the comments at the top of each file what each one is doing.

Seaborn is fairly straightforward to use with pandas. Here's a useful video on barplots: https://www.youtube.com/watch?v=3Yh4U5OB5Sk&ab_channel=KimberlyFessel. 

She has other videos on other approaches as well, such as box plots: https://www.youtube.com/watch?v=Vo-bfTqEFQk&t=730s&ab_channel=KimberlyFessel

## Programming Expectations 

* You should use pandas for all processing of data
* Code should be well designed python code, such as using functions as appropriate. Your code should do all of the required steps.
* Code should be well commented. This includes comments at the top of the file stating your name, the assignment, and the purpose of the file; comments at the start of a function stating purpose, parameters, and return values; and comments at each step/group of the code.
* The Jupyter notebook must use markdown effectively as described above
* Your code must run with the version of python installed on jhub, as that is where it will be graded. There are instructions on Moodle on how to run your code on the command line on jhub. Or, you can install the same versions on your own system. 


### Using git effectively
* commit and push your code as you work, not just at the end. Every time you step away from the project, or get something working, you should commit and push that code so that it's saved as a version in your repository. That way you will never lose your work, and can always return to an earlier version if you change your mind. The instructor will not have sympathy if you do not follow these steps and then lose your work.
* Write useful git commit messages, and commit as you work on your program so that you have a history of your work if you make a mistake and need to return to it. Because we have git to help with this aspect of coding, the instructor will have little sympathy for losing or overwriting your files as proper use of git should prevent that issue.
* Do not upload via the github website. Commit/push/pull from your IDE or the command line only.



## Final Interpretation and sharing of results

Update the `interpretations.md` file to answer the questions posed there about what you did and what you learned.

