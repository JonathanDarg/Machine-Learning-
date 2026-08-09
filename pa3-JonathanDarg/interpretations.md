# PA3 Interpretations

## Interpretations after Training

Answer the following questions after completing your training steps, before analyzing the hyperparameters in a Jupyter notebook.

### How well did each grid search appear to work based on the *training* data's classification report?

based on the training data's classification report, the grid search from accuracy to balanced accuracy seemed to increase in its precision of class 0 and not class 1, but greatly increased the recall and f1-score of the minority class 1.


### How does your interpretation change (or not change) after seeing the test data's classification report? Why do you think the results are what they are?

my interpretations do not change when looking at the tests data's classification report because in the balanced it is giving the minority class more weight. Allowing class 1 to increase its recall from 0.09 to 0.77 which is similar to class 0's recall. And how now in the test the number of rows have been decreased.

## Interpretation after Analysis

Your Jupyter notebook should have markdown with analysis after each output or graph created to analyze the cv results. Below answer the big picture questions related to your analysis.

### Did the effect of hyperparameter values match between metrics, or differ? In what way?

they slightly differed between metrics because in the accuracy Gridsearch the hyperparameters would kind of stay the same but in the balanced accuracy gridsearch the hyperparameters would change a lot. This is because the balanced accuracy is giving more weight to the minority class and not just focusing on the majority class.

### What about the analysis graphs surprised you? If nothing surprised you, what about the graphs were as expected?

it was surprising to see how some barely moved and stayed the same while others moved a lot. And how in the last one of max features for the balanced accuracy gridsearch the 20 balanced class weight decreased from having a max depth of 10 having a balanced class weight.

### Did either or both searches result in models with overfitting? How can you tell?

in the balanced accuracy gridsearch the model with the max depth of 10 and balanced class weight had a score of 0.75, but it degreased to 0.65 with a max depth of 20 in the test data. Showing how this decrease was a result of overfitting because the model was too complex and was not able to generalize well to the test data.


### Did the best model perform as well on the test data as you expected? How so?

The bbest model did not preform as well as i expected because in the Balanced accuracy gridsearch the precsion of class 0 increased to 0.95 but the precision of class 1 was still kind of low at 0.31 but the recall between class 0 and 1 is more balanced being in the 0.7s.

### This data was imbalanced. What did we do because of that in our process? Was it effective?

some things we did to deal with the imbalance was using the balanced class weight and using the balanced accuracy gridsearch. This was effective because it allowed us to increase the recall of the minority class 1 from 0.09 to 0.77 which is similar to class 0's recall.

### What else could we try later to deal with the imbalance?

Some other things we could try to deal with the class imbalance could be to use SMOTE to oversample the minority class or undersample the majority class. 

