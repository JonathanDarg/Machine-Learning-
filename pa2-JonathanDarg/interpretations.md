# Interpretations and Comments on PA2

Answer the following prompts in your own words, about your process and results. Explain in enough detail that it is clear that you understand what you did and why.

## About the Process

### Why did you choose the machine learning algorithm(s) you chose?
I chose SVM because it is used for classification as we want to determine the severity of a crash. And KNN which is also used for classication
grouping up based on its neighbors.

### Why did you choose the hyperparameters you chose?
I chose the hyperparameters because with KNN you have allot of rows of data so having a higher number of neighbors
to form a grouping seems reasonable. And since there is two forms of calulating the distance and you want 90-95% of the data and have equal weight on each neighbor
And for SVM you want to scale the data and keep 95% of the data and set the class weight to be balanced
And for both Grids use f1 scoreing instead of accuracy because of the class imbalance.


## About Results

### How well did each grid search perform?
SVM's best CV Score was 0.513 and KNN's best CV Score was 0.5529. 

So the KNN performed better than the SVM grid search.

### Were you surprised by any of the results? Explain.
I was surprised that KNN performed better than SVM because SVM is a more complex algorithm.
And how close the scores were to each other as the KNN performed slightly better than the SVM grid search. Since KNN is more sensitive to outliers and SVM is good for classifications.


### Would you do anything different next time in a similar type of dataset or process?
Next time I would take into condsideration of how many features I have in the dataset and what is relevant to determine the severity of a crash. Since at first I had to many features at first resulting in the code never finishing.

And it would be interesting to see how the results would change if I used a different algorithm like Random Forest.
