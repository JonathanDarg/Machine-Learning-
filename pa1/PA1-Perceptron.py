"""Pa1 Perceptron code"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
<<<<<<< HEAD

=======
>>>>>>> 1169403f61c135e8060024c2bf5feab779b23fa5

#reading the wine data
df = pd.read_csv('wine.data', header=0, encoding='utf-8')
df.head()

#targets
y = df.iloc[:, 0].values #targets class
#features
X = df.drop(df.columns[[0, 5, 13]], axis=1).values  #all data except class,Magnesium,Proline

#split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.35, stratify=y_train)

models = []

#------------------------------------------------------------------------
#target1
y_train_1 = np.where(y_train == 1, 1, 0) #class 1 vs all
y_valid_1 = np.where(y_valid == 1, 1, 0)

#training 
ppn = Perceptron(eta=0.1, n_iter=600) 
ppn.fit(X_train, y_train_1)
models.append(ppn)

#plot model1
plt.figure() 
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Class 1 vs Rest: Training Errors')
plt.show()
plt.savefig("model1.png")


#------------------------------------------------------------------------

#target 2
y_train_2 = np.where(y_train == 2, 1, 0) #class 1 vs all
y_valid_2 = np.where(y_valid == 2, 1, 0)

#training 
ppn = Perceptron(eta=0.1, n_iter=600) 
ppn.fit(X_train, y_train_2)
models.append(ppn)

#plot model2
plt.figure() 
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Class 2 vs Rest: Training Errors')
plt.show()
plt.savefig("model2.png")


#------------------------------------------------------------------------

#target 3
y_train_3 = np.where(y_train == 3, 1, 0) #class 3 vs all
y_valid_3 = np.where(y_valid == 3, 1, 0)

#training 
ppn = Perceptron(eta=0.1, n_iter=600) 
ppn.fit(X_train, y_train_3)
models.append(ppn)

#plot model3
plt.figure() 
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Class 3 vs Rest: Training Errors')
plt.show()
plt.savefig("model3.png")


#------------------------------------------------------------------------

#Multiclass Prediction Function
def predict_multiclass(X):
    scores = np.array([model.net_input(X) for model in models])
    return np.argmax(scores, axis=0) + 1  

#Final Evaluation

y_pred = predict_multiclass(X_test)

print("Final Weights and Biases")
print("-------------------------------------")
for i, model in enumerate(models, 1):
    print(f"Class {i} Model:")
    print(f"Weights: {model.w_}")
    print(f"Bias: {model.b_}\n")
    print("-------------------------------------")
print("Final Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
ConfusionMatrixDisplay(cm).plot()
plt.show()
plt.savefig("ConfusionMatrix.png")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))
