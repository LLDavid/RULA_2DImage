import numpy as np
from sklearn.metrics import classification_report,  confusion_matrix

y_test = np.load("gt.npy")
probabilities = np.load("pred.npy")
pred = []
for i in range(len(probabilities)):
    temp = np.exp(probabilities[i,:])/sum(np.exp(probabilities[i,:]))
    print(temp)
    print(np.dot(temp, [1,2,3,4,5,6,7]))
    pred.append(np.dot(temp, [1,2,3,4,5,6,7]))
print(classification_report(y_test, np.array(pred)))
print(confusion_matrix(y_test, np.array(pred)))