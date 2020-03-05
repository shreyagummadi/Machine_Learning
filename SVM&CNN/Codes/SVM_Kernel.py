import time
from sklearn import svm, metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

''' get mnist data'''
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

''' normalize the data'''
X = X/255.0

opt = int(input("Select 1.PCA 2.LDA "))

''' choose between LDA and PCA'''
if opt ==1:
    pca = decomposition.PCA()
    pca.n_components = 100
    data = pca.fit_transform(X)

elif opt ==2:
    lda = LDA(n_components = 9)
    data = lda.fit_transform(X,y)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.15, shuffle =False)

classifier = svm.SVC(kernel = 'rbf',C=5, gamma = 0.05)
#classfier = svm.SVC(kernel = 'poly', degree = 2)

''' learning on train samples'''
start_time = time.time()
print('Start')
classifier.fit(X_train, y_train)
end_time = time.time()
print('End')
t = end_time-start_time
print('Time taken to run: ', t)

''' evaluate on test samples'''
predicted = classifier.predict(X_test)


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
      
cm = metrics.confusion_matrix(y_test, predicted)
print("Confusion matrix:\n%s" % cm)

print("Accuracy={}".format(metrics.accuracy_score(y_test, predicted)))