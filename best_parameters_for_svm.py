from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
# download and read mnist
mnist = fetch_mldata('MNIST original', data_home='./')
# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

result = np.zeros((3,1))
test_max = 0
test_max_index = 0
train_max = 0
train_max_index = 0
for i in range(1,50,1):
    classifierSVM = LinearSVC(penalty='l2', loss='squared_hinge', 
                           dual=True, tol=0.0001, C=0.1 * i, 
                           multi_class='ovr', fit_intercept=True, 
                           intercept_scaling=1, class_weight=None, 
                           verbose=0, random_state=None, max_iter=100000)
    classifierSVM.fit(X_train,Y_train)
    y_predict = classifierSVM.predict(X_train)
    Y_predict = classifierSVM.predict(X_test)
    test_accuracy = classifierSVM.score(X_test,Y_test)
    train_accuracy = classifierSVM.score(X_train,Y_train)
#   print('Training accuracy: %0.2f%%' % (train_accuracy*100))
#   print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
    print('C:',0.1*i,' Traning_accuracy = ',train_accuracy,' Test_accuracy = ',test_accuracy)
    temp = np.array([[0.1 * i],[train_accuracy],[test_accuracy]])
    result = np.c_[result,temp]
    if test_accuracy > test_max:
        test_max = test_accuracy
        test_max_index = i
    if train_accuracy > train_max:
        train_max = train_accuracy
        train_max_index = c 
result = np.delete(result, 0, axis=1)
plt.subplot(211)
plt.plot(result[0,:],result[1,:],'r')
plt.subplot(212)
plt.plot(result[0,:],result[2,:],'g')
plt.show()
print('Best training accuracy: %0.2f%%' % (train_max*100))
print('Best C for train: %0.2f%%' % (train_max_index*100))
print('Best testing accuracy: %0.2f%%' % (test_max*100))
print('Best C for test: %0.2f%%' % (test_max_index*100))