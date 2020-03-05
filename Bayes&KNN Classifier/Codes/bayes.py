import numpy as np
from scipy.io import loadmat

from PCA import PCA
from LDA import LDA


data = int(input("Select 1.data.mat 2.pose.mat 3.illumination.mat: "))
opt = int(input("Select 1.Bayes 2.Bayes+PCA 3.Bayes+LDA: "))


''' selecting the dataset'''
''' dividing into train and test'''
if data ==1:
    x = loadmat('data.mat')
    t = x['face'] #acess the images
    t = np.matrix(t.reshape((504,600)))
    c = 200 # number of classes
    d = 504
    ntrain = 400
    ntest = 200
    n = 2 # no.of train samples per class
    x_train = np.matrix(np.zeros((d,ntrain)),dtype = complex)
    x_test = np.matrix(np.zeros((d,ntest)),dtype= complex)
    label_train = np.zeros((ntrain,1))
    label_test = np.zeros((ntest,1))
    for i in range(0,c):
        count = 0
        for j in range(0,3):
            if j==0 or j==1:
                x_train[:,2*i+count] =  t[:,3*i+j]
                label_train[2*i+count] = i
                count = count+1
            else:
                x_test[:,i] = t[:,3*i+j]
                label_test[i] = i

elif data==2:
    x = loadmat('pose.mat')
    t = x['pose']
    d = 1920
    c = 68
    percent = 0.6 #percent of data for training
    n = round(percent*13) # no.of train samples per class
    ntrain = n*68
    ntest = (13-n)*68
    x_train = np.matrix(np.zeros((d,ntrain)),dtype = complex)
    x_test = np.matrix(np.zeros((d,ntest)),dtype= complex)
    label_train = np.zeros((ntrain,1))
    label_test = np.zeros((ntest,1))
    for i in range(0,c):
        for j in range(0,n):
            x_train[:,n*i+j] = np.reshape(t[:,:,j,i],(d,1))
            label_train[n*i+j] = i
        for j in range(0,(13-n)):
            x_test[:,(13-n)*i+j] = np.reshape(t[:,:,n+j,i],(d,1))
            label_test[(13-n)*i+j]= i

elif data==3:
    x = loadmat('illumination.mat')
    t = x['illum']

    c = 68
    d  =1920
    percent = 0.8
    n = round(percent*21)
    ntrain = n*68
    ntest = (21-n)*68
    x_train = np.matrix(np.zeros((d,ntrain)),dtype = complex)
    x_test = np.matrix(np.zeros((d,ntest)),dtype= complex)
    label_train = np.zeros((ntrain,1))
    label_test = np.zeros((ntest,1))
    for i in range(0,c):
        for j in range(0,n):
            x_train[:,n*i+j] = np.reshape(t[:,j,i],(d,1))
            label_train[n*i+j] = i
        for j in range(0,(21-n)):
            x_test[:,(21-n)*i+j] = np.reshape(t[:,n+j,i],(d,1))
            label_test[(21-n)*i+j]= i


''' dimensionality reduction or original data'''
if opt==1:
    x_train,x_test = x_train,x_test
elif opt==2: 
    x_train,x_test = PCA(x_train,x_test)
    d,_ = x_train.shape
elif opt ==3:
    x_train,x_test = LDA(x_train,x_test,c,n)
    d,_ = x_train.shape


''' estimating mean and variance from train data'''

mean_train = np.matrix(np.zeros((d,c)),dtype=complex)
for k in range(0,c):
    for l in range(0,n):
        mean_train[:,k] = mean_train[:,k]+ x_train[:,n*(k)+l]
    mean_train[:,k] = (1/n)*mean_train[:,k]

cov_train = np.zeros((d,d,c),dtype=complex)
cov_inv = np.zeros((d,d,c),dtype=complex)
for a in range(0,c):
    for b in range(0,n):
        cov_train[:,:,a] = cov_train[:,:,a]+((x_train[:,n*(a)+b]-mean_train[:,a])*((x_train[:,n*(a)+b]-mean_train[:,a]).T))
    cov_train[:,:,a] = (1/n)*cov_train[:,:,a] 
    cov_train[:,:,a] = cov_train[:,:,a]+ 1*np.eye(d)
    
    cov_inv[:,:,a] = np.linalg.inv(cov_train[:,:,a])

''' calculate discriminant'''

W = np.zeros((d, d, c),dtype = complex)
w = np.matrix(np.zeros((d, c)),dtype = complex)
w0 = np.zeros((c,1),dtype = complex)

for m in range(0,c):
    W[:,:,m] = (-1/2) * cov_inv[:, :, m]
    w[:,m] = cov_inv[:, :, m] * mean_train[:,m]
    w0[m] = (-1/2) *((( mean_train[:, m]).T* cov_inv[:,:,m]*mean_train[:,m])+np.log(np.linalg.det(cov_train[:,:,m])))
    

''' assign label based on discriminant'''    
solution = np.zeros((ntest,1))
for u in range(0, ntest):
    max_g = (x_test[:,u].T * W[:,:,1] * x_test[:,u]) + ((w[:,1].T) * x_test[:,u]) + w0[1]
    for v in range(0,c):
        g = (x_test[:,u].T * W[:,:,v] * x_test[:,u]) +(( w[:,v].T) * x_test[:,u]) + w0[v]
#        print(g)
        if (g >= max_g):
           max_g = g
           solution[u] = v

''' accuracy'''
accuracy = 0.0
for z in range(0,ntest):
   if solution[z]== label_test[z]:
       accuracy = accuracy + 1

accuracy = accuracy / ntest;
print("The accuracy is: ")
print(accuracy)
            
