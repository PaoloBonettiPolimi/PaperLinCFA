import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.decomposition import PCA
import argparse
from random import randrange
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import utils
from sklearn.preprocessing import KernelCenterer, scale
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import linalg
from scipy.sparse.linalg import eigsh as ssl_eigsh

### compute correlation between two random variables
def compute_corr(x1,x2):
    return pearsonr(x1,x2)[0]

### compute the correlation threshold with empirical data
def compute_empirical_bound(x1,x2,y):

    x = np.zeros((len(x1),2))
    x[:,0] = x1
    x[:,1] = x2
    
    regr = LinearRegression()
    regr.fit(x,y)
    w1 = regr.coef_[0][0]
    w2 = regr.coef_[0][1]
    preds = regr.predict(x)
    residuals = y - preds
    
    n=len(x1)
    
    s_squared = np.dot(residuals.reshape(1,n),residuals)/(n-3)
    bound = 1 - (2*s_squared/((n-1)*(w1-w2)**2))
    corr = compute_corr(x1.reshape(n),x2.reshape(n))

    return bound

### aggregate each group of elements with their mean
def aggregate_clusters(cluster,x):
    a = np.zeros((len(x),len(cluster)))
    
    k=0
    for i in cluster:
        a[:,k] = np.mean(x[:,i],axis=1)
        k += 1
    return a

### D-dimensional experiment on the dataset
def single_experiment_realData_nDim(X_train, X_test, y_train, y_test):

    n_variables = X_train.shape[1]
    n_data = X_train.shape[0]
    
    score_full = []
    score_aggr = []
    mse_full = []
    mse_aggr = []
    predictions_tot = np.zeros((int(n_data),1))
    predictions_aggr = np.zeros((int(n_data),1))
    n_clust = []
    emp_score_full = []
    emp_score_aggr = []
    emp_mse_full = []
    emp_mse_aggr = []
    emp_predictions_tot = np.zeros((int(n_data),1))
    emp_predictions_aggr = np.zeros((int(n_data),1))
    emp_n_clust = []
        
    features_df = pd.DataFrame(X_train)
    target_df = pd.DataFrame(y_train)
    x = np.array(X_train)
    y = np.array(y_train)
    
    cluster=[]
    used_indices = []
    
    for j in range(n_variables):
        
        if j in used_indices: continue
        curr_list = [j]
        used_indices.append(j)
        for i in range(n_variables-j-1):
            corr = compute_corr(x[:,j],x[:,i+j+1])
            
            real_bound = compute_empirical_bound(x[:,j],x[:,i+j+1],y) 
                
            if ((real_bound<= corr)):
                curr_list.append(i+j+1)
                used_indices.append(i+j+1)
        cluster.append(curr_list)
    
    emp_n_clust = len(cluster)
    
    x_aggr=aggregate_clusters(cluster,x)
    x_test_aggr = aggregate_clusters(cluster,X_test)
    aggregate_df = pd.DataFrame(x_aggr)
        
    regr_full = LinearRegression().fit(features_df, target_df)
    regr_aggr = LinearRegression().fit(aggregate_df, target_df)
        
    print("full regression score: {0}".format(regr_full.score(X_test, y_test)))
    print("aggr regression score: {0}".format(regr_aggr.score(x_test_aggr, y_test)))
    score_full.append(regr_full.score(X_test, y_test))
    score_aggr.append(regr_aggr.score(x_test_aggr, y_test))
    
    print("full regression MSE: {0}".format(mean_squared_error(y_test,regr_full.predict(X_test))))
    print("aggr regression MSE: {0}".format(mean_squared_error(y_test,regr_aggr.predict(x_test_aggr))))
    
    mse_full=mean_squared_error(y_test,regr_full.predict(X_test))
    mse_aggr=mean_squared_error(y_test,regr_aggr.predict(x_test_aggr))
    
    return cluster,score_full,score_aggr,mse_full,mse_aggr


### supervised PCA
class spca(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_components, kernel="linear", eigen_solver='auto', 
                 max_iterations=None, gamma=0, degree=3, coef0=1, alpha=1.0, 
                 tolerance=0, fit_inverse_transform=False):
        
        self._num_components = num_components
        self._gamma = gamma
        self._tolerance = tolerance
        self._fit_inverse_transform = fit_inverse_transform
        self._max_iterations = max_iterations
        self._degree = degree
        self._kernel = kernel
        self._eigen_solver = eigen_solver
        self._coef0 = coef0
        self._centerer = KernelCenterer()
        self._alpha = alpha
        self._alphas = []
        self._lambdas = []
        
        
    def _get_kernel(self, X, Y=None):
        # Returns a kernel matrix K such that K_{i, j} is the kernel between the ith and jth vectors 
        # of the given matrix X, if Y is None. 
        
        # If Y is not None, then K_{i, j} is the kernel between the ith array from X and the jth array from Y.
        
        # valid kernels are 'linear, rbf, poly, sigmoid, precomputed'
        
        args = {"gamma": self._gamma, "degree": self._degree, "coef0": self._coef0}
        
        return pairwise_kernels(X, Y, metric=self._kernel, n_jobs=-1, filter_params=True, **args)
    
    
    
    def _fit(self, X, Y):
        
        # calculate kernel matrix of the labels Y and centre it and call it K (=H.L.H)
        K = self._centerer.fit_transform(self._get_kernel(Y))
        
        # deciding on the number of components to use
        if self._num_components is not None:
            num_components = min(K.shape[0], self._num_components)
        else:
            num_components = self.K.shape[0]
        
        # Scale X
        # scaled_X = scale(X)
        
        # calculate the eigen values and eigen vectors for X^T.K.X
        Q = (X.T).dot(K).dot(X)
        
        # If n_components is much less than the number of training samples, 
        # arpack may be more efficient than the dense eigensolver.
        if (self._eigen_solver=='auto'):
            if (Q.shape[0]/num_components) > 20:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self._eigen_solver
        
        if eigen_solver == 'dense':
            # Return the eigenvalues (in ascending order) and eigenvectors of a Hermitian or symmetric matrix.
            self._lambdas, self._alphas = linalg.eigh(Q, eigvals=(Q.shape[0] - num_components, Q.shape[0] - 1))
            # argument eigvals = Indexes of the smallest and largest (in ascending order) eigenvalues
        
        elif eigen_solver == 'arpack':
            # deprecated :: self._lambdas, self._alphas = utils.arpack.eigsh(A=Q, num_components, which="LA", tol=self._tolerance)
            self._lambdas, self._alphas = ssl_eigsh(A=Q, k=num_components, which="LA", tol=self._tolerance)
            
        indices = self._lambdas.argsort()[::-1]
        
        self._lambdas = self._lambdas[indices]
        self._lambdas = self._lambdas[self._lambdas > 0]  # selecting values only for non zero eigen values
        
        self._alphas = self._alphas[:, indices]
        #return self._alphas
        self._alphas = self._alphas[:, self._lambdas > 0]  # selecting values only for non zero eigen values
        
        self.X_fit = X

        
    def _transform(self):
        return self.X_fit.dot(self._alphas)
        
        
    def transform(self, X):
        return X.dot(self._alphas)
        
        
    def fit(self, X, Y):
        self._fit(X,Y)
        return
        
        
    def fit_and_transform(self, X, Y):
        self.fit(X, Y)
        return self._transform()
    

### main run
if __name__ == "__main__":

################### Life Expectancy ###################
    print('\n### Life Expectancy ###')

    df = pd.read_csv('../Paper_LinBVA/dataset/Life Expectancy Data.csv')
    df = df.dropna()

    X = df.iloc[:,4:]
    Y = df.iloc[:,3]

    print("Dataset shape: {}".format(X.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    scaler = preprocessing.StandardScaler().fit(np.array(X_train))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    scaler = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
    y_train = scaler.transform(y_train.reshape(-1,1))
    y_test = scaler.transform(y_test.reshape(-1,1))

    cluster,score_full,score_aggr,mse_full,mse_aggr = single_experiment_realData_nDim(X_train, X_test, y_train, y_test)

    pca_res = []
    mse_res = []
    i=0
    
    while True:
        try:
            trial = spca(num_components=i+1, kernel='linear')
            X_train_proj = trial.fit_and_transform(X_train,y_train)
            X_test_proj = trial.transform(X_test)
            
            if X_train_proj.shape[1]==0: continue
            
            regr = LinearRegression().fit(X_train_proj, y_train)
            pca_res.append(regr.score(X_test_proj, y_test))
            mse_res.append(mean_squared_error(y_test,regr.predict(X_test_proj)))
            i += 1
        except:
            break
            
    print("Supervised PCA best number of components, R2 score, MSE:\n {0}".format(np.argmax(pca_res)))
    print(pca_res[np.argmax(pca_res)])
    print(mse_res[np.argmax(pca_res)])

    print('LinBVA number of reduced dimensions: ')
    print(len(cluster))

    ################### Finance ###################

    print('\n### Finance ###')

    df = pd.read_csv('../Paper_LinBVA/dataset/fundamentals.csv')
    df = df.select_dtypes(['number'])
    cols_to_delete = df.columns[df.isnull().sum()/len(df) > .50]
    df.drop(cols_to_delete, axis = 1, inplace = True)
    df = df.dropna()
    X = df.drop(['Unnamed: 0','Cash Ratio'],axis=1)
    y=df['Cash Ratio']

    print("Dataset shape: {}".format(X.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    scaler = preprocessing.StandardScaler().fit(np.array(X_train))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    scaler = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
    y_train = scaler.transform(y_train.reshape(-1,1))
    y_test = scaler.transform(y_test.reshape(-1,1))

    cluster,score_full,score_aggr,mse_full,mse_aggr = single_experiment_realData_nDim(X_train, X_test, y_train, y_test)
    print('LinBVA number of reduced dimensions: ')
    print(len(cluster))

    for kernel in ['linear', 'poly', 'sigmoid']:
        i=0
        pca_res = []
        mse_res = []
        while (i<=50):
            try:
            
                trial = spca(num_components=i+1, kernel=kernel, degree=3, gamma=None, coef0=1)
                X_train_proj = trial.fit_and_transform(X_train,y_train)
                X_test_proj = trial.transform(X_test)
                
                if X_train_proj.shape[1]==0: continue
            
                regr = LinearRegression().fit(X_train_proj, y_train)
                pca_res.append(regr.score(X_test_proj, y_test))
                mse_res.append(mean_squared_error(y_test,regr.predict(X_test_proj)))

                i += 1

            except: break 
        
        print("Supervised PCA best number of components, R2 score, MSE:\n {0}".format(np.argmax(pca_res)))
        print(pca_res[np.argmax(pca_res)])
        print(mse_res[np.argmax(pca_res)])


    ################### Climate ###################

    print('\n### Climate ###')

    # 1000 dati, 20 colonne
    df = pd.read_csv('../Paper_LinBVA/dataset/NDVI_anomalies.csv')
    df = df.select_dtypes(['number'])
    cols_to_delete = df.columns[df.isnull().sum()/len(df) > .50]
    df.drop(cols_to_delete, axis = 1, inplace = True)
    df = df.dropna()
    y=df[df.columns[3]]
    #df = df.iloc[:,4:]
    X = df.iloc[:,4:104]
    #y=df[df.columns[1]]

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    scaler = preprocessing.StandardScaler().fit(np.array(X_train))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    scaler = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
    y_train = scaler.transform(y_train.reshape(-1,1))
    y_test = scaler.transform(y_test.reshape(-1,1))

    cluster,score_full,score_aggr,mse_full,mse_aggr = single_experiment_realData_nDim(X_train, X_test, y_train, y_test)
    print('LinBVA number of reduced dimensions: ')
    print(len(cluster))

    for kernel in ['linear', 'poly', 'sigmoid']:
        i=0
        pca_res = []
        mse_res = []
        while (i<=50):
            try:
            
                trial = spca(num_components=i+1, kernel=kernel, degree=3, gamma=None, coef0=1)
                X_train_proj = trial.fit_and_transform(X_train,y_train)
                X_test_proj = trial.transform(X_test)
                
                if X_train_proj.shape[1]==0: continue
            
                regr = LinearRegression().fit(X_train_proj, y_train)
                pca_res.append(regr.score(X_test_proj, y_test))
                mse_res.append(mean_squared_error(y_test,regr.predict(X_test_proj)))

                i += 1

            except: break 
        
        print("Supervised PCA best number of components, R2 score, MSE:\n {0}".format(np.argmax(pca_res)))
        print(pca_res[np.argmax(pca_res)])
        print(mse_res[np.argmax(pca_res)])