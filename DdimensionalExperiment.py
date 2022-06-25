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

### compute correlation between two random variables
def compute_corr(x1,x2):
    return pearsonr(x1,x2)[0]

### generate a dataset of n samples and standardize the variables x1,x2,x3
def generate_dataset_n(n_data=3000, noise=1, p1=0.5, p2=0.5, n_variables=10, coeffs=[0]):
    x = np.zeros((n_data,n_variables))
    
    x[:,0] = np.random.uniform(size=n_data)
    if coeffs[0]==0:
        coeffs = np.random.uniform(size=n_variables)
    
    for i in range(n_variables-1):
        j = randrange(i+1)
        x[:,i+1] = p1*np.random.uniform(size=n_data) + p2*x[:,j]
    
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    
    delta = np.random.normal(0, noise, size=(n_data,1))
    
    y = np.dot(x,coeffs).reshape(n_data,1) + delta.reshape(n_data,1)
    return x,y,coeffs

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

### compute the theoretical correlation threshold
def compute_real_bound(w1,w2,n,s_squared):
    return 1 - (2*s_squared/((n-1)*(w1-w2)**2))

### aggregate each group of elements with their mean
def aggregate_clusters(cluster,x):
    a = np.zeros((len(x),len(cluster)))
    
    k=0
    for i in cluster:
        a[:,k] = np.mean(x[:,i],axis=1)
        k += 1
    return a

### D-dimensional experiment, repeated n_repetitions times
def single_experiment_realCoeffs_nDim(n_rep=500, n_variables=10, n_data = 3000, noise = 1, p1 = 0.5, p2=0.5, coeffs=[0]):

    x_test,y_test,coeffs = generate_dataset_n(n_data=n_data, noise=noise, p1=p1, p2=p2, n_variables=n_variables,coeffs=coeffs)
    
    score_full = []
    score_aggr = []
    mse_full = []
    mse_aggr = []
    predictions_tot = np.zeros((int(n_data),n_rep))
    predictions_aggr = np.zeros((int(n_data),n_rep))
    n_clust = []
    emp_score_full = []
    emp_score_aggr = []
    emp_mse_full = []
    emp_mse_aggr = []
    emp_predictions_tot = np.zeros((int(n_data),n_rep))
    emp_predictions_aggr = np.zeros((int(n_data),n_rep))
    emp_n_clust = []
    
    for r in range(n_rep):
    
        x,y,coeffs = generate_dataset_n(n_data=n_data, noise=noise, p1=p1, p2=p2, n_variables=n_variables, coeffs=coeffs)
        
        features_df = pd.DataFrame(x)
        target_df = pd.DataFrame(y)
        
        ### theoretical bound

        cluster=[]
        
        for j in range(n_variables):
            aux=0
            for c in cluster:
                if j in c: aux=1
            if aux==1: continue
            curr_list = [j]
            for i in range(n_variables-j-1):
                corr = compute_corr(x[:,j],x[:,i+j+1])
                
                real_bound = compute_real_bound(coeffs[j],coeffs[i+j+1],n_data,noise)
                
                if ((real_bound<= corr)):
                    curr_list.append(i+j+1)
                    #print(j,i+j+1)
                    #print(corr)
                    #print(real_bound)
            cluster.append(curr_list)
        
        n_clust.append(len(cluster))
        x_aggr=aggregate_clusters(cluster,x)
        x_test_aggr = aggregate_clusters(cluster,x_test)
        aggregate_df = pd.DataFrame(x_aggr)
            
        regr_full = LinearRegression().fit(features_df, target_df)
        regr_aggr = LinearRegression().fit(aggregate_df, target_df)
            
        score_full.append(regr_full.score(x_test, y_test))
        score_aggr.append(regr_aggr.score(x_test_aggr, y_test))
    
        mse_full.append(mean_squared_error(y_test,regr_full.predict(x_test)))
        mse_aggr.append(mean_squared_error(y_test,regr_aggr.predict(x_test_aggr)))
        
        predictions_aggr[:,r] = regr_aggr.predict(x_test_aggr)[:,0]
        predictions_tot[:,r] = regr_full.predict(x_test)[:,0]
        
        ### empirical bound

        cluster=[]
        
        for j in range(n_variables):
            aux=0
            for c in cluster:
                if j in c: aux=1
            if aux==1: continue
            curr_list = [j]
            for i in range(n_variables-j-1):
                corr = compute_corr(x[:,j],x[:,i+j+1])
                
                real_bound = compute_empirical_bound(x[:,j],x[:,i+j+1],y) # only difference
                
                if ((real_bound<= corr)):
                    curr_list.append(i+j+1)

            cluster.append(curr_list)
        
        emp_n_clust.append(len(cluster))
        x_aggr=aggregate_clusters(cluster,x)
        x_test_aggr = aggregate_clusters(cluster,x_test)
        aggregate_df = pd.DataFrame(x_aggr)
            
        regr_full = LinearRegression().fit(features_df, target_df)
        regr_aggr = LinearRegression().fit(aggregate_df, target_df)
            
        emp_score_full.append(regr_full.score(x_test, y_test))
        emp_score_aggr.append(regr_aggr.score(x_test_aggr, y_test))
    
        emp_mse_full.append(mean_squared_error(y_test,regr_full.predict(x_test)))
        emp_mse_aggr.append(mean_squared_error(y_test,regr_aggr.predict(x_test_aggr)))
        
        emp_predictions_aggr[:,r] = regr_aggr.predict(x_test_aggr)[:,0]
        emp_predictions_tot[:,r] = regr_full.predict(x_test)[:,0]

    return y_test,coeffs,cluster,score_full,score_aggr,mse_full,mse_aggr,predictions_tot,predictions_aggr,n_clust,emp_score_full,emp_score_aggr,emp_mse_full,emp_mse_aggr,emp_predictions_tot,emp_predictions_aggr,emp_n_clust


### print 95% CI considering the distribution to be gaussian
def print_95CI(mylist):
    return str(round(np.mean(mylist),6))+'±'+str(round(1.96*np.std(mylist)/np.sqrt(3000),6))

### compute 95% CI considering the distribution to be gaussian
def compute_95CI(mylist):
    #return str(round(np.mean(mylist),6))+'±'+str(round(1.96*np.std(mylist)/np.sqrt(3000),6))
    return np.mean(mylist)-1.96*np.std(mylist)/np.sqrt(3000),np.mean(mylist)+1.96*np.std(mylist)/np.sqrt(len(mylist))
 
### empirical bias-variance decomposition of MSE with CI
def compute_biasVariance(predictions_tot,predictions_aggr,y_test):
    means_tot = np.mean(predictions_tot,axis=1)
    sq_diff_tot = (np.transpose(predictions_tot) - np.transpose(means_tot))**2

    avg_var_tot = np.mean(sq_diff_tot)
    inf_avg_var_tot,sup_avg_var_tot = compute_95CI(sq_diff_tot)
        
    means_aggr = np.mean(predictions_aggr,axis=1)
    sq_diff_aggr = (np.transpose(predictions_aggr) - np.transpose(means_aggr))**2
    
    avg_var_aggr = np.mean(sq_diff_aggr)
    inf_avg_var_aggr,sup_avg_var_aggr = compute_95CI(sq_diff_aggr)
    
    sq_diff_avg_tot = (means_tot.reshape(-1,1) - y_test)**2
    avg_bias_tot = np.mean(sq_diff_avg_tot)
    inf_avg_bias_tot,sup_avg_bias_tot = compute_95CI(sq_diff_avg_tot)

    sq_diff_avg_aggr = (means_aggr.reshape(-1,1) - y_test)**2
    avg_bias_aggr = np.mean(sq_diff_avg_aggr)
    inf_avg_bias_aggr,sup_avg_bias_aggr = compute_95CI(sq_diff_avg_aggr)
    
    return avg_var_tot,inf_avg_var_tot,sup_avg_var_tot,avg_var_aggr,inf_avg_var_aggr,sup_avg_var_aggr,avg_bias_tot,inf_avg_bias_tot,sup_avg_bias_tot,avg_bias_aggr,inf_avg_bias_aggr,sup_avg_bias_aggr

### main run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_data", default=500, type=int)
    parser.add_argument("--n_variables", default=100, type=int)
    parser.add_argument("--noise", default=10, type=float)
    parser.add_argument("--n_repetitions", default=500, type=int)
    parser.add_argument("--p1", default=0.3, type=float)
    parser.add_argument("--p2", default=0.7, type=float)

    args = parser.parse_args()
    print(args)

    ##################### experiments ########################

    y_test,coeffs,cluster,score_full,score_aggr,mse_full,mse_aggr,predictions_tot,predictions_aggr,n_clust,emp_score_full,emp_score_aggr,emp_mse_full,emp_mse_aggr,emp_predictions_tot,emp_predictions_aggr,emp_n_clust = single_experiment_realCoeffs_nDim(n_variables = args.n_variables, n_data = args.n_data, noise = args.noise, p1 = args.p1, p2=args.p2, coeffs=[0], n_rep=args.n_repetitions)
    
    ##################### print the results ########################
    print("Real and empirical reduced dimensions: {0}, {1}\n".format(print_95CI(n_clust),print_95CI(emp_n_clust)))

    print("Sample R2 full: {0}\n".format(print_95CI(score_full)))
    print("Sample R2 aggr (theo): {0}\n".format(print_95CI(score_aggr)))
    print("Sample R2 aggr (emp): {0}\n".format(print_95CI(emp_score_aggr)))

    print("Sample MSE full: {0}\n".format(print_95CI(mse_full)))
    print("Sample MSE aggr (theo): {0}\n".format(print_95CI(mse_aggr)))
    print("Sample MSE aggr (emp): {0}\n".format(print_95CI(emp_mse_aggr)))
