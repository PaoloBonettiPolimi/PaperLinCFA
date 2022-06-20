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

### compute correlation between two random variables
def compute_corr(x1,x2):
    return pearsonr(x1,x2)[0]

### generate a dataset of n samples and standardize the variables x1,x2
def generate_dataset(n=3000, noise=1, c1=0.5, c2=0.5, p1=0.5, p2=0.5):
    x = np.zeros((n,2))
    
    x[:,0] = np.random.uniform(size=n)
    x[:,1] = p1*np.random.uniform(size=n) + p2*x[:,0]
    coeffs = np.array([c1,c2]).reshape(2,1)
    
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    
    delta = np.random.normal(0, noise, size=(n,1)) # standard deviation
    
    y = np.dot(x,coeffs).reshape(n,1) + delta.reshape(n,1)
    return x,y,coeffs

### compute the correlation threshold with empirical data
def compute_bound_forCluster(x1,x2,y):

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
        
    return bound,corr,s_squared

### compute the theoretical correlation threshold
def compute_real_bound(w1,w2,n,s_squared):
    return 1 - (2*s_squared/((n-1)*(w1-w2)**2))

### bidimensional experiment, repeated n_repetitions times
def perform_experiment(n_data=3000, noise=0.1, c1=0.3, c2=0.7, n_repetitions=500, p1=0.5, p2=0.5):
    n_aggr = 0 
    n_aggr_empirical = 0
    list_corr = [] 
    list_bound = []
    list_s_squared = [] 
    w1_full = []
    w2_full = []
    w_aggr = []
    list_r2_full = []
    list_r2_aggr = []
    mse_tot = []
    mse_aggr = []
    list_real_bound = []

    predictions_tot = np.zeros((int(n_data),n_repetitions))
    predictions_aggr = np.zeros((int(n_data),n_repetitions))
    
    x_test,y_test,coeffs_test = generate_dataset(n=int(n_data), noise=noise, c1=c1, c2=c2, p1=p1, p2=p2)
    x_test_aggr = (x_test[:,0]+x_test[:,1])/2
    
    for i in range(n_repetitions):
    
        x,y,coeffs = generate_dataset(n=n_data, noise=noise, c1=c1, c2=c2, p1=p1, p2=p2)
        features_df = pd.DataFrame(x)
        target_df = pd.DataFrame(y)
        
        aggregate_col = (x[:,0] + x[:,1]) / 2
        aggregations_df = pd.DataFrame(aggregate_col)
            
        bound,corr,s_squared = compute_bound_forCluster(features_df.iloc[:,0].values,features_df.iloc[:,1].values,target_df.values)
        list_corr.append(corr)
        list_bound.append(bound)
        list_s_squared.append(s_squared)
        
        regr_full = LinearRegression().fit(features_df, target_df)
        mse_tot.append(mean_squared_error(y_test,regr_full.predict(x_test)))
        predictions_tot[:,i] = regr_full.predict(x_test)[:,0]
        w1_full.append(regr_full.coef_[0][0])
        w2_full.append(regr_full.coef_[0][1])
        list_r2_full.append(regr_full.score(x_test, y_test))
        
        regr_aggr = LinearRegression().fit(aggregations_df, target_df)
        mse_aggr.append(mean_squared_error(y_test,regr_aggr.predict(x_test_aggr.reshape(-1, 1))))
        predictions_aggr[:,i] = regr_aggr.predict(x_test_aggr.reshape(-1, 1))[:,0]
        w_aggr.append(regr_aggr.coef_[0][0])
        list_r2_aggr.append(regr_aggr.score(x_test_aggr.reshape(-1, 1), y_test))

        real_bound = compute_real_bound(w1=c1,w2=c2,n=n_data,s_squared=noise*noise)
        list_real_bound.append(real_bound)

        if corr>=real_bound : 
            n_aggr +=1
        
        if corr>=bound:
            n_aggr_empirical += 1
            
    return n_aggr,n_aggr_empirical,mse_tot,mse_aggr,predictions_tot,predictions_aggr,y_test,w1_full,w2_full,w_aggr,list_corr,list_bound, list_s_squared, list_r2_full, list_r2_aggr, list_real_bound

### compute 95% CI considering the distribution to be gaussian
def compute_95CI(mylist):
    #return str(round(np.mean(mylist),6))+'±'+str(round(1.96*np.std(mylist)/np.sqrt(3000),6))
    return np.mean(mylist)-1.96*np.std(mylist)/np.sqrt(3000),np.mean(mylist)+1.96*np.std(mylist)/np.sqrt(len(mylist))
 
### print 95% CI considering the distribution to be gaussian
def print_95CI(mylist):
    return str(round(np.mean(mylist),6))+'±'+str(round(1.96*np.std(mylist)/np.sqrt(3000),6))
    #return np.mean(mylist)-1.96*np.std(mylist)/np.sqrt(3000),np.mean(mylist)+1.96*np.std(mylist)/np.sqrt(len(mylist))

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
    parser.add_argument("--n_data", default=0, type=int)
    parser.add_argument("--noise", default=0.1, type=float)
    parser.add_argument("--c1", default=0.5, type=float)
    parser.add_argument("--c2", default=0.5, type=float)
    parser.add_argument("--n_repetitions", default=1, type=int)
    parser.add_argument("--p1", default=0.5, type=float)
    parser.add_argument("--p2", default=0.5, type=float)

    args = parser.parse_args()
    print(args)

    ##################### experiments ########################

    n_aggr,n_aggr_empirical,mse_tot,mse_aggr,predictions_tot,predictions_aggr,y_test,w1_full,w2_full,w_aggr,list_corr,list_bound,list_s_squared,list_r2_full,list_r2_aggr,list_real_bound = perform_experiment(n_data=args.n_data,noise=args.noise,c1=args.c1,c2=args.c2, n_repetitions=args.n_repetitions, p1=args.p1, p2=args.p2)
    
    ##################### bias-variance estimates ########################

    avg_var_tot,inf_avg_var_tot,sup_avg_var_tot,avg_var_aggr,inf_avg_var_aggr,sup_avg_var_aggr,avg_bias_tot,inf_avg_bias_tot,sup_avg_bias_tot,avg_bias_aggr,inf_avg_bias_aggr,sup_avg_bias_aggr = compute_biasVariance(predictions_tot,predictions_aggr,y_test)

    ##################### print the results ########################

    print("Real and empirical aggregations: {0}, {1}\n".format(n_aggr,n_aggr_empirical))
    print("w1 full: {0}\n".format(print_95CI(w1_full)))
    print("w2 full: {0}\n".format(print_95CI(w2_full)))
    print("w aggr: {0}\n".format(print_95CI(w_aggr)))
    print("Sample correlation: {0}\n".format(print_95CI(list_corr)))
    print("Sample bound: {0}\n".format(print_95CI(list_bound)))
    print("Median Sample bound: {0}\n".format(np.median(list_bound)))
    print("Theoretical bound: {0}\n".format(print_95CI(list_real_bound)))
    print("Sample noise variance: {0}\n".format(print_95CI(list_s_squared)))
    print("Sample R2 full: {0}\n".format(print_95CI(list_r2_full)))
    print("Sample R2 aggr: {0}\n".format(print_95CI(list_r2_aggr)))
    
    print("Sample MSE full: {0}\n".format(print_95CI(mse_tot)))
    print("Sample MSE aggr: {0}\n".format(print_95CI(mse_aggr)))
    
    print("Sample variance full: {0}\n".format(str(round(avg_var_tot,6))+'±'+str(round(sup_avg_var_tot-avg_var_tot,6))))
    print("Sample variance aggr: {0}\n".format(str(round(avg_var_aggr,6))+'±'+str(round(sup_avg_var_aggr-avg_var_aggr,6))))
    
    print("Sample bias full: {0}\n".format(str(round(avg_bias_tot,6))+'±'+str(round(sup_avg_bias_tot-avg_bias_tot,6))))
    print("Sample bias aggr: {0}\n".format(str(round(avg_bias_aggr,6))+'±'+str(round(sup_avg_bias_aggr-avg_bias_aggr,6))))

