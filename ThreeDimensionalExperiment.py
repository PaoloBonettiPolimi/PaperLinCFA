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

### generate a dataset of n samples and standardize the variables x1,x2,x3
def generate_dataset_multi(n=3000, noise=1, c1=0.5, c2=0.5, c3=0.5, p1=0.5, p2=0.5, p3=0.5, p4=0.5, p5=0.5):
    x = np.zeros((n,3))
    
    x[:,0] = np.random.uniform(size=n)
    x[:,1] = p1*np.random.uniform(size=n) + p2*x[:,0]
    x[:,2] = p3*np.random.uniform(size=n) + p4*x[:,0] + p5*x[:,0]
    coeffs = np.array([c1,c2,c3]).reshape(3,1)
    
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    
    delta = np.random.normal(0, noise, size=(n,1))
    
    y = np.dot(x,coeffs).reshape(n,1) + delta.reshape(n,1)
    return x,y,coeffs

### compute the correlation threshold with empirical data
def compute_bound_multi(x1,x2,x3,y):

    x = np.zeros((len(x1),3))
    x[:,0] = x1
    x[:,1] = x2
    x[:,2] = x3

    regr = LinearRegression()
    regr.fit(x,y)
    w1 = regr.coef_[0][0]
    w2 = regr.coef_[0][1]
    w3 = regr.coef_[0][2]
    preds = regr.predict(x)
    residuals = y - preds
    
    n=len(x1)
    s_squared = np.dot(residuals.reshape(1,n),residuals)/(n-4)
    
    a = s_squared/((n-1)*((w1-w2)**2))
    b = (compute_corr(x1,x3)-compute_corr(x2,x3))*w3/(w1-w2)
    lower_bound,upper_bound = 1-(a-b)-np.sqrt(a*(a-2*b)), 1-(a-b)+np.sqrt(a*(a-2*b))
    corr = compute_corr(x1.reshape(n),x2.reshape(n))
        
    return a,b,lower_bound,upper_bound,corr,s_squared,w1,w2,w3


### compute the theoretical correlation threshold
def compute_real_bound_multi(w1,w2,w3,n,s_squared, x1, x2, x3):
    a = s_squared/((n-1)*((w1-w2)**2))
    b = (compute_corr(x1,x3)-compute_corr(x2,x3))*w3/(w1-w2)
    return a,b,1-(a-b)-np.sqrt(a*(a-2*b)), 1-(a-b)+np.sqrt(a*(a-2*b))

### three-dimensional experiment, repeated n_repetitions times
def multi_experiment(n_repetitions=500, n_data=3000, noise=0.5, c1=0.4, c2=0.6, c3=0.2, p1=0.4, p2=0.6, p3=0.5, p4=0.5, p5=0.5):
    mse_tot = []
    mse_aggr = []
    list_corr = []
    list_upper_bound = []
    list_lower_bound = []
    list_s_squared = []
    w1_full = []
    w2_full = []
    w3_full = []
    w1_aggr = []
    w2_aggr = []
    list_r2_full = []
    list_r2_aggr = []
    list_upper_bound_real= []
    list_lower_bound_real = []
    predictions_tot = np.zeros((int(n_data),n_repetitions))
    predictions_aggr = np.zeros((int(n_data),n_repetitions))
    n_aggr = 0 
    n_aggr_empirical = 0
    
    x_test,y_test,coeffs_test = generate_dataset_multi(n=n_data, noise=noise, c1=c1, c2=c2, c3=c3, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5)
    x_test_aggr = (x_test[:,0]+x_test[:,1])/2
    
    for i in range(n_repetitions):
        x,y,coeffs = generate_dataset_multi(n=n_data, noise=noise, c1=c1, c2=c2, c3=c3, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5)
        features_df = pd.DataFrame(x[:,0:2]) # x1 e x2, non x3
        target_df = pd.DataFrame(y)
        aggregate_col = (x[:,0] + x[:,1]) / 2
        aggregations_df = pd.DataFrame(aggregate_col) # solo colonna con x1+x2/2
        
        a,b,lower_bound,upper_bound,corr,s_squared,w1,w2,w3 = compute_bound_multi(x[:,0],x[:,1],x[:,2],target_df.values)
        list_corr.append(corr)
        list_upper_bound.append(upper_bound)
        list_lower_bound.append(lower_bound)
        list_s_squared.append(s_squared)
        w1_full.append(w1)
        w2_full.append(w2)
        w3_full.append(w3)

        regr_full = LinearRegression().fit(features_df, target_df)
        list_r2_full.append(regr_full.score(x_test[:,0:2], y_test))
        regr_aggr = LinearRegression().fit(aggregations_df, target_df)
        mse_tot.append(mean_squared_error(y_test,regr_full.predict(x_test[:,0:2])))
        mse_aggr.append(mean_squared_error(y_test,regr_aggr.predict(x_test_aggr.reshape(-1, 1))))
        predictions_tot[:,i] = regr_full.predict(x_test[:,0:2])[:,0]
        predictions_aggr[:,i] = regr_aggr.predict(x_test_aggr.reshape(-1, 1))[:,0]
        w1_aggr.append(regr_aggr.coef_[0][0])
        list_r2_aggr.append(regr_aggr.score(x_test_aggr.reshape(-1, 1), y_test))
        
        a,b,lower_bound_real,upper_bound_real = compute_real_bound_multi(w1=c1,w2=c2,w3=c3,n=n_data,s_squared=noise*noise, x1=x[:,0], x2=x[:,1], x3=x[:,2])
        list_upper_bound_real.append(upper_bound_real)
        list_lower_bound_real.append(lower_bound_real)

        if ((corr>=lower_bound_real) & (corr<=upper_bound_real)) : 
            n_aggr +=1
            
        if ((corr>=lower_bound) & (corr<=upper_bound)):
            n_aggr_empirical += 1
            
    return n_aggr,n_aggr_empirical,mse_tot,mse_aggr,predictions_tot,predictions_aggr,y_test,list_corr,list_upper_bound,list_lower_bound,list_s_squared,w1_full,w2_full,w3_full,list_r2_full,w1_aggr,w2_aggr,list_r2_aggr,list_upper_bound_real,list_lower_bound_real


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

    
    return avg_var_tot,avg_var_aggr,avg_bias_tot,avg_bias_aggr
        
### main run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_data", default=0, type=int)
    parser.add_argument("--noise", default=0.1, type=float)
    parser.add_argument("--c1", default=0.5, type=float)
    parser.add_argument("--c2", default=0.5, type=float)
    parser.add_argument("--c3", default=0.5, type=float)
    parser.add_argument("--n_repetitions", default=1, type=int)
    parser.add_argument("--p1", default=0.5, type=float)
    parser.add_argument("--p2", default=0.5, type=float)
    parser.add_argument("--p3", default=0.5, type=float)
    parser.add_argument("--p4", default=0.5, type=float)
    parser.add_argument("--p5", default=0.5, type=float)

    args = parser.parse_args()
    print(args)

    ##################### experiments ########################

    n_aggr,n_aggr_empirical,mse_tot,mse_aggr,predictions_tot,predictions_aggr,y_test,list_corr,list_upper_bound,list_lower_bound,list_s_squared,w1_full,w2_full,w3_full,list_r2_full,w1_aggr,w2_aggr,list_r2_aggr,list_upper_bound_real,list_lower_bound_real = multi_experiment(n_repetitions=args.n_repetitions, n_data=args.n_data, noise=args.noise, c1=args.c1, c2=args.c2, c3=args.c3, p1=args.p1, p2=args.p2, p3=args.p3, p4=args.p4, p5=args.p5)
    
    ##################### bias-variance estimates ########################

    avg_var_tot,inf_avg_var_tot,sup_avg_var_tot,avg_var_aggr,inf_avg_var_aggr,sup_avg_var_aggr,avg_bias_tot,inf_avg_bias_tot,sup_avg_bias_tot,avg_bias_aggr,inf_avg_bias_aggr,sup_avg_bias_aggr = compute_biasVariance(predictions_tot,predictions_aggr,y_test)

    ##################### print the results ########################

    print("Real and empirical aggregations: {0}, {1}\n".format(n_aggr,n_aggr_empirical))
    print("w1 full: {0}\n".format(print_95CI(w1_full)))
    print("w2 full: {0}\n".format(print_95CI(w2_full)))
    print("w3 full: {0}\n".format(print_95CI(w3_full)))
    print("w aggr: {0}\n".format(print_95CI(w1_aggr)))
    print("Sample correlation: {0}\n".format(print_95CI(list_corr)))
    print("Sample bound median: {0}\n".format([np.nanmedian(list_lower_bound),np.nanmedian(list_upper_bound)]))
    print("Sample bound real: {0}\n".format([print_95CI(list_lower_bound_real),print_95CI(list_upper_bound_real)]))

    print("Sample noise variance: {0}\n".format(print_95CI(list_s_squared)))
    print("Sample R2 full: {0}\n".format(print_95CI(list_r2_full)))
    print("Sample R2 aggr: {0}\n".format(print_95CI(list_r2_aggr)))

    print("Sample MSE full: {0}\n".format(print_95CI(mse_tot)))
    print("Sample MSE aggr: {0}\n".format(print_95CI(mse_aggr)))

    print("Sample variance full: {0}\n".format(str(round(avg_var_tot,6))+'±'+str(round(sup_avg_var_tot-avg_var_tot,6))))
    print("Sample variance aggr: {0}\n".format(str(round(avg_var_aggr,6))+'±'+str(round(sup_avg_var_aggr-avg_var_aggr,6))))
    
    print("Sample bias full: {0}\n".format(str(round(avg_bias_tot,6))+'±'+str(round(sup_avg_bias_tot-avg_bias_tot,6))))
    print("Sample bias aggr: {0}\n".format(str(round(avg_bias_aggr,6))+'±'+str(round(sup_avg_bias_aggr-avg_bias_aggr,6))))
