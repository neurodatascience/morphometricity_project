
# %%
# load the script
import numpy as np
from numpy import linalg
from numpy.core.numeric import Inf, identity
import pandas as pd
import csv
import itertools
import statsmodels.regression.mixed_linear_model as sm
import seaborn as sn
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


from reML import compute_Score, compute_FisherInfo, EM_update, morph_fit

#%%
# small test cases with simulated data to debug
# y = beta0 + beta0i + sum Xij betaj  + ei

# b0 = 1
# beta0i ~ N(0, ASM) (as a vector)
# betai ~ N(0, 1) i.i.d
# ei ~ N(0,2) i.i.d.

Z = np.array([[1,2,3], [2,1,2], [3,5,1], [1,0,-3], [0,5,2]]) #N=5, M=3
ASM = pd.DataFrame(Z.T).corr() 

X = np.array([[0,1,20],[1,1,30],[1,0,40],[0,1,35], [0,0,25]])

N, L = X.shape
beta = np.random.normal(loc=0, scale=1, size = L)
beta0i = np.random.multivariate_normal(mean = [0] * N, cov = 8*ASM)
eps = np.random.normal(loc=0, scale=2, size = N)
y = 1 + beta0i + beta.dot(X.T) + eps


#morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=10)
#morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=20)
morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=50)
morph_fit(y=y, X=X, K=ASM, method="expected", max_iter=50)
morph_fit(y=y, X=X, K=ASM, method="average", max_iter=50)


# %%
# simulation based on the age/sex distribution from ukb
[N,M,L] = [50, 100, 2]
[Va,Ve] = [8,2]
Z = np.random.normal(0,2, size = (N,M))
ASM = pd.DataFrame(Z.T).corr() 

age = np.random.normal(56, 8 ,size=(N,1))
sex = np.random.binomial(N, 0.54, size=(N,1))
X = np.concatenate((age, sex), axis=1)
 

beta = np.random.normal(loc=0, scale=1, size = L)
beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)
eps = np.random.normal(loc=0, scale= Ve**(1/2), size = N)
# morphometricity should be 0.8 by definition

y = beta0i + beta.dot(X.T) + eps
 
ASM = ASM.values


morph_fit(y=y, X=X, K=ASM, method="expected", max_iter=50)
 
# when converge, in general under estimate the morphometricity. 





# %%
[N,M,L] = [50, 100, 2]
[Va,Ve] = [8,2]
n_sim = 500
res = np.ndarray(shape = (n_sim, 7))

for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM = pd.DataFrame(Z.T).corr() 

    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)


    beta = np.random.normal(loc=0, scale=1, size = L)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
    ASM = ASM.values


    temp = morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=200)
    res[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
    temp['iteration'], temp['Estimated morphometricity'], 
    temp['Estimated standard error'], temp['Morphological variance'], 
    temp['Residual variance'], temp['ReML likelihood'] ]
    print(temp['Morphological variance'])

res = res[res[:,0]==1] # subset those who converged 91/100
res = res[np.isnan(res[:,3])==False]
print(res.shape[0]/n_sim)

# obs:
# 9.8% covergence rate for 0.8 morphometricity
# 15.6% for 0.5 morph
# 15.4% for 0.2 morph
# exp: 99.4% covergence rate

# %%

# plot the distribution of estimated morphometricity
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
ax1.text(0.4,3,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')

ax2.hist(res[:,3], density=True)
txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
ax2.text(0.05,8,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

fig.tight_layout()
#plt.show()
plt.savefig('sim_res_avg.png',dpi=150)

# %%
# simulation with real Z:

sub_dat = pd.read_csv("sub_dat.csv")
[N, M] = sub_dat.shape
L = 2
# generate anatomic similarity matrix 
subject = sub_dat["eid"]
Z = sub_dat.drop(["eid","Unnamed: 0"], axis=1)
N = Z.shape[0]
# Z.shape = (100, 124) N=100, M=124
ASM = pd.DataFrame(Z.T).corr() 
ASM = ASM.values 
# almost all values = 1, causing error in estimation?
age = np.random.normal(56, 8 ,size=(N,1))
sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb
X = np.concatenate((age, sex), axis=1)

beta = np.random.normal(loc=0, scale=10, size = L)
beta0i = np.random.multivariate_normal(mean = [0] * N, cov = 8*ASM)
eps = np.random.normal(loc=0, scale=2**(1/2), size = N)
y = 1 + beta0i + beta.dot(X.T) + eps

# X = (X - np.mean(X, axis=0))/np.std(X, axis= 0)
# y = (y - y.mean())/(y.std())
# %%
morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=100)
morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=500)
morph_fit(y=y, X=X, K=ASM, method="expected", max_iter=100)
# negative variance estimate by observed fisher, does not converge, explosing Va, Ve without standardization
# very low convergence rate (same negative variance problem) with standardized Va, Ve 


# under-estimated m2 by expected (due to almost 1 ASM? ) -> need some filtering on imaging measurements?
# normalizing x and y will speed up convergence but already very fast with expected fisher
# %%
# increase sample size N = 100


[N,M,L] = [100, 100, 2]
[Va,Ve] = [8,2]
n_sim = 500
res = np.ndarray(shape = (n_sim, 7))

for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM = pd.DataFrame(Z.T).corr() 

    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)


    beta = np.random.normal(loc=0, scale=1, size = L)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
    ASM = ASM.values


    temp = morph_fit(y=y, X=X, K=ASM, method="expected", max_iter=50)
    res[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
    temp['iteration'], temp['Estimated morphometricity'], 
    temp['Estimated standard error'], temp['Morphological variance'], 
    temp['Residual variance'], temp['ReML likelihood'] ]
    #print(temp['Morphological variance'])

res = res[res[:,0]==1] # subset those who converged  
res = res[np.isnan(res[:,3])==False]
print(res.shape[0]/n_sim)

 


# plot the distribution of estimated morphometricity
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
ax1.text(0.55,3,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')

ax2.hist(res[:,3], density=True)
txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
ax2.text(0.05,8,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

fig.tight_layout()
#plt.show()
plt.savefig('sim_res_exp_n100.png',dpi=150)
# %%
# increase sample size N = 500


[N,M,L] = [500, 100, 2]
[Va,Ve] = [8,2]
n_sim = 500
res = np.ndarray(shape = (n_sim, 7))

for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM = pd.DataFrame(Z.T).corr() 

    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)


    beta = np.random.normal(loc=0, scale=1, size = L)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
    ASM = ASM.values


    temp = morph_fit(y=y, X=X, K=ASM, method="expected", max_iter=50)
    res[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
    temp['iteration'], temp['Estimated morphometricity'], 
    temp['Estimated standard error'], temp['Morphological variance'], 
    temp['Residual variance'], temp['ReML likelihood'] ]
    print(temp['Morphological variance'])

res = res[res[:,0]==1] # subset those who converged  
res = res[np.isnan(res[:,3])==False]
print(res.shape[0]/n_sim)

 


# plot the distribution of estimated morphometricity
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
ax1.text(0.6,3,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')

ax2.hist(res[:,3], density=True)
txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
ax2.text(0.05,8,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

fig.tight_layout()
#plt.show()
plt.savefig('sim_res_exp_n500.png',dpi=150)

# large variance problem solved with increasing N (expected Fisher)
# try others
# %%

[N,M,L] = [100, 100, 2]
[Va,Ve] = [8,2]
n_sim = 500
res = np.ndarray(shape = (n_sim, 7))

for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM = pd.DataFrame(Z.T).corr() 

    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)


    beta = np.random.normal(loc=0, scale=1, size = L)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
    ASM = ASM.values


    temp = morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=50)
    res[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
    temp['iteration'], temp['Estimated morphometricity'], 
    temp['Estimated standard error'], temp['Morphological variance'], 
    temp['Residual variance'], temp['ReML likelihood'] ]
    print(temp['Morphological variance'])

res = res[res[:,0]==1] # subset those who converged  
res = res[np.isnan(res[:,3])==False]
print(res.shape[0]/n_sim)
# out of 500 runs only 9.2% converges within 50 iterations...


# %%
# plot the distribution of estimated morphometricity
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
ax1.text(0.5,4,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')

ax2.hist(res[:,3], density=True)
txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
ax2.text(0.10,15,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

fig.tight_layout()
#plt.show()
plt.savefig('sim_res_obs_n100.png',dpi=150)



# increase sample size N = 500


[N,M,L] = [500, 100, 2]
[Va,Ve] = [8,2]
n_sim = 500
res = np.ndarray(shape = (n_sim, 7))

for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM = pd.DataFrame(Z.T).corr() 

    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)


    beta = np.random.normal(loc=0, scale=1, size = L)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
    ASM = ASM.values


    temp = morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=50)
    res[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
    temp['iteration'], temp['Estimated morphometricity'], 
    temp['Estimated standard error'], temp['Morphological variance'], 
    temp['Residual variance'], temp['ReML likelihood'] ]
    print(temp['Morphological variance'])

res = res[res[:,0]==1] # subset those who converged  
res = res[np.isnan(res[:,3])==False]
print(res.shape[0]/n_sim)
# out of 500 runs 21.4% converges within 50 iterations...


# %%
# plot the distribution of estimated morphometricity
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
ax1.text(0.85,8,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')

ax2.hist(res[:,3], density=True)
txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
ax2.text(0.01,50,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

fig.tight_layout()
#plt.show()
plt.savefig('sim_res_obs_n500.png',dpi=150)
# %%
# summary on the larger sample size simulations:
# 1. For expected Fisher (which is the best in small sample size):
#    covergences in 2-3 iterations, faster and more accurate (smaller 
#    variance) with larger N
# 2. For observed Fisher (problematic in the proportion in convergence):
#    proportion of convergence within 50 iterations gets larger 
#    (N=50 1.6% -> N=100 9.4% -> N=500 21.4%)
#    Same pattern as expected Fisher, higher accuracy, smaller variance 
#
# Todo: SE estimates seem too optimistic -> double check (extra variance), increase N 
#       take the same sample size as 
#       set up on Beluga


# Tried with N = 5000, computing the ASM takes ~27s
#                      computing the morphometricity takes ~ 2m47s
#                      converges at 2nd iteration so it's not the convergence problem
# The current algorithm does not scale up well with biobank size data becuase of the matrix inversion
# Checking the draft paper of Yi Yang now...

# -> Check other methods to approximate the inversion...

# %% increase N further:

[N,M,L] = [5000, 100, 2]
[Va,Ve] = [8,2]
n_sim = 500
res = np.ndarray(shape = (n_sim, 7))

for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM = pd.DataFrame(Z.T).corr()
    # ASM = Z.dot(Z.T) ?
    # np correlation ?

    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)


    beta = np.random.normal(loc=0, scale=1, size = L)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
    ASM = ASM.values


    temp = morph_fit(y=y, X=X, K=ASM, method="expected", max_iter=50)
    res[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
    temp['iteration'], temp['Estimated morphometricity'], 
    temp['Estimated standard error'], temp['Morphological variance'], 
    temp['Residual variance'], temp['ReML likelihood'] ]
    print([i,temp['Morphological variance']])

res = res[res[:,0]==1] # subset those who converged  
res = res[np.isnan(res[:,3])==False]
print(res.shape[0]/n_sim)
# out of 500 runs 21.4% converges within 50 iterations...


# %%
# plot the distribution of estimated morphometricity
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
ax1.text(0.85,8,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')

ax2.hist(res[:,3], density=True)
txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
ax2.text(0.01,50,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

fig.tight_layout()
#plt.show()
plt.savefig('sim_res_obs_n500.png',dpi=150)



# %%

# Source of extra variance may due to the data generating process?
# Fixed by the following simulation. Method for estimating se(m2) works.

[N,M,L] = [500, 100, 2]
[Va,Ve] = [8,2]
beta = np.random.normal(loc=0, scale=1, size = L)
n_sim = 500
res = np.ndarray(shape = (n_sim, 7))

for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM = pd.DataFrame(Z.T).corr() 

    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
    ASM = ASM.values


    temp = morph_fit(y=y, X=X, K=ASM, method="expected", max_iter=50)
    res[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
    temp['iteration'], temp['Estimated morphometricity'], 
    temp['Estimated standard error'], temp['Morphological variance'], 
    temp['Residual variance'], temp['ReML likelihood'] ]
    print([i,temp['Morphological variance']])

res = res[res[:,0]==1] # subset those who converged  
res = res[np.isnan(res[:,3])==False]
print(res.shape[0]/n_sim)
# out of 500 runs 21.4% converges within 50 iterations...


# %%
# plot the distribution of estimated morphometricity
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
ax1.text(0.7,8,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')


ax2.hist(res[:,3], density=True)
txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
ax2.text(0.02,80,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

fig.tight_layout()
#plt.show()
plt.savefig('sim_res_obs_n500.png',dpi=150)
# %%


# draw some random sample from ukb that of same size 
# as Sabuncu's paper to quickly replicate the estimated morphometricity
# change the code for computing Z.dot(Z.T)
# find approximation of matrix inversion
# time the matlab code as well
# %%

os.chdir('/Users/tingzhang/Documents/GitHub/morphometricity_project/data')

instance2_dat = pd.read_csv('sub_dat.csv')

y = instance2_dat['age_at_recruitment']
X = instance2_dat['sex'].to_numpy().reshape(-1,1)

Z = instance2_dat.drop(['eid','age_at_recruitment','sex'], axis=1)
 

Z = (Z - np.mean(Z, axis=0))/np.std(Z, axis= 0)
ASM = np.cov(Z)
#morph_fit(y=y, X=X, K=ASM, method="expected", max_iter=50)
 

# %%
y = instance2_dat['sex']
X = instance2_dat['age_at_recruitment'].to_numpy().reshape(-1,1)

Z = instance2_dat.drop(['eid','age_at_recruitment','sex'], axis=1)
 

Z = (Z - np.mean(Z, axis=0))/np.std(Z, axis= 0)
ASM = np.cov(Z)
morph_fit(y=y, X=X, K=ASM, method="expected", max_iter=50)
 
# %%
