# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:55:05 2018

@author: StepanAsus
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS as lm
from numpy import transpose as t
from numpy.linalg import inv as inv
from scipy.stats import chi2 as chi2

#%%

# Exercise 1
np.random.seed(1000)

mc_rep = 5000
n = 100
k = 10
a = np.array([1, 0.6, 0.3, 0.15, 0.07, 0.04, 0.02, 0])
rho = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95])
e10 = np.zeros(k)
Z = np.random.normal(0, 1, ([n,k]))
t_stat = np.zeros([mc_rep])
rej_freq = np.zeros([len(a), len(rho)])

for i in range(len(a)):
  for p in range(len(rho)):
    np.random.seed(1000)
    for j in range(mc_rep):
      res = np.random.multivariate_normal(mean=[0, 0], cov=np.array([[1, rho[p]], [rho[p], 1]]), 
                                          size=(n))
      eps, v = res[:, 0], res[:, 1]
      pi = e10
      pi[0] = a[i]
      x = Z @ pi + v
      y = eps
      beta_2sls = (x.T @ Z @ inv(Z.T @ Z) @ Z.T @ x) ** (-1) * (x.T @ 
                        Z @ inv(Z.T @ Z) @ Z.T @ y)
      s_2sls = np.dot((y - x * beta_2sls).T, (y - x * beta_2sls)) / (n - 1)
      var_2sls = s_2sls * (x.T @ Z @ inv(Z.T @ Z) @ Z.T @ x) ** (-1)
      t_stat[j] = abs(beta_2sls / np.sqrt(var_2sls)) > 1.96
    rej_freq[i, t] = (np.mean(t_stat))
      

#%%
    
# Exercise 2
np.random.seed(1000)

r_b0 = np.arange(0, 200, step=0.5)
crit_values_lr = np.zeros([len(r_b0), ])

for i in range(len(r_b0)):
  psi_k = np.random.chisquare(k - 1, size=mc_rep)
  psi_1 = np.random.chisquare(1, size=mc_rep)
  
  lr_b0 = (psi_k + psi_1 - r_b0[i] + np.sqrt((psi_k + psi_1 + r_b0[i]) ** 2 - 4 * 
           r_b0[i] * psi_k)) / 2
  crit_values_lr[i] = np.percentile(lr_b0, q=95)
  
  
#%%
  
# Exercise 3
np.random.seed(1000)

mc_rep = 5000

# all in one loop
ar_stat = np.zeros([mc_rep])
rej_freq_ar = np.zeros([len(a), len(rho)])

lr_stat = np.zeros([mc_rep])
rej_freq_lr = np.zeros([len(a), len(rho)])

lm_stat = np.zeros([mc_rep])
rej_freq_lm = np.zeros([len(a), len(rho)])

for i in range(len(a)):
  for p in range(len(rho)):
    np.random.seed(1000)
    for j in range(mc_rep):
      # data
      res = np.random.multivariate_normal(mean=[0, 0], cov=np.array([[1, rho[p]], 
                                                                    [rho[p], 1]]), size=(n))
      eps, v = res[:, 0], res[:, 1]
      pi = e10
      pi[0] = a[i]
      x = Z @ pi + v
      y = eps
      pz = Z @ inv(Z.T @ Z) @ Z.T
      
      # AR
      ar_stat[j] = (y.T @ pz @ y) / (y.T @ (np.identity(pz.shape[0]) - pz) @ 
             y) * ((n - k) / k) 
      
      # LM
      rho_hat = (y.T @ (np.identity(pz.shape[0]) - pz) @ x) / (y.T @ 
           (np.identity(pz.shape[0]) - pz) @ y)
      pi_b0 = inv(Z.T @ Z) @ Z.T @ (x - y * rho_hat)
      pz_pi = np.outer(Z @ pi_b0, pi_b0.T @ Z.T) * ((pi_b0.T @ Z.T @ Z @ pi_b0) ** (-1))
      lm_stat[j] = (y.T @ pz_pi @ y) / (y.T @ (np.identity(pz.shape[0]) - pz) @ y) * (n - k)
      
      # LR
      psi_k = np.random.chisquare(k - 1, size=mc_rep)
      psi_1 = np.random.chisquare(1, size=mc_rep)
      r_b0 = (pi_b0.T @ Z.T @ Z @ pi_b0) / (((x.T @ 
             (np.identity(pz.shape[0]) - pz) @ x) - (((y.T @ 
             (np.identity(pz.shape[0]) - pz) @ x) ** 2) / 
             (y.T @ (np.identity(pz.shape[0]) - pz) @ y))) / (n - k))
      
      lr_b0 = (psi_k + psi_1 - r_b0 + np.sqrt((psi_k + psi_1 + r_b0) ** 2 - 4 * 
               r_b0 * psi_k)) / 2
      crit_values = np.percentile(lr_b0, q=95)
      lr_stat[j] = (1 / 2) * (k * ar_stat[j] - r_b0 + np.sqrt((k * ar_stat[j] + r_b0) ** (2) - 
             4 * r_b0 * (k * ar_stat[j] - lm_stat[j]))) > crit_values
    
    rej_freq_ar[i, p] = np.mean(ar_stat > (chi2.ppf(0.95, 10) / k))
    rej_freq_lr[i, p] = np.mean(lr_stat)
    rej_freq_lm[i, p] = np.mean(lm_stat > chi2.ppf(0.95, 1))


#%%
    
# Exercise 4
np.random.seed(1000)

r_b0 = np.arange(0, 200, step=0.5)
crit_values_lr_2 = np.zeros([len(r_b0), ])

for i in range(len(r_b0)):
  psi_k = np.random.chisquare(4, size=mc_rep)
  psi_1 = np.random.chisquare(1, size=mc_rep)
  
  lr_b0 = (psi_k + psi_1 - r_b0[i] + np.sqrt((psi_k + psi_1 + r_b0[i]) ** 2 - 4 * 
           r_b0[i] * psi_k)) / 2
  crit_values_lr_2[i] = np.percentile(lr_b0, q=95)


#%%
  
# data load for Exercise 5
np.random.seed(1000)

data = pd.read_csv("dest.csv", header=None)
data.columns = ["age", "age2", "ed", "exper", "exper2", "nearc2", "nearc4", "nearc4a", "nearc4b",
                      "race", "smsa", "South", "wage"]
y = data.loc[:, 'wage']
x = data.loc[:, 'ed']
z = data.loc[:, ("nearc2", "nearc4", "nearc4a", "nearc4b")]
w = data.loc[:, ("exper", "exper2", "race", "smsa", "South")]

#%%

# Ex 5a, 2SLS and AR
np.random.seed(1000)

betas = np.arange(-50, 50, 1)

new_x = np.array(lm(x, w).fit().fittedvalues)
new_y = np.array(lm(y, w).fit().fittedvalues)
new_z = np.array([lm(np.array(z)[:, 0], w).fit().fittedvalues, 
                 lm(np.array(z)[:, 1], w).fit().fittedvalues,
                 lm(np.array(z)[:, 2], w).fit().fittedvalues,
                 lm(np.array(z)[:, 3], w).fit().fittedvalues]).T

# t-statistic
pz_5a = (np.outer(new_z[:, 0], new_z[:, 0].T)  * ((new_z[:, 0].T @ new_z[:, 0]) ** (- 1)))
beta_2sls = ((new_x.T @ pz_5a @ new_x) ** (- 1)) * new_x.T @ pz_5a @ new_y
var_2sls = ((new_y - new_x * beta_2sls).T @ (new_y - new_x * beta_2sls) * ((new_x.T
           @ pz_5a @ new_x) ** (- 1))) / (3010 - 1)
se_2sls = np.sqrt(var_2sls)
t_stat_5a = beta_2sls / se_2sls
ci_2sls_5a = (beta_2sls - 1.96 * se_2sls, beta_2sls + 1.96 * se_2sls)

for i in betas:
  



z_5a = mw @ np.array(z.loc[:, 'nearc2'])
pz_5a = np.outer(z_5a, np.dot(np.int(np.dot(t(z_5a), z_5a)) ** np.int(-1), t(z_5a)))
mz_5a = np.identity(pz_5a.shape[0]) - pz_5a
ar_5a = (t(y_5a) @ pz_5a @ y_5a) / (t(y_5a) @ mz_5a @ y_5a) * ((pz_5a.shape[0] - 1) / 1)

# alternative
ar_regr_5a = lm(y, z_5a).fit()
ar_regr_5a.f_test('x1 = 0')

# confidence intervals using AR?

#%%

# Ex 5e, 2SLS

np.random.seed(1000)

fs_5e = lm(x, pd.concat([w, z], axis=1), hasconst=True).fit()
print(fs_5e.summary())

ss_5e = lm(y, pd.concat([fs_5e.fittedvalues.rename('ed'), w, z], axis=1),
           hasconst=True).fit()
print(ss_5e.summary())

ss_5e.conf_int(alpha=0.05)
ss_5e.t_test('ed = 0') # gives us same results

# Ex 5e, AR

mw = w_np @ inv(t(w_np) @ w_np) @ t(w_np)
z_5e = mw @ np.array(z)
y_5e = mw @ np.array(y)
pz_5e = z_5e @ inv(t(z_5e) @ z_5e) @ t(z_5e)
mz_5e = np.identity(pz_5e.shape[0]) - pz_5e
ar_5e = (t(y_5e) @ pz_5e @ y_5e) / (t(y_5e) @ mz_5e @ y_5e) * ((pz_5e.shape[0] - 4) / 4)

# alternative
ar_regr_5e = lm(y, z_5e).fit()
ar_regr_5e.f_test('x1 = 0, x2 = 0, x3 = 0, x4 = 0')

# confidence intervals using AR?








