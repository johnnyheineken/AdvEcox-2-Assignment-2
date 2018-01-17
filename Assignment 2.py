#%%
import numpy as np
from numpy import zeros, random, sqrt, mean, percentile
from numpy import transpose as t
from numpy.linalg import inv
from matplotlib import pyplot as plt
a_list = [1, 0.6, 0.3, 0.15, 0.07, 0.04, 0.02, 0]
rho_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
N=100
k=10
beta = 0
REP = 1000

#%%
'''
EXERCISE 1
'''

Z = random.normal(size = (N,k))

results = zeros((len(a_list), len(rho_list)))
counter_a = 0
for a in a_list:
    PI = zeros(shape=k)
    PI[0] = a
    counter_rho = 0
    for rho in rho_list:
        SIGMA = np.array([[1, rho], [rho, 1]])
        tstat_result = zeros(REP)
        random.seed(2110)

        for i in range(REP):
            
            eps_V = random.multivariate_normal(mean=[0, 0], cov=SIGMA, size=N)
            epsilon = eps_V[:, 0]
            V = eps_V[:, 1]
            X = Z @ PI + V
            Y = X * beta + epsilon
            
            PI_hat = inv(t(Z) @ Z) @ t(Z) @ X
            X_hat = Z @ PI_hat
            beta_hat = (1 / (t(X_hat) @ X_hat)) * t(X_hat) @ Y
            s_2 = (t(Y - X * beta_hat) @ (Y - X * beta_hat)) / (N - 1)
            var = s_2 * (1 / (t(X_hat) @ X_hat))
            tstat_result[i] = beta_hat / sqrt(var)
        rej_rate = mean((tstat_result > 1.96) | (tstat_result < -1.96))
        results[counter_a, counter_rho] = rej_rate
        print(rej_rate)
        counter_rho += 1
    counter_a += 1

print(results)
#%%


num_plots = results.shape[0]
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, num_plots)])

labels = []
for i in range(num_plots):
    plt.plot(rho_list, results[i, :])
    labels.append('a = ' + str(a_list[i]))

plt.legend(labels)
plt.xlabel('value of rho')
plt.ylabel('rejection frequency')
plt.show()


#%%
'''
EXERCISE 2
'''
k = 10
grid = [i * 0.25 for i in range(1000)]
S = 5000

results = zeros(len(grid))
counter = 0
for r_b0 in grid:
    random.seed(2110)
    LR = zeros(S)
    psi_1 = random.chisquare(1, size=S)
    psi_k = random.chisquare(k-1, size=S)
    LR = 1/2 * (psi_k + psi_1 - r_b0 + \
             sqrt(((psi_k + psi_1 + r_b0) ** 2) - 4 * r_b0 * psi_k))
    results[counter] = percentile(LR, q=95)
    counter += 1
print(results)

plt.plot(grid, results)
plt.xlabel('r(β)')
plt.ylabel('95% criticial value')
plt.show()


#%%
'''
EXERCISE 4
'''
k = 4
grid = [i * 0.25 for i in range(1000)]
S = 5000

results = zeros(len(grid))
counter = 0
for r_b0 in grid:
    random.seed(2110)
    LR = zeros(S)
    psi_1 = random.chisquare(1, size=S)
    psi_k = random.chisquare(k - 1, size=S)
    LR = 1 / 2 * (psi_k + psi_1 - r_b0 +
                  sqrt(((psi_k + psi_1 + r_b0) ** 2) - 4 * r_b0 * psi_k))
    results[counter] = percentile(LR, q=95)
    counter += 1
print(results)

plt.plot(grid, results)
plt.xlabel('r(β)')
plt.ylabel('95% criticial value')
plt.show()
