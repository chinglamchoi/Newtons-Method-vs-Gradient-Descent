"""
PROBLEMS:
- No alpha for NM, bad alpha for GD: SOLVED
- NaN values for GD: SOLVED
- Hardcoded starting theta for NM & GD: !!!!!
- Timing: Need per epoch & Need hundred iters
"""

from sklearn import datasets
import numpy as np
import numpy.linalg as lnp
from time import time
import matplotlib.pyplot as plt

#np.random.seed(1)

#data, valid = datasets.load_boston(), False
data, valid = datasets.load_iris(), True
X = data["data"]
Y = data["target"]
m = Y.shape[0]

def MSE(dist):
	global m
	return lnp.norm(dist**2, 1)/(2*m)

# for the bias:
X = np.insert(X, 0, 1, axis=1)
if valid:
	margin = 0.0233 #global minima at 3 sig fig for MSE on this distribution & overflow
else:
	margin = 11 #preliminary

#random init: sample to make better
#np.random.seed(1)
#theta = np.ndarray((X.shape[1]))
theta = np.full((X.shape[1]), 5)
y_hat = np.matmul(X, theta)
dist = Y-y_hat
X_T = X.T
cost = MSE(dist) #no need transpose because np doesn't differentiate between column & row vectors

def nm(theta, y_hat, X_T, dist, cost, lr):
	start = time()
	global X, Y, margin, m, indices
	cost_log = []
	epoch = 0
	hessian = np.matmul(X_T, X)
	inv_hessian = lnp.inv(hessian)

	while cost > margin:
		#cost_log.append(cost) #commented for timing
		#print("Epoch #" + str(epoch) + ":", cost)
		cost_log.append(cost)
		epoch += 1
		grad = np.matmul(X_T, dist/-m)
		theta = theta-lr*np.matmul(inv_hessian, grad)
		
		y_hat = np.matmul(X, theta)
		dist = Y-y_hat
		X_T = X.T
		cost = MSE(dist)
	cost_log.append(cost)
	end = time()
	print("\n" + "Finished NM in " + str(epoch+1), "epochs with error " + str(cost_log[-1]) + "\n")
	print("Optimal theta:", theta)
	print("\n\n" + "y_hat, y")
	for i in indices:
		print(y_hat[i], Y[i])

	return end-start, cost_log

def gd(theta, y_hat, X_T, dist, cost, lr):
	start = time()
	global X, Y, margin, m, indices
	cost_log = []
	epoch = 0
	
	while cost > margin:
		cost_log.append(cost)
		#cost_log.append(cost) #commented for timing
		epoch += 1
		grad = np.matmul(X_T, dist)/-m
		theta = theta-lr*grad

		y_hat = np.matmul(X, theta)
		dist = Y-y_hat
		X_T = X.T
		cost = MSE(dist)
	cost_log.append(cost)
	end = time()
	print("\n" + "Finished GD in " + str(epoch+1), "epochs with error " + str(cost_log[-1]) + "\n")
	print("Optimal theta:", theta)

	print("\n\n" + "y_hat, y")
	for i in indices:
		print(y_hat[i], Y[i])

	return end-start, cost_log

indices = [np.random.randint(0,m) for i in range(10)] #randomly sample 10 pairs for testing
#print(theta)

GD_time, gd_log = gd(theta, y_hat, X_T, dist, cost, 0.032) # max 3 dp. cuz overflow
print("\n\n" + "#"*10 + "\n\n")
NM_time, nm_log = nm(theta, y_hat, X_T, dist, cost, 150)

print("\n\n\n" + "GD time", GD_time, "| NM time", NM_time)

plt.plot(gd_log, label="GD")
plt.plot(nm_log, label="NM")
plt.xlabel("Epochs")
plt.ylabel("MSE Cost")
plt.title("Newton's Method vs. Gradient Descent")
plt.show() #can save too

if GD_time > NM_time:
	print("NM is faster")
	#if runtime is to fast, can't always tell cuz of sig fig
else:
	print("GD is faster")

"""
SANITY CHECK:
#foo1, foo2, foo3, foo4, foo5, foo6, foo7 = theta, y_hat, cost, Y, m, X, margin #insert before GD
assert(theta.all() == foo1.all())
assert(y_hat.all() == foo2.all())
assert(cost == foo3)
assert(Y.all() == foo4.all())
assert(m == foo5)
assert(X.all() == foo6.all())
assert(margin == foo7)
"""
