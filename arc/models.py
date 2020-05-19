import numpy as np
from scipy.stats import norm

def sigmoid(x):
    return(1/(1 + np.exp(-x)))

class Model_Ex1:
    def __init__(self, K, p, magnitude=1):
        self.K = K
        self.p = p
        self.magnitude = magnitude
        # Generate model parameters
        self.beta_Z = self.magnitude*np.random.randn(self.p,self.K)

    def sample_X(self, n):
        X = np.random.normal(0, 1, (n,self.p))
        factor = 0.2
        X[0:int(n*factor),0] = 1
        X[int(n*factor):,0] = -8
        return X.astype(np.float32)
    
    def compute_prob(self, X):
        f = np.matmul(X,self.beta_Z)
        prob = np.exp(f)
        prob_y = prob / np.expand_dims(np.sum(prob,1),1)
        return prob_y

    def sample_Y(self, X):
        prob_y = self.compute_prob(X)
        g = np.array([np.random.multinomial(1,prob_y[i]) for i in range(X.shape[0])], dtype = float)
        classes_id = np.arange(self.K)
        y = np.array([np.dot(g[i],classes_id) for i in range(X.shape[0])], dtype = int)
        return y.astype(np.int)
    
class Model_Ex2:
    def __init__(self, K, p, magnitude=1):
        self.K = K
        self.p = p
        self.magnitude = magnitude
        
    def sample_X(self, n):
        X = np.random.normal(0, 1, (n,self.p))
        X[:,0] = np.random.choice([-1,1], size=n, replace=True, p=[1/self.K, (self.K-1)/self.K])
        X[:,1] = np.random.choice([-1,1], size=n, replace=True, p=[1/4, 3/4])
        X[:,2] = np.random.choice([-1,1], size=n, replace=True, p=[1/2, 1/2])
        X[:,3] = np.random.choice(self.K, n, replace=True)
        return X.astype(np.float32)
        
    def compute_prob(self, X):
        prob_y = np.zeros((X.shape[0], self.K))
        right_0 = X[:,0] > 0
        right_1 = X[:,1] > 0
        right_2 = X[:,2] > 0
        leaf_0 = np.where(1-right_0)[0]
        leaf_1 = np.where((right_0) * (1-right_1) * (1-right_2))[0]
        leaf_2 = np.where((right_0) * (1-right_1) * (right_2))[0]
        leaf_3 = np.where((right_0) * (right_1))[0]
        # Zeroth leaf: uniform distribution over all labels
        prob_y[leaf_0] = 1.0/self.K
        # First leaf: uniform distribution over the first half of the labels
        K_half = int(np.round(self.K/2))
        prob_y[leaf_1, 0:K_half] = 2.0/self.K
        prob_y[leaf_1, K_half:self.K] = 0
        # Second leaf: uniform distribution over the second half of the labels
        prob_y[leaf_2, 0:K_half] = 0
        prob_y[leaf_2, K_half:self.K] = 2.0/self.K
        # Third leaf: 90% probability to label determined by 4th variable
        X3 = np.round(X[leaf_3,3]).astype(int)
        for k in range(self.K):
            prob_y[leaf_3[X3==k],:] = (1-0.9)/(self.K-1.0)
            prob_y[leaf_3[X3==k],k] = 0.9
        # Standardize probabilities for each sample        
        prob_y = prob_y / prob_y.sum(axis=1)[:,None]
        return prob_y

    def sample_Y(self, X):
        prob_y = self.compute_prob(X)
        g = np.array([np.random.multinomial(1,prob_y[i]) for i in range(X.shape[0])], dtype = float)
        classes_id = np.arange(self.K)
        y = np.array([np.dot(g[i],classes_id) for i in range(X.shape[0])], dtype = int)
        return y.astype(np.int)
