import numpy as np
from sklearn.model_selection import train_test_split 
from scipy.stats.mstats import mquantiles

# Note: skgarden has recent compatibility issues
#from skgarden import RandomForestQuantileRegressor

# Note: skgarden has recent compatibility issues
#import sys
#sys.path.insert(0, '../third_party')
#from cqr_comparison import NeuralNetworkQR

import torch

from arc.classification import ProbabilityAccumulator as ProbAccum

# Note: skgarden has recent compatibility issues
# class NeuralQuantileRegressor:
#     def __init__(self, p, alpha, random_state=2020, verbose=True):
#         # Parameters of the neural network
#         params = dict()
#         params['in_shape'] = p
#         params['epochs'] = 1000
#         params['lr'] = 0.0005
#         params['hidden_size'] = 64
#         params['batch_size'] = 64
#         params['dropout'] = 0.1
#         params['wd'] = 1e-6
#         params['test_ratio'] = 0.05
#         params['random_state'] = random_state
        
#         # Which quantiles to estimate
#         quantiles_net = [alpha, 1-alpha]
        
#         np.random.seed(random_state)
#         torch.manual_seed(random_state)
        
#         self.model = NeuralNetworkQR(params, quantiles_net, verbose=verbose)
        
#     def fit(self, X, y):
#         # Reshape the data
#         X = np.asarray(X)
#         y = np.asarray(y)
#         self.model.fit(X, y)
        
#     def predict(self, X):
#         y = self.model.predict(X)
#         return y

# Note: skgarden has recent compatibility issues
# class ForestQuantileRegressor:
#     def __init__(self, p, alpha, random_state=2020, verbose=True):
#         # Parameters of the random forest
#         self.alpha = 100*alpha
                
#         self.model = RandomForestQuantileRegressor(random_state=random_state,
#                                                    min_samples_split=3,
#                                                    n_estimators=100)
        
#     def fit(self, X, y):
#         # Reshape the data
#         X = np.asarray(X)
#         y = np.asarray(y)
#         self.model.fit(X, y)
        
#     def predict(self, X):
#         lower = self.model.predict(X, quantile=self.alpha)
#         y = np.concatenate((lower[:,np.newaxis], self.model.predict(X, quantile=100.0-self.alpha)[:,np.newaxis]),1)
#         return y
      
class SplitConformalHomogeneous:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False):
        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)
        n2 = X_calib.shape[0]
        self.black_box = black_box
        self.alpha = alpha
        self.allow_empty = allow_empty

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Estimate probabilities on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)

        # Break ties at random
        rng = np.random.default_rng(random_state)
        p_hat_calib += 1e-9 * rng.uniform(low=-1.0, high=1.0, size=p_hat_calib.shape)
        p_hat_calib = p_hat_calib / p_hat_calib.sum(axis=1)[:,None]        
        p_y_calib = np.array([ p_hat_calib[i, Y_calib[i]] for i in range(len(Y_calib)) ])        
        
        # Compute threshold
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        self.threshold_calibrated = mquantiles(p_y_calib, prob=1.0-level_adjusted)

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        p_hat = self.black_box.predict_proba(X)
        # Break ties at random
        rng = np.random.default_rng(random_state)
        p_hat += 1e-9 * rng.uniform(low=-1.0, high=1.0, size=p_hat.shape)
        p_hat = p_hat / p_hat.sum(axis=1)[:,None]        
        # Make prediction sets
        S_hat = [None]*n
        for i in range(n):
            S_hat[i] = np.where(p_hat[i,:] >= self.threshold_calibrated)[0]
            if (not self.allow_empty) and (len(S_hat[i])==0):
                S_hat[i] = [np.argmax(p_hat[i,:])]
        return S_hat


class BaseCQC:
    def __init__(self, X, y, black_box, alpha, qr_method, random_state=2020, allow_empty=True, verbose=False):
        self.allow_empty = allow_empty
        # Problem dimensions
        self.p = X.shape[1]
        
        # Alpha for conformal prediction intervals
        self.alpha = alpha

        # Black box for probability estimates
        self.black_box = black_box
        
        if qr_method == "NNet":
            # Quantiles for neural network
            self.quantile_black_box = NeuralQuantileRegressor(self.p,
                                                              self.alpha,
                                                              random_state=random_state, 
                                                              verbose=verbose)
        # elif qr_method == "RF":
        #     self.quantile_black_box = ForestQuantileRegressor(self.p,
        #                                                       self.alpha,
        #                                                       random_state=random_state,
        #                                                       verbose=verbose)
        else:
            raise
            
        # Split data into training and calibration sets
        X_train, X_calibration, y_train, y_calibration = train_test_split(X, y, test_size=0.333, 
                                                                          random_state=random_state)
        
        # Further split training data
        X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.5,
                                                                      random_state=random_state)
        
        # Estimate probabilities with the black box on the first training set
        n1 = X_train_1.shape[0]
        if verbose:
            print("Training the black-box classifier with {} samples...". format(n1))
            sys.stdout.flush()
        self.black_box.fit(X_train_1, y_train_1)
        p_hat_2 = self.black_box.predict_proba(X_train_2)
        sys.stdout.flush()
        # Compute scores on second training set
        n2 = X_train_2.shape[0]
        scores_2 = np.array([p_hat_2[i,int(y_train_2[i])] for i in range(n2)])
        
        # Train the quantile estimator on the above scores
        if verbose:
            print("Training the quantile regression black box with {} samples...". format(n2))
            sys.stdout.flush()
        self.quantile_black_box.fit(X_train_2, scores_2)
        # Estimate the quantiles on the calibration data (keep only the upper quantiles)
        q_hat = self.quantile_black_box.predict(X_calibration)[:,1]
        sys.stdout.flush()
        
        # Compute conformity scores
        n3 = X_calibration.shape[0]
        p_hat_3 = self.black_box.predict_proba(X_calibration)
        scores_3 = np.array([p_hat_3[i,int(y_calibration[i])] for i in range(n3)])
        conf_scores = q_hat - scores_3
        
        # Compute quantile of conformity scores
        level_adjusted = (1.0-self.alpha)*(1.0+1.0/float(n3))
        self.score_correction = mquantiles(conf_scores, prob=level_adjusted)
        
    def predict(self, X):
        n = X.shape[0]
        p_hat = self.black_box.predict_proba(X)
        q_hat = self.quantile_black_box.predict(X)[:,1]
        S = [None]*n
        for i in range(n):
            S[i] = np.where(p_hat[i,:] >= q_hat[i] - self.score_correction)[0]
            if (not self.allow_empty) and (len(S[i])==0):
                S[i] = [np.argmax(p_hat[i,:])]
        return S
    
class CQC:
    def __init__(self, X, y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False):
        self.base_cqc = BaseCQC(X, y, black_box, alpha, "NNet", random_state=random_state, allow_empty=allow_empty, verbose=verbose)
        
    def predict(self, X):
        return self.base_cqc.predict(X)

# class CQCRF:
#     def __init__(self, X, y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False):
#         self.base_cqc = BaseCQC(X, y, black_box, alpha, "RF", random_state=random_state, allow_empty=allow_empty, verbose=verbose)
        
#     def predict(self, X):
#         return self.base_cqc.predict(X)
    
class Oracle:
    def __init__(self, data_model, alpha, random_state=2020, allow_empty=True, verbose=True):
        self.data_model = data_model
        self.alpha = alpha
        self.allow_empty = allow_empty
        
    def predict(self, X, randomize=True, random_state=2020):
        if randomize:
            rng = np.random.default_rng(random_state)
            epsilon = rng.uniform(low=0.0, high=1.0, size=X.shape[0])
        else:
            epsilon = None
        prob_y = self.data_model.compute_prob(X)
        grey_box = ProbAccum(prob_y)
        S = grey_box.predict_sets(self.alpha, epsilon=epsilon, allow_empty=self.allow_empty)
        return S
