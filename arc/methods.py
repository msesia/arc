import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats.mstats import mquantiles
import sys
from tqdm import tqdm

from arc.classification import ProbabilityAccumulator as ProbAccum

class CVPlus:
    def __init__(self, X, Y, black_box, alpha, n_folds=10, random_state=2020, allow_empty=True, verbose=False):
        X = np.array(X)
        Y = np.array(Y)
        self.black_box = black_box
        self.n = X.shape[0]
        self.classes = np.unique(Y)
        self.n_classes = len(self.classes)
        self.n_folds = n_folds
        self.cv = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        self.alpha = alpha
        self.allow_empty = allow_empty
        self.verbose = verbose
        
        # Fit prediction rules on leave-one out datasets
        self.mu_LOO = [ black_box.fit(X[train_index], Y[train_index]) for train_index, _ in self.cv.split(X) ]

        # Accumulate probabilities for the original data with the grey boxes
        test_indices = [test_index for _, test_index in self.cv.split(X)]
        self.test_indices = test_indices
        self.folds = [[]]*self.n
        for k in range(self.n_folds):
            for i in test_indices[k]:
                self.folds[i] = k
        self.grey_boxes = [[]]*self.n_folds
        if self.verbose:
            print("Training black boxes on {} samples with {}-fold cross-validation:".
                  format(self.n, self.n_folds), file=sys.stderr)
            sys.stderr.flush()
            for k in tqdm(range(self.n_folds), ascii=True, disable=True):
                self.grey_boxes[k] = ProbAccum(self.mu_LOO[k].predict_proba(X[test_indices[k]]))
        else:
            for k in range(self.n_folds):
                self.grey_boxes[k] = ProbAccum(self.mu_LOO[k].predict_proba(X[test_indices[k]]))
               
        # Compute scores using real labels
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=self.n)
        self.alpha_max = np.zeros((self.n, 1))
        if self.verbose:
            print("Computing scores for {} samples:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for k in tqdm(range(self.n_folds), ascii=True, disable=True):
                idx = test_indices[k]
                self.alpha_max[idx,0] = self.grey_boxes[k].calibrate_scores(Y[idx], epsilon=epsilon[idx])
        else:
            for k in range(self.n_folds):
                idx = test_indices[k]
                self.alpha_max[idx,0] = self.grey_boxes[k].calibrate_scores(Y[idx], epsilon=epsilon[idx])
            
    def predict(self, X, random_state=2020):
        n = X.shape[0]
        S = [[]]*n
        n_classes = len(self.classes)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
        prop_smaller = np.zeros((n,n_classes))

        if self.verbose:
            print("Computing predictive sets for {} samples:". format(n), file=sys.stderr)
            sys.stderr.flush()
            for fold in tqdm(range(self.n_folds), ascii=True, disable=True):
                gb = ProbAccum(self.mu_LOO[fold].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    for i in self.test_indices[fold]:
                        prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
        else:
            for fold in range(self.n_folds):
                gb = ProbAccum(self.mu_LOO[fold].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    for i in self.test_indices[fold]:
                        prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])

        for k in range(n_classes):
            prop_smaller[:,k] /= float(self.n)
                
        level_adjusted = (1.0-self.alpha)*(1.0+1.0/float(self.n))
        S = [None]*n
        for i in range(n):
            S[i] = np.where(prop_smaller[i,:] < level_adjusted)[0]
            if (not self.allow_empty) and (len(S[i])==0): # Note: avoid returning empty sets
                if len(S[i])==0:
                    S[i] = [np.argmin(prop_smaller[i,:])]            
        return S

class JackknifePlus:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False):
        self.black_box = black_box
        self.n = X.shape[0]
        self.classes = np.unique(Y)
        self.alpha = alpha
        self.allow_empty = allow_empty
        self.verbose = verbose

        # Fit prediction rules on leave-one out datasets
        self.mu_LOO = [[]] * self.n
        if self.verbose:
            print("Training black boxes on {} samples with the Jacknife+:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                self.mu_LOO[i] = black_box.fit(np.delete(X,i,0),np.delete(Y,i))
        else:
            for i in range(self.n):
                self.mu_LOO[i] = black_box.fit(np.delete(X,i,0),np.delete(Y,i))

        # Accumulate probabilities for the original data with the grey boxes
        self.grey_boxes = [ ProbAccum(self.mu_LOO[i].predict_proba(X[i])) for i in range(self.n) ]

        # Compute scores using real labels
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=self.n)
        
        self.alpha_max = np.zeros((self.n, 1))    
        if self.verbose:
            print("Computing scores for {} samples:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                self.alpha_max[i,0] = self.grey_boxes[i].calibrate_scores(Y[i], epsilon=epsilon[i])
        else:
            for i in range(self.n):
                self.alpha_max[i,0] = self.grey_boxes[i].calibrate_scores(Y[i], epsilon=epsilon[i])
                
    def predict(self, X, random_state=2020):
        n = X.shape[0]
        S = [[]]*n
        n_classes = len(self.classes)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
        prop_smaller = np.zeros((n,n_classes))
        
        if self.verbose:
            print("Computing predictive sets for {} samples:". format(n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                gb = ProbAccum(self.mu_LOO[i].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
        else:
            for i in range(self.n):
                gb = ProbAccum(self.mu_LOO[i].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
                
        for k in range(n_classes):
            prop_smaller[:,k] /= float(self.n)
        level_adjusted = (1.0-self.alpha)*(1.0+1.0/float(self.n))
        S = [None]*n
        for i in range(n):
            S[i] = np.where(prop_smaller[i,:] < level_adjusted)[0]
            if (not self.allow_empty) and (len(S[i])==0): # Note: avoid returning empty sets
                if len(S[i])==0:
                    S[i] = [np.argmin(prop_smaller[i,:])]            
        return S

class SplitConformal:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)
        n2 = X_calib.shape[0]

        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)
        grey_box = ProbAccum(p_hat_calib)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)
        return S_hat
