import os, sys
import pdb
import numpy as np
from skgarden import RandomForestQuantileRegressor as RF
from sklearn.model_selection import train_test_split

class RandomForestQR:
    def __init__(self, params, quantiles, verbose=False):
        self.regressor = RF(n_estimators = params['n_estimators'],
                            max_features = params['max_features'],
                            min_samples_leaf = params['min_samples_leaf'],
                            random_state = params['random_state'],
                            n_jobs = params['n_jobs'])
        self.quantiles = quantiles
        self.cv_quantiles = quantiles
        self.verbose = verbose
        self.cv = params["cv"]

    def fit(self, X, y, cv=True):
        if self.cv and cv:
            self.tune(X, y)
        self.regressor.fit(X, y)

    def predict(self, X, quantiles=None):
        if quantiles is None:
            quantiles = self.cv_quantiles

        predictions = np.zeros((X.shape[0],len(quantiles)))
        for j in range(len(quantiles)):
            q = 100.0 * quantiles[j]
            predictions[:,j] = self.regressor.predict(X, q)

        predictions.sort(axis=1)
        return predictions

    def tune(self, X, y, test_ratio=0.2, random_state=1):
        "Tune using cross-validation"
        coverage_factor = 0.85
        target_coverage = round(self.quantiles[-1] - self.quantiles[0],3) * coverage_factor
        range_vals = 0.3
        num_vals = 10

        print("  [CV] target coverage = %.3f" %(target_coverage))
        sys.stdout.flush()

        quantiles = np.array(self.quantiles)
        grid_q_low = np.linspace(quantiles[0],quantiles[0]+range_vals,num_vals).reshape(-1,1)
        grid_q_median = np.repeat(0.5,num_vals).reshape(-1,1)
        grid_q_high = np.linspace(quantiles[-1],quantiles[-1]-range_vals,num_vals).reshape(-1,1)
        grid_q = np.concatenate((grid_q_low,grid_q_median,grid_q_high),1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
        print("  [CV] Fitting random forest... ", end="")
        sys.stdout.flush()
        self.fit(X_train, y_train, cv=False)
        print("done.")
        sys.stdout.flush()

        best_avg_length = 1e10
        best_q = grid_q[0]
        for q in grid_q:
            print("  [CV] q = [%.3f,%.3f,%.3f], " %(q[0],q[1],q[-1]), end="")
            sys.stdout.flush()
            y_predictions = self.predict(X_test, quantiles=q)
            lower = y_predictions[:,0]
            upper = y_predictions[:,-1]
            coverage = np.mean((y_test >= lower) & (y_test <= upper))
            avg_length = np.mean(upper-lower)
            print("coverage = %.3f, length = %.3f" %(coverage, avg_length))
            sys.stdout.flush()
            if (coverage >= target_coverage) and (avg_length < best_avg_length):
                best_avg_length = avg_length
                best_q = q
            else:
                break

        print("  [CV] Best q = [%.3f,%.3f,%.3f]" %(best_q[0], best_q[1], best_q[-1]))
        sys.stdout.flush()

        self.cv_quantiles = best_q
        return best_q
