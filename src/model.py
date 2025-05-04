# src/model.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np

import src.utils as utils

class PricingModel:
    def __init__(self, quantiles: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]):
        self.base_models = {}
        self.evals_results = {}
        self.opti_pricing_problem = None
        self.opti_pricing_results = None
        self.feature_names = None
        self.price_adjustment_factors = None
        self.quantiles = quantiles
        self.xgb_params = {
            'objective': 'reg:quantileerror',  # Use quantile regression
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'num_boost_round': 300,
        }
        self.optimizer_params = {
            'n_gen': 100,
            'pop_size': 100,
            'seed': 42,
            'verbose': True
        }

    def _train_quantile_model(self, 
                             X_train: pd.DataFrame, 
                             y_train: pd.Series,
                             X_val: pd.DataFrame,
                             y_val: pd.Series,
                             quantile: float) -> xgb.Booster:
        """Train a quantile regression model for a specific quantile."""
        
        # Define model parameters
        params = self.xgb_params
        params['quantile_alpha'] = quantile
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals_result = {}
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('num_boost_round', 300),
            evals=[(dtrain, 'train'), (dval, 'val')],
            evals_result=evals_result,
            early_stopping_rounds=50,
            verbose_eval=100
        )

        self.evals_results[quantile] = evals_result
        
        return model
    
    def _train_base_models(self, 
                          X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          X_val: pd.DataFrame,
                          y_val: pd.Series) -> None:
        """Train quantile regression models for all specified quantiles."""
        
        for quantile in self.quantiles:
            self.base_models[quantile] = self._train_quantile_model(
                X_train, y_train, X_val, y_val, quantile
            )
    
    def _optimize_pricing_strategy(self, X_train, y_train, X_val, y_val):
        """Optimize the pricing strategy."""
        self.opti_pricing_problem = PricingOptimizationProblem(self, X_train, y_train)
        algorithm = NSGA2(
            pop_size=self.optimizer_params.get('pop_size', 100),
            eliminate_duplicates=True
        )
        res = minimize(self.opti_pricing_problem,
                       algorithm,
                       ('n_gen', self.optimizer_params.get('n_gen', 100)),
                       seed=self.optimizer_params.get('seed', 42),
                       verbose=self.optimizer_params.get('verbose', True)
                       )
        self.opti_pricing_results = res

    



class PricingOptimizationProblem(Problem):
    """
    Pricing optimization problem for the pricing model. 
    The problem is to find the optimal pricing strategy that maximizes the market share (up to 30%) and minimizes the average loss.
    We vary 4 parameters:
    - pred_quantile: the quantile to predict price offered
    - upper_quantile: the upper quantile to use in spread calculation
    - lower_quantile: the lower quantile to use in spread calculation
    - spread_switch: the threshold for the spread, above which price prediction is discarded
    """
    VAR_NAMES = ["pred_quantile",
                 "upper_quantile",
                 "lower_quantile", 
                 "spread_switch"]
    def __init__(self, pricing_model, X_train, y_train):
        super().__init__(n_var=4, n_obj=2, n_constr=0,
                          xl=np.array([0.05, 0.05, 0.05, -3]),
                          xu=np.array([0.95, 0.95, 0.95, 3]))
        self.pricing_model = pricing_model
        self.X_train = X_train
        self.y_train = y_train

        self.predictions = {q: self.pricing_model.base_models[q].predict(xgb.DMatrix(self.X_train))
                             for q in self.pricing_model.base_models.keys()}
          
    def _evaluate(self, x, out, *args, **kwargs):
        # Extract parameters
        f_values = np.zeros((len(x), 2))  # Two objectives: avg_loss and -market_share
        
        for i, params in enumerate(x):
            (
            pred_quantile,
            upper_quantile,
            lower_quantile,
            upper_spread_switch,
            ) = params
            
            adjusted_prices = self.interp_pred_from_precomputed(pred_quantile)
            upper_prices = self.interp_pred_from_precomputed(upper_quantile)
            lower_prices = self.interp_pred_from_precomputed(lower_quantile)
            spread = upper_prices - lower_prices
            spread_norm = (spread - spread.mean())/spread.std()
            adjusted_prices = adjusted_prices +  1e6*(spread_norm > upper_spread_switch)
            #adjusted_prices = adjusted_prices +  1e6*(spread_norm < lower_spread_switch)
            
            # Calculate metrics
            avg_loss, market_share = utils.metrics(adjusted_prices, self.y_train)
            
            if market_share == 0:
                f_values[i, 0] = 1e6  # Large penalty for no sales
                f_values[i, 1] = 1.0  # Worst market share (0%)
            else:
                f_values[i, 0] = avg_loss
                f_values[i, 1] = -market_share  # Negative because we want to maximize market share
        
        out["F"] = f_values

    def interp_pred_from_precomputed(self, target_quantile):
        """
        Interpolate between precomputed quantile predictions to get predictions for any quantile.
        
        Args:
            target_quantile: The desired quantile (between 0 and 1) to interpolate
            
        Returns:
            Interpolated predictions for the specified quantile
        """
        # Get the available quantiles
        available_quantiles = sorted(self.predictions.keys())
        
        # Handle edge cases
        if target_quantile <= min(available_quantiles):
            return self.predictions[min(available_quantiles)]
        
        if target_quantile >= max(available_quantiles):
            return self.predictions[max(available_quantiles)]
        
        # Find the two quantiles that sandwich the requested quantile
        lower_quantile = max([q for q in available_quantiles if q <= target_quantile])
        upper_quantile = min([q for q in available_quantiles if q >= target_quantile])
        
        # If we hit an exact quantile, just return that prediction
        if lower_quantile == upper_quantile:
            return self.predictions[lower_quantile]
        
        # Get predictions for the two quantiles
        lower_preds = self.predictions[lower_quantile]
        upper_preds = self.predictions[upper_quantile]
        
        # Linearly interpolate between the two predictions
        weight = (target_quantile - lower_quantile) / (upper_quantile - lower_quantile)
        interpolated_preds = lower_preds * (1 - weight) + upper_preds * weight
        
        return interpolated_preds
