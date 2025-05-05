# src/model.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Union
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np

import src.utils as utils

BIG_NUMBER_TO_AVOID_SALE = 1e6
DEFAULT_QUANTILES = np.linspace(0.05, 0.95, 19)

class PricingModel:
    def __init__(self, quantiles: Union[List[float], np.ndarray] = DEFAULT_QUANTILES):
        self.base_models = {}
        self.evals_results = {}
        self.opti_pricing_problem = None
        self.opti_pricing_results = None
        self.opti_pricing_params = None
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
            'pop_size': 200,
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
    
    def train_base_models(self, 
                          X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          X_val: pd.DataFrame,
                          y_val: pd.Series) -> None:
        """Train quantile regression models for all specified quantiles."""
        
        for quantile in self.quantiles:
            self.base_models[quantile] = self._train_quantile_model(
                X_train, y_train, X_val, y_val, quantile
            )
    
    def optimize_pricing_strategy(self, X_train, y_train):
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

    def select_optimized_pricing_strategy(self, market_share_threshold: float = 0.3):
        """Select the optimized pricing strategy based on the market share threshold."""
        assert self.opti_pricing_results is not None, "No optimized pricing results found. Please run optimize_pricing_strategy first."
        assert self.opti_pricing_results.X is not None, "No optimized pricing results found. Please run optimize_pricing_strategy first."
        assert self.opti_pricing_results.F is not None, "No optimized pricing results found. Please run optimize_pricing_strategy first."
        solutions = self.opti_pricing_results.X
        market_shares = -self.opti_pricing_results.F[:, 1]
        avg_losses = self.opti_pricing_results.F[:, 0]
        
        # Filter solutions based on market share threshold
        valid_indices = np.where(market_shares >= market_share_threshold)[0]
        
        if len(valid_indices) == 0:
            raise ValueError("No valid strategies found. Please try a different market share threshold.")
        
        # Sort valid solutions by market share
        valid_indices = valid_indices[np.argsort(market_shares[valid_indices])]
        
        # Select the solution with market share just above the threshold
        selected_index = valid_indices[0]  # The first one after sorting will be just above threshold
        
        # Get the corresponding strategy
        valid_strategy = solutions[selected_index]
        market_share = market_shares[selected_index]
        avg_loss = avg_losses[selected_index]
        self.opti_pricing_params = valid_strategy

        return valid_strategy, market_share, avg_loss
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the prices for the given input data."""
        assert self.opti_pricing_params is not None, "No optimized pricing parameters found. Please run select_optimized_pricing_strategy first."
        assert self.base_models is not None, "No base models found. Please run train_base_models first."
        assert self.quantiles is not None, "No quantiles found. Please set the quantiles."
        
        predictions = {q: self.base_models[q].predict(xgb.DMatrix(X))
                             for q in self.base_models.keys()}
        
        adjusted_prices = PricingOptimizationProblem.compute_adjusted_prices(predictions, self.opti_pricing_params)
        
        
        return adjusted_prices
    



class PricingOptimizationProblem(Problem):
    """
    Pricing optimization problem for the pricing model. 
    The problem is to find the optimal pricing strategy that maximizes the market share (up to 30%) and minimizes the average loss.
    """
    VAR_NAMES = ["q",
                 "q_high",
                 "q_low", 
                 "s_lim"]
    
    @staticmethod
    def interp_pred_from_precomputed(predictions, target_quantile):
        """
        Interpolate between precomputed quantile predictions to get predictions for any quantile.
        
        Args:
            target_quantile: The desired quantile (between 0 and 1) to interpolate
            
        Returns:
            Interpolated predictions for the specified quantile
        """
        # Get the available quantiles
        available_quantiles = sorted(predictions.keys())
        
        # Handle edge cases
        if target_quantile <= min(available_quantiles):
            return predictions[min(available_quantiles)]
        
        if target_quantile >= max(available_quantiles):
            return predictions[max(available_quantiles)]
        
        # Find the two quantiles that sandwich the requested quantile
        lower_quantile = max([q for q in available_quantiles if q <= target_quantile])
        upper_quantile = min([q for q in available_quantiles if q >= target_quantile])
        
        # If we hit an exact quantile, just return that prediction
        if lower_quantile == upper_quantile:
            return predictions[lower_quantile]
        
        # Get predictions for the two quantiles
        lower_preds = predictions[lower_quantile]
        upper_preds = predictions[upper_quantile]
        
        # Linearly interpolate between the two predictions
        weight = (target_quantile - lower_quantile) / (upper_quantile - lower_quantile)
        interpolated_preds = lower_preds * (1 - weight) + upper_preds * weight
        
        return interpolated_preds
    
    @staticmethod
    def compute_adjusted_prices(predictions, params):
        """Compute the adjusted prices for the given input data."""
        q, q_high, q_low, s_lim = params
        adjusted_prices = PricingOptimizationProblem.interp_pred_from_precomputed(predictions, q)
        upper_prices = PricingOptimizationProblem.interp_pred_from_precomputed(predictions, q_high)
        lower_prices = PricingOptimizationProblem.interp_pred_from_precomputed(predictions, q_low)
        spread = upper_prices - lower_prices
        spread_norm = (spread - spread.mean())/spread.std()
        adjusted_prices = adjusted_prices +  BIG_NUMBER_TO_AVOID_SALE*(spread_norm > s_lim)
            
        return adjusted_prices

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
            
            adjusted_prices = PricingOptimizationProblem.compute_adjusted_prices(self.predictions, params)
            
            # Calculate metrics
            avg_loss, market_share = utils.metrics(adjusted_prices, self.y_train)
            
            if market_share == 0:
                f_values[i, 0] = 1e6  # Large penalty for no sales
                f_values[i, 1] = 1.0  # Worst market share (0%)
            else:
                f_values[i, 0] = avg_loss
                f_values[i, 1] = -market_share  # Negative because we want to maximize market share
        
        out["F"] = f_values


