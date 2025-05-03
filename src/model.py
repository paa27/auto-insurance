# src/model.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import optuna

class PricingModel:
    def __init__(self, quantiles: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]):
        self.base_models = {}
        self.evals_result = {}
        self.feature_names = None
        self.price_adjustment_factors = None
        self.quantiles = quantiles
        
    def _train_quantile_model(self, 
                             X_train: pd.DataFrame, 
                             y_train: pd.Series,
                             X_val: pd.DataFrame,
                             y_val: pd.Series,
                             quantile: float) -> xgb.Booster:
        """Train a quantile regression model for a specific quantile."""
        
        # Define model parameters
        params = {
            'objective': 'reg:quantileerror',  # Use quantile regression
            'quantile_alpha': quantile,        # Specify which quantile to estimate
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist'
        }
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            evals_result=evals_result,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
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
    
    def _optimize_pricing_strategy(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 base_predictions: np.ndarray) -> None:
        """Optimize pricing strategy using Optuna."""
        def objective(trial):
            # Define pricing adjustment parameters
            base_discount = trial.suggest_float('base_discount', 0.90, 0.99)
            risk_factor = trial.suggest_float('risk_factor', 0.95, 1.05)
            claims_penalty = trial.suggest_float('claims_penalty', 1.0, 1.1)
            
            # Apply pricing strategy
            adjusted_prices = base_predictions * base_discount
            
            # Apply risk adjustments
            if 'claims_score' in X.columns:
                adjusted_prices *= (1 + (X['claims_score'] * (claims_penalty - 1)))
            
            if 'risk_score' in X.columns:
                adjusted_prices *= risk_factor ** X['risk_score']
            
            # Calculate metrics
            has_sold = y > adjusted_prices
            market_share = has_sold.mean()
            
            if market_share < 0.3:  # Market share constraint
                return float('inf')
            
            # Calculate average loss on sold policies
            sold_policies_mask = y > adjusted_prices
            if sold_policies_mask.sum() == 0:
                return float('inf')
            
            avg_loss = (y[sold_policies_mask] - 
                       adjusted_prices[sold_policies_mask]).mean()
            
            return avg_loss
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        self.price_adjustment_factors = study.best_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Train the complete pricing model."""
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Train base model
        self._train_base_model(X_train, y_train, X_val, y_val)
        
        # Get base predictions
        base_predictions = self.base_model.predict(xgb.DMatrix(X_train))
        
        # Optimize pricing strategy
        self._optimize_pricing_strategy(X_train, y_train, base_predictions)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate price predictions."""
        # Get base predictions
        base_predictions = self.base_model.predict(xgb.DMatrix(X))
        
        # Apply pricing strategy
        adjusted_prices = base_predictions * self.price_adjustment_factors['base_discount']
        
        # Apply risk adjustments
        if 'claims_score' in X.columns:
            adjusted_prices *= (1 + (X['claims_score'] * 
                                   (self.price_adjustment_factors['claims_penalty'] - 1)))
        
        if 'risk_score' in X.columns:
            adjusted_prices *= (self.price_adjustment_factors['risk_factor'] ** 
                              X['risk_score'])
        
        return adjusted_prices