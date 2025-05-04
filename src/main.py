# src/main.py
import pandas as pd
from src.data_processing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import PricingModel

def main():
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    pricing_model = PricingModel()
    
    # Load and preprocess data
    train_df, test_df = preprocessor.preprocess(
        train_path='data/train.xlsx',
        test_path='data/test.xlsx'
    )
    
    # Engineer features
    train_processed, test_processed = feature_engineer.transform(train_df, test_df)
    
    # Prepare training data
    X_train = train_processed.drop(['quote_id', 'competitor_lowest_price'], axis=1)
    y_train = train_processed['competitor_lowest_price']
    
    # Train model
    pricing_model.train(X_train, y_train)
    
    # Generate predictions for test set
    X_test = test_processed.drop(['quote_id'], axis=1)
    predictions = pricing_model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'quote_id': test_processed['quote_id'],
        'proposed_price': predictions
    })
    
    # Save predictions
    submission.to_csv('predictions.csv', index=False)
    
    # Calculate and print metrics for training set
    train_predictions = pricing_model.predict(X_train)
    has_sold = y_train > train_predictions
    sold_policies = train_predictions[has_sold]
    actual_prices = y_train[has_sold]
    
    avg_loss = (actual_prices - sold_policies).mean()
    market_share = has_sold.mean()
    
    print(f"Training Metrics:")
    print(f"Average Loss: {avg_loss:.2f}")
    print(f"Market Share: {market_share:.2%}")

if __name__ == "__main__":
    main()