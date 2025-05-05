# src/feature_processor.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import OneHotEncoder
#from category_encoders import TargetEncoder

class FeatureProcessor:
    def __init__(self):
        # Statistics and encoders storage
        self.numeric_statistics: Dict[str, float] = {}
        self.categorical_statistics: Dict[str, str] = {}
        self.onehot_encoders: Dict[str, OneHotEncoder] = {}
        #self.target_encoders: Dict[str, TargetEncoder] = {}
        
        # Fixed reference date
        self.reference_date = pd.to_datetime('2025-05-04')
        
        # Column definitions
        self.datetime_columns = [
            'driver_birth_date', 
            'driver_driving_license_ym',
            'occasional_driver_birth_date', 
            'vehicle_buy_ym',
            'vehicle_registration_ym', 
            'timestamp'
        ]
        
        self.onehot_columns = [
            'driver_other_vehicles',
        ]

        self.todrop_columns = [
            'number_of_competitors',
        ]
        
        self.claims_columns = [
            'driver_claims_last_1_year',
            'driver_claims_from_year_1_to_2',
            'driver_claims_from_year_2_to_3',
            'driver_claims_from_year_3_to_4',
            'driver_claims_from_year_4_to_5'
        ]
        
        # Mapping for categorical to numeric conversion
        self.insured_years_mapping = {
            'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4,
            'FIVE': 5, 'SIX': 6, 'SEVEN': 7, 'EIGHT': 8,
            'NINE': 9, 'TEN': 10, 'MORE_THAN_TEN': 11
        }
    
    def _compute_statistics(self, df: pd.DataFrame) -> None:
        """Compute and store statistics from training data."""
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if 'claims' in col.lower():
                self.numeric_statistics[f"{col}_fill"] = 0
            else:
                self.numeric_statistics[f"{col}_fill"] = df[col].median()
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.categorical_statistics[f"{col}_fill"] = df[col].mode()[0]
    
    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all datetime-related features."""
        df = df.copy()
        
        # Convert to datetime
        for col in self.datetime_columns:
            if col in df.columns:
                if 'ym' in col:  # Year-month format
                    df[col] = pd.to_datetime(df[col], format='%Y-%m')
                else:
                    df[col] = pd.to_datetime(df[col])
        
        # Calculate ages and time differences
        if 'driver_birth_date' in df.columns:
            df['driver_age'] = (self.reference_date - df['driver_birth_date']).dt.days / 365.25
            
        if 'driver_driving_license_ym' in df.columns:
            df['driving_experience'] = (self.reference_date - df['driver_driving_license_ym']).dt.days / 365.25
            
        if 'vehicle_registration_ym' in df.columns:
            df['vehicle_age'] = (self.reference_date - df['vehicle_registration_ym']).dt.days / 365.25
        
        # Extract temporal features from timestamp
        # if 'timestamp' in df.columns:
        #     df['quote_hour'] = df['timestamp'].dt.hour.astype(int)
        #     df['quote_day'] = df['timestamp'].dt.day.astype(int)
        #     df['quote_month'] = df['timestamp'].dt.month.astype(int)
        #     df['quote_day_of_week'] = df['timestamp'].dt.dayofweek.astype(int)
        #     df['quote_is_weekend'] = df['quote_day_of_week'].isin([5, 6]).astype(int)
        
        # Drop original datetime columns
        df = df.drop(columns=[col for col in self.datetime_columns if col in df.columns])
        
        return df
    
    def _process_insured_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process insured years feature."""
        df = df.copy()
        
        # Convert insured years
        if 'driver_insured_years' in df.columns:
            df['driver_insured_years'] = df['driver_insured_years'].map(self.insured_years_mapping)
        
        return df
    
    def _process_vehicle_acquisition_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process vehicle acquisition state feature."""
        df = df.copy()
        
        # Convert vehicle acquisition state
        if 'vehicle_acquisition_state' in df.columns:
            # Create vehicle_acquisition_time feature
            df['vehicle_acquisition_time_OWNED'] = df['vehicle_acquisition_state'].str.lower().str.contains('owned').astype(int)
            df['vehicle_acquisition_time_BUYING'] = df['vehicle_acquisition_state'].str.lower().str.contains('buying').astype(int)
            df['vehicle_acquisition_time_RECENT'] = df['vehicle_acquisition_state'].str.lower().str.contains('recent').astype(int)
            
            # Create vehicle_acquisition_first_hand feature
            # If owned, it's second_hand by default
            df['vehicle_acquisition_FIRST_HAND'] = 0  # Default to second_hand (0)
            
            # check if first_hand
            first_hand_mask = df['vehicle_acquisition_state'].str.lower().str.contains('first_hand')
            df.loc[first_hand_mask, 'vehicle_acquisition_FIRST_HAND'] = 1
            
            # Create vehicle_acquisition_private feature
            # If owned, it's always private
            df['vehicle_acquisition_PRIVATE'] = 1  # Default to private (1)        
            dealer_mask = df['vehicle_acquisition_state'].str.lower().str.contains('dealer')
            df.loc[dealer_mask, 'vehicle_acquisition_PRIVATE'] = 0
            
            # Drop original vehicle_acquisition_state column
            df = df.drop(columns=['vehicle_acquisition_state'])

        return df
        
    
    def _process_vehicle_use(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process vehicle use feature."""
        df = df.copy()
        
        # Convert vehicle use
        if 'vehicle_use' in df.columns:
            # Create professional vs personal feature
            df['vehicle_use_PROFESSIONAL'] = df['vehicle_use'].str.lower().str.contains('professional').astype(int)
            
            # Create habitual vs occasional feature
            df['vehicle_use_HABITUAL'] = df['vehicle_use'].str.lower().str.contains('habitual').astype(int)
            
            # Drop original vehicle_use column
            df = df.drop(columns=['vehicle_use'])
        
        return df
    
    def _process_total_claims(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process total claims feature."""
        df = df.copy()
        
        # Convert number of claims
        if all(claim in df.columns for claim in self.claims_columns):
            df['total_claims'] = df[self.claims_columns].sum(axis=1)
            #df = df.drop(columns=self.claims_columns)
        
        return df
    
    def _process_categorical(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        """Process categorical features including encoding."""
        df = df.copy()
        
        # One-hot encoding
        for col in self.onehot_columns:
            if col in df.columns:
                if training:
                    self.onehot_encoders[col] = OneHotEncoder(
                        sparse_output=False,
                        handle_unknown='ignore'
                    )
                    encoded = self.onehot_encoders[col].fit_transform(df[[col]].fillna(self.categorical_statistics[f"{col}_fill"]))
                else:
                    encoded = self.onehot_encoders[col].transform(df[[col]].fillna(self.categorical_statistics[f"{col}_fill"]))
                
                feature_names = [f"{col}_{val}" for val in 
                               self.onehot_encoders[col].categories_[0]]
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=feature_names,
                    index=df.index
                )
                
                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(columns=[col])
        
        return df
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are not needed."""
        df = df.copy()
        for col in self.todrop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        """Fill missing values using stored statistics."""
        df = df.copy()
        
        if training:
            self._compute_statistics(df)
        
        # Fill numeric missing values
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if f"{col}_fill" in self.numeric_statistics:
                df[col] = df[col].fillna(self.numeric_statistics[f"{col}_fill"])
        
        return df
    
    
    def transform(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Transform data using processor, optionally fitting it first.
        
        Args:
            df: DataFrame to transform
            fit: Whether to fit the processor on this data
        
        Returns:
            Transformed DataFrame
        """
        if not self.numeric_statistics and not fit:
            raise ValueError("Processor not fitted. Call with fit=True or use fit() before transform().")
        
        df = df.copy()

        # Sequential processing
        df = self._process_dates(df)
        df = self._process_insured_years(df)
        df = self._process_vehicle_use(df)
        df = self._process_vehicle_acquisition_state(df)
        df = self._process_total_claims(df)
        # Fit step (compute statistics) if requested
        if fit:
            self._compute_statistics(df)
            
        df = self._handle_missing_values(df, training=fit)
        df = self._process_categorical(df, training=fit)
        df = self._drop_columns(df)

        
        # Ensure all features are numeric
        non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
        non_numeric_cols = [col for col in non_numeric_cols 
                            if col not in ['quote_id']]
        if len(non_numeric_cols) > 0:
            raise ValueError(f"Non-numeric columns found after processing: {non_numeric_cols}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the processor and transform training data."""
        return self.transform(df, fit=True)