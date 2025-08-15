import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FlightFeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_time_features(self, df):
        """
        Create time-based features from flight datetime
        """
        df = df.copy()
        
        # Convert FL_DATE to datetime
        df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
        
        # Extract basic time features
        df['year'] = df['FL_DATE'].dt.year
        df['month'] = df['FL_DATE'].dt.month
        df['day_of_week'] = df['FL_DATE'].dt.dayofweek
        df['day_of_month'] = df['FL_DATE'].dt.day
        df['quarter'] = df['FL_DATE'].dt.quarter
        
        # Create cyclical features (important for capturing patterns)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Hour features from departure and arrival times
        if 'CRS_DEP_TIME' in df.columns:
            df['dep_hour'] = (df['CRS_DEP_TIME'] // 100).fillna(12)
            df['dep_hour_sin'] = np.sin(2 * np.pi * df['dep_hour'] / 24)
            df['dep_hour_cos'] = np.cos(2 * np.pi * df['dep_hour'] / 24)
        
        if 'CRS_ARR_TIME' in df.columns:
            df['arr_hour'] = (df['CRS_ARR_TIME'] // 100).fillna(12)
            df['arr_hour_sin'] = np.sin(2 * np.pi * df['arr_hour'] / 24)
            df['arr_hour_cos'] = np.cos(2 * np.pi * df['arr_hour'] / 24)
        
        # Holiday features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        
        return df
    
    def create_airport_features(self, df):
        """
        Create airport-related features
        """
        df = df.copy()
        
        # Create route feature
        df['route'] = df['ORIGIN'] + '_' + df['DEST']
        
        # Calculate flight distance (simplified using coordinates if available)
        # For now, we'll create a simple feature based on common routes
        major_airports = ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA']
        
        df['origin_is_major'] = df['ORIGIN'].isin(major_airports).astype(int)
        df['dest_is_major'] = df['DEST'].isin(major_airports).astype(int)
        df['is_major_route'] = (df['origin_is_major'] & df['dest_is_major']).astype(int)
        
        return df
    
    def create_aircraft_lag_features(self, df):
        """
        Create features based on aircraft's previous flight performance
        This is often a strong predictor of delays
        """
        df = df.copy()
        
        if 'TAIL_NUM' not in df.columns:
            return df
        
        # Sort by aircraft and date/time
        df = df.sort_values(['TAIL_NUM', 'FL_DATE', 'CRS_DEP_TIME'])
        
        # Calculate previous flight arrival delay for same aircraft
        df['prev_flight_arr_delay'] = df.groupby('TAIL_NUM')['ARR_DELAY'].shift(1)
        
        # Time since last flight for same aircraft
        df['FL_DATETIME'] = pd.to_datetime(df['FL_DATE'].astype(str) + ' ' + 
                                          (df['CRS_DEP_TIME'] // 100).astype(str).str.zfill(2) + ':' +
                                          (df['CRS_DEP_TIME'] % 100).astype(str).str.zfill(2))
        
        df['prev_flight_datetime'] = df.groupby('TAIL_NUM')['FL_DATETIME'].shift(1)
        df['hours_since_last_flight'] = (df['FL_DATETIME'] - df['prev_flight_datetime']).dt.total_seconds() / 3600
        
        # Create lag features
        df['prev_flight_delayed'] = (df['prev_flight_arr_delay'] > 15).astype(int)
        df['aircraft_has_recent_delay'] = (df['prev_flight_arr_delay'] > 0).astype(int)
        
        return df
    
    def create_congestion_features(self, df):
        """
        Create airport congestion features
        """
        df = df.copy()
        
        # Create datetime for easier grouping
        df['FL_DATETIME'] = pd.to_datetime(df['FL_DATE'].astype(str) + ' ' + 
                                          (df['CRS_DEP_TIME'] // 100).fillna(12).astype(int).astype(str).str.zfill(2) + ':' +
                                          (df['CRS_DEP_TIME'] % 100).fillna(0).astype(int).astype(str).str.zfill(2))
        
        # Count flights departing from same airport within 1-hour window
        df['dep_hour_window'] = df['FL_DATETIME'].dt.floor('H')
        origin_congestion = df.groupby(['ORIGIN', 'dep_hour_window']).size().reset_index(name='origin_departures_per_hour')
        df = df.merge(origin_congestion, on=['ORIGIN', 'dep_hour_window'], how='left')
        
        # Count flights arriving at same airport within 1-hour window
        df['arr_hour_window'] = pd.to_datetime(df['FL_DATE'].astype(str) + ' ' + 
                                              (df['CRS_ARR_TIME'] // 100).fillna(12).astype(int).astype(str).str.zfill(2) + ':' +
                                              (df['CRS_ARR_TIME'] % 100).fillna(0).astype(int).astype(str).str.zfill(2)).dt.floor('H')
        
        dest_congestion = df.groupby(['DEST', 'arr_hour_window']).size().reset_index(name='dest_arrivals_per_hour')
        df = df.merge(dest_congestion, on=['DEST', 'arr_hour_window'], how='left')
        
        return df
    
    def create_airline_features(self, df):
        """
        Create airline-specific features
        """
        df = df.copy()
        
        # Airline historical performance (simplified)
        airline_stats = df.groupby('OP_CARRIER')['ARR_DELAY'].agg(['mean', 'std']).reset_index()
        airline_stats.columns = ['OP_CARRIER', 'airline_avg_delay', 'airline_delay_std']
        df = df.merge(airline_stats, on='OP_CARRIER', how='left')
        
        # Route-specific airline performance
        route_airline_stats = df.groupby(['route', 'OP_CARRIER'])['ARR_DELAY'].mean().reset_index()
        route_airline_stats.columns = ['route', 'OP_CARRIER', 'route_airline_avg_delay']
        df = df.merge(route_airline_stats, on=['route', 'OP_CARRIER'], how='left')
        
        return df
    
    def create_weather_features(self, df):
        """
        Create derived weather features
        """
        df = df.copy()
        
        weather_cols = [col for col in df.columns if 'origin_' in col or 'dest_' in col]
        
        if not weather_cols:
            return df
        
        # Create adverse weather indicators
        if 'origin_conditions' in df.columns:
            adverse_conditions = ['Rain', 'Snow', 'Fog', 'Thunderstorm', 'Heavy']
            df['origin_adverse_weather'] = df['origin_conditions'].str.contains('|'.join(adverse_conditions), na=False).astype(int)
        
        if 'dest_conditions' in df.columns:
            df['dest_adverse_weather'] = df['dest_conditions'].str.contains('|'.join(adverse_conditions), na=False).astype(int)
        
        # Low visibility indicator
        if 'origin_visibility' in df.columns:
            df['origin_low_visibility'] = (df['origin_visibility'] < 3).astype(int)
        
        if 'dest_visibility' in df.columns:
            df['dest_low_visibility'] = (df['dest_visibility'] < 3).astype(int)
        
        # High wind indicator
        if 'origin_wind_speed' in df.columns:
            df['origin_high_wind'] = (df['origin_wind_speed'] > 25).astype(int)
        
        if 'dest_wind_speed' in df.columns:
            df['dest_high_wind'] = (df['dest_wind_speed'] > 25).astype(int)
        
        # Precipitation indicator
        if 'origin_precipitation' in df.columns:
            df['origin_precipitation_flag'] = (df['origin_precipitation'] > 0).astype(int)
        
        if 'dest_precipitation' in df.columns:
            df['dest_precipitation_flag'] = (df['dest_precipitation'] > 0).astype(int)
        
        return df
    
    def create_target_variable(self, df, delay_threshold=15):
        """
        Create binary target variable for delays
        """
        df = df.copy()
        
        if 'ARR_DELAY' in df.columns:
            df['delayed'] = (df['ARR_DELAY'] > delay_threshold).astype(int)
        
        return df
    
    def encode_categorical_features(self, df, categorical_cols=None):
        """
        Encode categorical features
        """
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = ['OP_CARRIER', 'ORIGIN', 'DEST', 'route']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # For unseen categories, assign a default value
                        df[f'{col}_encoded'] = 0
        
        return df
    
    def engineer_all_features(self, df):
        """
        Apply all feature engineering steps
        """
        print("Creating time features...")
        df = self.create_time_features(df)
        
        print("Creating airport features...")
        df = self.create_airport_features(df)
        
        print("Creating aircraft lag features...")
        df = self.create_aircraft_lag_features(df)
        
        print("Creating congestion features...")
        df = self.create_congestion_features(df)
        
        print("Creating airline features...")
        df = self.create_airline_features(df)
        
        print("Creating weather features...")
        df = self.create_weather_features(df)
        
        print("Creating target variable...")
        df = self.create_target_variable(df)
        
        print("Encoding categorical features...")
        df = self.encode_categorical_features(df)
        
        return df

if __name__ == "__main__":
    # Test feature engineering with sample data
    engineer = FlightFeatureEngineer()
    
    # Create sample data
    sample_data = {
        'FL_DATE': ['2023-01-15', '2023-01-15', '2023-01-16'],
        'CRS_DEP_TIME': [800, 1200, 1500],
        'CRS_ARR_TIME': [1000, 1400, 1700],
        'ORIGIN': ['ATL', 'LAX', 'ORD'],
        'DEST': ['LAX', 'JFK', 'DFW'],
        'OP_CARRIER': ['DL', 'AA', 'UA'],
        'ARR_DELAY': [5, 25, -10],
        'TAIL_NUM': ['N123DL', 'N456AA', 'N789UA']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    
    engineered_df = engineer.engineer_all_features(df)
    print("Engineered data shape:", engineered_df.shape)
    print("New features:", [col for col in engineered_df.columns if col not in df.columns])