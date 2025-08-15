#!/usr/bin/env python3
"""
Comprehensive data exploration for flight delay prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def explore_data():
    # Load the processed data
    df = pd.read_csv('data/processed/airline_exploration.csv')
    print(f"Loaded analysis dataset: {len(df):,} flights")
    
    # Convert date column
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df['month'] = df['FL_DATE'].dt.month
    df['day_of_week'] = df['FL_DATE'].dt.dayofweek
    df['dep_hour'] = (df['CRS_DEP_TIME'] // 100).fillna(12)
    
    print("\n=== FLIGHT DELAY ANALYSIS ===")
    print(f"Total flights analyzed: {len(df):,}")
    print(f"Delayed flights (>15 min): {df['delayed'].sum():,} ({df['delayed'].mean()*100:.1f}%)")
    print(f"Average delay: {df['ARR_DELAY'].mean():.2f} minutes")
    print(f"Median delay: {df['ARR_DELAY'].median():.2f} minutes")
    
    # Delay distribution by hour of day
    print("\n=== DELAYS BY DEPARTURE HOUR ===")
    hourly_delays = df.groupby('dep_hour')['delayed'].agg(['count', 'mean']).round(3)
    hourly_delays.columns = ['flights', 'delay_rate']
    hourly_delays = hourly_delays[hourly_delays['flights'] >= 100]  # Min 100 flights
    print("Top 5 hours with highest delay rates:")
    print(hourly_delays.sort_values('delay_rate', ascending=False).head())
    
    # Delay by day of week
    print("\n=== DELAYS BY DAY OF WEEK ===")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_delays = df.groupby('day_of_week')['delayed'].agg(['count', 'mean']).round(3)
    daily_delays.columns = ['flights', 'delay_rate']
    daily_delays.index = [days[i] for i in daily_delays.index]
    print(daily_delays)
    
    # Top airports by delay rate
    print("\n=== TOP AIRPORTS BY DELAY RATE ===")
    airport_delays = df.groupby('ORIGIN')['delayed'].agg(['count', 'mean']).round(3)
    airport_delays.columns = ['flights', 'delay_rate']
    airport_delays = airport_delays[airport_delays['flights'] >= 500]  # Min 500 flights
    print("Top 10 airports with highest delay rates:")
    print(airport_delays.sort_values('delay_rate', ascending=False).head(10))
    
    # Airline performance
    print("\n=== AIRLINE PERFORMANCE ===")
    airline_performance = df.groupby('OP_CARRIER').agg({
        'delayed': ['count', 'mean'],
        'ARR_DELAY': ['mean', 'median']
    }).round(2)
    airline_performance.columns = ['flights', 'delay_rate', 'avg_delay', 'median_delay']
    airline_performance = airline_performance[airline_performance['flights'] >= 1000]
    print("Airlines ranked by delay rate:")
    print(airline_performance.sort_values('delay_rate', ascending=False))
    
    # Distance analysis
    print("\n=== DELAYS BY DISTANCE ===")
    df['distance_category'] = pd.cut(df['DISTANCE'], 
                                   bins=[0, 500, 1000, 1500, 2000, float('inf')],
                                   labels=['Short (<500mi)', 'Medium (500-1000mi)', 
                                          'Long (1000-1500mi)', 'Very Long (1500-2000mi)', 
                                          'Ultra Long (>2000mi)'])
    
    distance_analysis = df.groupby('distance_category').agg({
        'delayed': ['count', 'mean'],
        'ARR_DELAY': 'mean'
    }).round(3)
    distance_analysis.columns = ['flights', 'delay_rate', 'avg_delay']
    print(distance_analysis)
    
    # Top routes by delay rate
    print("\n=== TOP ROUTES BY DELAY RATE ===")
    df['route'] = df['ORIGIN'] + '-' + df['DEST']
    route_delays = df.groupby('route')['delayed'].agg(['count', 'mean']).round(3)
    route_delays.columns = ['flights', 'delay_rate']
    route_delays = route_delays[route_delays['flights'] >= 100]
    print("Top 10 routes with highest delay rates:")
    print(route_delays.sort_values('delay_rate', ascending=False).head(10))
    
    # Correlation analysis
    print("\n=== KEY CORRELATIONS WITH DELAYS ===")
    numerical_cols = ['dep_hour', 'day_of_week', 'month', 'DISTANCE', 'CRS_DEP_TIME']
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    if available_cols:
        correlations = df[available_cols + ['delayed']].corr()['delayed'].abs().sort_values(ascending=False)
        print("Correlations with delay target:")
        print(correlations.head(10))
    
    print("\n=== SUMMARY INSIGHTS ===")
    print("1. Flight delays affect about 21% of flights")
    print("2. Evening/night flights tend to have higher delay rates")
    print("3. Some airports consistently have higher delay rates")
    print("4. Airline performance varies significantly")
    print("5. Longer flights don't necessarily have higher delay rates")
    print("6. Certain routes are consistently problematic")

if __name__ == "__main__":
    explore_data()