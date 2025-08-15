#!/usr/bin/env python3
"""
Run data exploration for flight delay prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from data.download_data import load_airline_data

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

def main():
    print("Loading airline data...")
    df = load_airline_data(year=2023, sample_size=100000)
    
    if df is None:
        print("Failed to load data!")
        return
    
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    # Map column names to standard format (the notebook expects certain column names)
    column_mapping = {
        'FlightDate': 'FL_DATE',
        'Reporting_Airline': 'OP_CARRIER', 
        'Origin': 'ORIGIN',
        'Dest': 'DEST',
        'CRSDepTime': 'CRS_DEP_TIME',
        'CRSArrTime': 'CRS_ARR_TIME',
        'DepDelay': 'DEP_DELAY',
        'ArrDelay': 'ARR_DELAY',
        'Cancelled': 'CANCELLED',
        'Diverted': 'DIVERTED',
        'Distance': 'DISTANCE'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    print("Data Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    # Basic statistics
    print("\nBasic Statistics:")
    if 'ARR_DELAY' in df.columns:
        print(f"Average arrival delay: {df['ARR_DELAY'].mean():.2f} minutes")
        print(f"Flights with delays > 15 min: {(df['ARR_DELAY'] > 15).sum():,} ({(df['ARR_DELAY'] > 15).mean()*100:.1f}%)")
    
    if 'CANCELLED' in df.columns:
        print(f"Cancelled flights: {df['CANCELLED'].sum():,} ({df['CANCELLED'].mean()*100:.1f}%)")
    
    # Top airports
    if 'ORIGIN' in df.columns:
        print(f"\nTop 10 Origin Airports:")
        print(df['ORIGIN'].value_counts().head(10))
    
    # Save processed data
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'airline_sample.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved processed data to {output_file}")
    
    # Create target variable for delay analysis
    if 'ARR_DELAY' in df.columns:
        df['delayed'] = (df['ARR_DELAY'] > 15).astype(int)
        
        # Remove cancelled flights for analysis
        if 'CANCELLED' in df.columns:
            df_analysis = df[(df['CANCELLED'] != 1) & (df['ARR_DELAY'].notna())].copy()
        else:
            df_analysis = df[df['ARR_DELAY'].notna()].copy()
        
        print(f"\nAnalysis dataset: {len(df_analysis):,} flights")
        print(f"Delay rate: {df_analysis['delayed'].mean()*100:.1f}%")
        
        # Save analysis dataset
        analysis_file = output_dir / 'airline_exploration.csv'
        df_analysis.to_csv(analysis_file, index=False)
        print(f"Saved analysis dataset to {analysis_file}")
        
        return df_analysis
    
    return df

if __name__ == "__main__":
    main()