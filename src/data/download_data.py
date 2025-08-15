import pandas as pd
import requests
import os
from pathlib import Path
import zipfile

def download_airline_data(year=2023, months=None):
    """
    Download airline on-time performance data from Bureau of Transportation Statistics
    
    Args:
        year (int): Year to download data for
        months (list): List of months to download (1-12). If None, downloads all months.
    """
    if months is None:
        months = range(1, 13)
    
    base_url = "https://transtats.bts.gov/PREZIP/"
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for month in months:
        filename = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
        url = f"{base_url}{filename}"
        
        print(f"Downloading data for {year}-{month:02d}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            zip_path = data_dir / filename
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove the zip file to save space
            os.remove(zip_path)
            
            print(f"Successfully downloaded and extracted data for {year}-{month:02d}")
            
        except requests.RequestException as e:
            print(f"Failed to download data for {year}-{month:02d}: {e}")
        except zipfile.BadZipFile as e:
            print(f"Failed to extract zip file for {year}-{month:02d}: {e}")

def load_airline_data(year=2023, sample_size=None):
    """
    Load and combine airline data from multiple CSV files
    
    Args:
        year (int): Year to load data for
        sample_size (int): If specified, return a random sample of this size
    
    Returns:
        pandas.DataFrame: Combined airline data
    """
    data_dir = Path("data/raw")
    csv_files = list(data_dir.glob(f"*{year}*.csv"))
    
    if not csv_files:
        print(f"No CSV files found for year {year}. Please download the data first.")
        return None
    
    dataframes = []
    for file in csv_files:
        print(f"Loading {file.name}...")
        df = pd.read_csv(file, low_memory=False)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    if sample_size and len(combined_df) > sample_size:
        combined_df = combined_df.sample(n=sample_size, random_state=42)
    
    return combined_df

if __name__ == "__main__":
    # Download data for 2023 (you can change this or add more years)
    print("Starting data download...")
    download_airline_data(year=2023, months=[1, 2, 3])  # Start with first 3 months for testing
    
    print("Loading data...")
    df = load_airline_data(year=2023, sample_size=100000)  # Load sample for testing
    
    if df is not None:
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        print("\nColumn names:")
        print(df.columns.tolist())
        
        # Save sample for quick access
        df.to_csv("data/processed/airline_sample.csv", index=False)
        print("Saved sample to data/processed/airline_sample.csv")