#!/usr/bin/env python3
"""
Flight Delay Prediction Project Runner

This script provides an easy way to run the entire project pipeline or specific components.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_notebooks():
    """Run all notebooks in sequence"""
    notebooks = [
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_feature_engineering.ipynb", 
        "notebooks/03_model_training.ipynb"
    ]
    
    print("Running notebooks in sequence...")
    for notebook in notebooks:
        if Path(notebook).exists():
            print(f"\n{'='*50}")
            print(f"Running {notebook}")
            print('='*50)
            
            # Convert notebook to script and run
            script_name = notebook.replace('.ipynb', '_temp.py')
            
            # Convert notebook to python script
            subprocess.run([
                'jupyter', 'nbconvert', '--to', 'python', 
                '--output', script_name, notebook
            ], check=True)
            
            # Run the script
            subprocess.run(['python', script_name], check=True)
            
            # Clean up temporary script
            os.remove(script_name)
            
            print(f"✅ Completed {notebook}")
        else:
            print(f"❌ Notebook not found: {notebook}")

def run_streamlit():
    """Launch the Streamlit web application"""
    print("Launching Streamlit web application...")
    print("Open your browser and go to: http://localhost:8501")
    subprocess.run(['streamlit', 'run', 'app/streamlit_app.py'])

def download_sample_data():
    """Download sample data only"""
    print("Downloading sample airline data...")
    
    # Add src to Python path
    sys.path.append('src')
    
    from data.download_data import download_airline_data, load_airline_data
    
    # Download first 3 months of 2023
    download_airline_data(year=2023, months=[1, 2, 3])
    
    # Load and save sample
    df = load_airline_data(year=2023, sample_size=100000)
    if df is not None:
        # Basic preprocessing
        df['delayed'] = (df['ARR_DELAY'] > 15).astype(int)
        if 'CANCELLED' in df.columns:
            df = df[(df['CANCELLED'] != 1) & (df['ARR_DELAY'].notna())].copy()
        
        # Save sample
        output_path = "data/processed/airline_sample.csv"
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Sample data saved to {output_path}")
        print(f"Shape: {df.shape}, Delay rate: {df['delayed'].mean()*100:.1f}%")

def setup_environment():
    """Set up the project environment"""
    print("Setting up project environment...")
    
    # Create directories
    directories = [
        "data/raw", "data/processed", "models", 
        "notebooks", "src", "app"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Check if requirements are installed
    try:
        import pandas, numpy, sklearn, xgboost, lightgbm, streamlit
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Flight Delay Prediction Project Runner")
    parser.add_argument('command', choices=[
        'setup', 'download', 'notebooks', 'app', 'all'
    ], help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_environment()
    
    elif args.command == 'download':
        download_sample_data()
    
    elif args.command == 'notebooks':
        if not setup_environment():
            return
        run_notebooks()
    
    elif args.command == 'app':
        run_streamlit()
    
    elif args.command == 'all':
        print("Running complete project pipeline...")
        if not setup_environment():
            return
        download_sample_data()
        run_notebooks()
        print("\n" + "="*60)
        print("🎉 Project pipeline completed successfully!")
        print("="*60)
        print("\nTo launch the web app, run:")
        print("python run_project.py app")

if __name__ == "__main__":
    main()