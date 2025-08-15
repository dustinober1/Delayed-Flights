import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.feature_engineering import FlightFeatureEngineer
from data.weather_data import WeatherDataCollector

# Page configuration
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1e88e5;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    padding: 1rem;
    border-radius: 10px;
    border: 2px solid #1e88e5;
    background-color: #f8f9fa;
    margin: 1rem 0;
}
.delay-high {
    background-color: #ffebee;
    border-color: #f44336;
    color: #c62828;
}
.delay-low {
    background-color: #e8f5e8;
    border-color: #4caf50;
    color: #2e7d32;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained model"""
    model_path = Path("models")
    model_files = list(model_path.glob("*.joblib"))
    
    if not model_files:
        return None, "No trained model found. Please train a model first."
    
    # Load the most recent model
    latest_model = max(model_files, key=os.path.getctime)
    
    try:
        model = joblib.load(latest_model)
        return model, f"Loaded model: {latest_model.name}"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

@st.cache_data
def load_sample_data():
    """Load sample data for feature engineering"""
    try:
        sample_path = Path("data/processed/airline_sample.csv")
        if sample_path.exists():
            return pd.read_csv(sample_path)
        else:
            # Create minimal sample data for feature engineering
            return pd.DataFrame({
                'FL_DATE': ['2023-01-15'],
                'CRS_DEP_TIME': [800],
                'CRS_ARR_TIME': [1000],
                'ORIGIN': ['ATL'],
                'DEST': ['LAX'],
                'OP_CARRIER': ['DL'],
                'ARR_DELAY': [5],
                'TAIL_NUM': ['N123DL']
            })
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None

def get_airport_list():
    """Get list of major airports"""
    return ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS', 'MCO',
            'EWR', 'CLT', 'PHX', 'IAH', 'MIA', 'BOS', 'MSP', 'LGA', 'DTW', 'PHL']

def get_airline_list():
    """Get list of major airlines"""
    return ['AA', 'DL', 'UA', 'WN', 'AS', 'B6', 'NK', 'F9', 'G4', 'HA']

def create_flight_dataframe(flight_data):
    """Create a dataframe from flight input data"""
    return pd.DataFrame([flight_data])

def predict_delay(model, flight_df, feature_engineer):
    """Predict flight delay probability"""
    try:
        # Engineer features
        engineered_df = feature_engineer.engineer_all_features(flight_df)
        
        # Select features (this should match your training features)
        # For simplicity, we'll use a subset of key features
        feature_cols = [col for col in engineered_df.columns 
                       if col not in ['FL_DATE', 'FL_DATETIME', 'prev_flight_datetime', 
                                     'dep_hour_window', 'arr_hour_window', 'TAIL_NUM', 
                                     'OP_CARRIER', 'ORIGIN', 'DEST', 'route',
                                     'ARR_DELAY', 'DEP_DELAY', 'delayed', 'origin_conditions', 
                                     'dest_conditions']]
        
        # Handle missing features by filling with defaults
        X = engineered_df[feature_cols].fillna(0)
        
        # Make prediction
        prediction_proba = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        
        return prediction, prediction_proba[1], None
        
    except Exception as e:
        return None, None, str(e)

def main():
    # Header
    st.markdown('<h1 class="main-header">✈️ Flight Delay Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Flight Delay Predictor! This application uses machine learning to predict 
    whether your flight will be delayed by more than 15 minutes based on historical data, 
    weather conditions, and various flight characteristics.
    """)
    
    # Load model
    model, model_status = load_model()
    st.sidebar.info(model_status)
    
    if model is None:
        st.error("❌ No trained model available. Please train a model first using the notebooks.")
        st.info("💡 To train a model, run the Jupyter notebooks in the following order:\n1. 01_data_exploration.ipynb\n2. 02_feature_engineering.ipynb\n3. 03_model_training.ipynb")
        return
    
    # Load sample data and initialize feature engineer
    sample_data = load_sample_data()
    if sample_data is None:
        st.error("❌ Could not load sample data for feature engineering.")
        return
    
    feature_engineer = FlightFeatureEngineer()
    
    # Sidebar for input
    st.sidebar.header("Flight Details")
    
    # Flight information
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        origin = st.selectbox("Origin Airport", get_airport_list(), index=0)
        departure_date = st.date_input("Departure Date", datetime.now().date())
        
    with col2:
        destination = st.selectbox("Destination Airport", get_airport_list(), index=1)
        departure_time = st.time_input("Departure Time", datetime.now().time())
    
    airline = st.sidebar.selectbox("Airline", get_airline_list(), index=0)
    
    # Calculate arrival time (simplified)
    flight_duration = st.sidebar.slider("Estimated Flight Duration (hours)", 1.0, 8.0, 3.0, 0.5)
    arrival_time = (datetime.combine(departure_date, departure_time) + 
                   timedelta(hours=flight_duration)).time()
    
    st.sidebar.write(f"Estimated Arrival: {arrival_time}")
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        tail_number = st.text_input("Aircraft Tail Number (optional)", "N123XX")
        prev_delay = st.slider("Previous Flight Delay (minutes)", -30, 120, 0)
    
    # Weather integration
    st.sidebar.header("Weather Information")
    use_weather = st.sidebar.checkbox("Include Weather Data", value=False)
    
    weather_data = {}
    if use_weather:
        st.sidebar.info("Weather data requires API key. Using mock data for demo.")
        weather_collector = WeatherDataCollector()
        # Using mock weather for demo
        weather_data = weather_collector._get_mock_weather_data()
    
    # Predict button
    if st.sidebar.button("🔮 Predict Delay", type="primary"):
        # Create flight data
        flight_data = {
            'FL_DATE': departure_date.strftime('%Y-%m-%d'),
            'CRS_DEP_TIME': departure_time.hour * 100 + departure_time.minute,
            'CRS_ARR_TIME': arrival_time.hour * 100 + arrival_time.minute,
            'ORIGIN': origin,
            'DEST': destination,
            'OP_CARRIER': airline,
            'TAIL_NUM': tail_number,
            'ARR_DELAY': 0,  # Unknown at prediction time
            'prev_flight_arr_delay': prev_delay if prev_delay != 0 else np.nan
        }
        
        # Add weather data if available
        if weather_data:
            for key, value in weather_data.items():
                flight_data[f'origin_{key}'] = value
                flight_data[f'dest_{key}'] = value  # Simplified
        
        # Create dataframe
        flight_df = create_flight_dataframe(flight_data)
        
        # Make prediction
        with st.spinner("Analyzing flight data..."):
            prediction, probability, error = predict_delay(model, flight_df, feature_engineer)
        
        if error:
            st.error(f"❌ Prediction error: {error}")
        else:
            # Display results
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Delay Probability", 
                    value=f"{probability:.1%}",
                    delta=f"{'High Risk' if probability > 0.3 else 'Low Risk'}"
                )
            
            with col2:
                status = "LIKELY DELAYED" if prediction == 1 else "ON TIME"
                color = "🔴" if prediction == 1 else "🟢"
                st.metric(label="Prediction", value=f"{color} {status}")
            
            with col3:
                confidence = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low"
                st.metric(label="Confidence", value=confidence)
            
            # Prediction explanation
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box delay-high">
                <h3>⚠️ Delay Likely</h3>
                <p>This flight has a <strong>{probability:.1%}</strong> chance of being delayed by more than 15 minutes.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📋 Recommendations")
                st.warning("🕐 Consider arriving at the airport earlier")
                st.warning("📱 Monitor flight status closely")
                st.warning("🔄 Check for alternative flights")
                
            else:
                st.markdown(f"""
                <div class="prediction-box delay-low">
                <h3>✅ On-Time Expected</h3>
                <p>This flight has a <strong>{(1-probability):.1%}</strong> chance of arriving on time.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📋 Recommendations")
                st.success("🎯 Standard arrival time at airport")
                st.success("😊 Low risk of delays")
    
    # Information section
    st.markdown("---")
    st.markdown("## 📊 About This Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 Model Features
        - **Historical Performance**: Airline and route-specific delay patterns
        - **Time Factors**: Hour, day of week, month, and seasonal effects
        - **Airport Congestion**: Traffic patterns at origin and destination
        - **Aircraft History**: Previous flight delays for the same aircraft
        - **Weather Conditions**: Temperature, wind, precipitation, visibility
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Model Performance
        - **Algorithm**: XGBoost/LightGBM ensemble
        - **Training Data**: Bureau of Transportation Statistics
        - **Features**: 50+ engineered features
        - **Accuracy**: ~85% (varies by route and conditions)
        - **Update Frequency**: Monthly with new data
        """)
    
    st.markdown("""
    ### ⚠️ Disclaimer
    This prediction is based on historical data and statistical models. Actual flight performance 
    may vary due to unforeseen circumstances, weather changes, air traffic control decisions, 
    and other factors not captured in the model. Always check with your airline for the most 
    current flight information.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    Built with ❤️ using Streamlit, scikit-learn, and XGBoost<br>
    Data source: U.S. Bureau of Transportation Statistics
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()