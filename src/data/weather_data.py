import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

load_dotenv()

class WeatherDataCollector:
    def __init__(self, api_key=None):
        """
        Initialize weather data collector with Visual Crossing API
        Get free API key from: https://www.visualcrossing.com/weather-api
        """
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        
    def get_airport_coordinates(self):
        """
        Return dictionary of major US airport coordinates
        """
        return {
            'ATL': (33.6407, -84.4277),  # Atlanta
            'LAX': (33.9425, -118.4081), # Los Angeles
            'ORD': (41.9742, -87.9073),  # Chicago O'Hare
            'DFW': (32.8998, -97.0403),  # Dallas/Fort Worth
            'DEN': (39.8561, -104.6737), # Denver
            'JFK': (40.6413, -73.7781),  # New York JFK
            'SFO': (37.6213, -122.3790), # San Francisco
            'SEA': (47.4502, -122.3088), # Seattle
            'LAS': (36.0840, -115.1537), # Las Vegas
            'MCO': (28.4312, -81.3081),  # Orlando
            'EWR': (40.6895, -74.1745),  # Newark
            'CLT': (35.2144, -80.9473),  # Charlotte
            'PHX': (33.4484, -112.0740), # Phoenix
            'IAH': (29.9902, -95.3368),  # Houston Intercontinental
            'MIA': (25.7959, -80.2870),  # Miami
            'BOS': (42.3656, -71.0096),  # Boston
            'MSP': (44.8848, -93.2223),  # Minneapolis
            'LGA': (40.7769, -73.8740),  # LaGuardia
            'DTW': (42.2162, -83.3554),  # Detroit
            'PHL': (39.8719, -75.2411),  # Philadelphia
        }
    
    def get_weather_data(self, airport_code, date, hour=12):
        """
        Get weather data for specific airport and date
        
        Args:
            airport_code (str): 3-letter airport code
            date (str): Date in YYYY-MM-DD format
            hour (int): Hour of day (0-23)
        
        Returns:
            dict: Weather data or None if failed
        """
        if not self.api_key:
            print("Warning: No weather API key found. Using mock data.")
            return self._get_mock_weather_data()
        
        coordinates = self.get_airport_coordinates()
        if airport_code not in coordinates:
            return None
        
        lat, lon = coordinates[airport_code]
        
        url = f"{self.base_url}/{lat},{lon}/{date}/{date}"
        
        params = {
            'key': self.api_key,
            'include': 'hours',
            'elements': 'datetime,temp,humidity,precip,windspeed,visibility,conditions'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract hourly data closest to requested hour
            day_data = data['days'][0]
            
            if 'hours' in day_data:
                target_hour = min(hour, len(day_data['hours']) - 1)
                hour_data = day_data['hours'][target_hour]
                
                return {
                    'airport': airport_code,
                    'date': date,
                    'hour': hour,
                    'temperature': hour_data.get('temp'),
                    'humidity': hour_data.get('humidity'),
                    'precipitation': hour_data.get('precip', 0),
                    'wind_speed': hour_data.get('windspeed'),
                    'visibility': hour_data.get('visibility'),
                    'conditions': hour_data.get('conditions', '')
                }
            else:
                # Use daily data if hourly not available
                return {
                    'airport': airport_code,
                    'date': date,
                    'hour': hour,
                    'temperature': day_data.get('temp'),
                    'humidity': day_data.get('humidity'),
                    'precipitation': day_data.get('precip', 0),
                    'wind_speed': day_data.get('windspeed'),
                    'visibility': day_data.get('visibility'),
                    'conditions': day_data.get('conditions', '')
                }
                
        except requests.RequestException as e:
            print(f"Error fetching weather data for {airport_code} on {date}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def _get_mock_weather_data(self):
        """
        Return mock weather data for development/testing
        """
        import random
        
        conditions = ['Clear', 'Partly cloudy', 'Cloudy', 'Rain', 'Snow', 'Fog']
        
        return {
            'temperature': round(random.uniform(20, 80), 1),
            'humidity': round(random.uniform(30, 90), 1),
            'precipitation': round(random.uniform(0, 2), 2),
            'wind_speed': round(random.uniform(0, 30), 1),
            'visibility': round(random.uniform(1, 10), 1),
            'conditions': random.choice(conditions)
        }
    
    def enrich_flight_data_with_weather(self, flight_df, sample_size=1000):
        """
        Add weather data to flight dataset
        
        Args:
            flight_df (pd.DataFrame): Flight data
            sample_size (int): Number of flights to enrich (for API limits)
        
        Returns:
            pd.DataFrame: Flight data with weather features
        """
        if len(flight_df) > sample_size:
            sample_df = flight_df.sample(n=sample_size, random_state=42).copy()
        else:
            sample_df = flight_df.copy()
        
        weather_data = []
        
        for idx, row in sample_df.iterrows():
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(sample_df)} flights...")
            
            # Get origin airport weather
            origin_weather = self.get_weather_data(
                row.get('ORIGIN', ''),
                row.get('FL_DATE', ''),
                int(row.get('CRS_DEP_TIME', 1200) // 100)  # Convert to hour
            )
            
            # Get destination airport weather
            dest_weather = self.get_weather_data(
                row.get('DEST', ''),
                row.get('FL_DATE', ''),
                int(row.get('CRS_ARR_TIME', 1400) // 100)  # Convert to hour
            )
            
            weather_row = {'flight_index': idx}
            
            if origin_weather:
                for key, value in origin_weather.items():
                    if key not in ['airport', 'date', 'hour']:
                        weather_row[f'origin_{key}'] = value
            
            if dest_weather:
                for key, value in dest_weather.items():
                    if key not in ['airport', 'date', 'hour']:
                        weather_row[f'dest_{key}'] = value
            
            weather_data.append(weather_row)
            
            # Rate limiting for API
            if self.api_key:
                time.sleep(0.1)  # Be nice to the API
        
        weather_df = pd.DataFrame(weather_data)
        
        # Merge with flight data
        sample_df = sample_df.reset_index().rename(columns={'index': 'flight_index'})
        enriched_df = sample_df.merge(weather_df, on='flight_index', how='left')
        
        return enriched_df

if __name__ == "__main__":
    # Test weather data collection
    collector = WeatherDataCollector()
    
    # Test single airport
    weather = collector.get_weather_data('ATL', '2023-01-15', 12)
    print("Sample weather data:")
    print(weather)