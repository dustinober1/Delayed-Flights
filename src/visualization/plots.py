import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

class FlightDelayVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_delay_distribution(self, df):
        """
        Plot distribution of flight delays
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of delays
        delays = df['ARR_DELAY'].dropna()
        delays_filtered = delays[(delays >= -60) & (delays <= 300)]  # Remove extreme outliers
        
        ax1.hist(delays_filtered, bins=50, alpha=0.7, color=self.colors[0])
        ax1.axvline(15, color='red', linestyle='--', label='15-min threshold')
        ax1.set_xlabel('Arrival Delay (minutes)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Flight Delays')
        ax1.legend()
        
        # Delay categories
        delay_categories = ['On-time (≤15 min)', 'Delayed (>15 min)']
        delay_counts = [(df['ARR_DELAY'] <= 15).sum(), (df['ARR_DELAY'] > 15).sum()]
        
        ax2.pie(delay_counts, labels=delay_categories, autopct='%1.1f%%', 
                colors=self.colors[:2], startangle=90)
        ax2.set_title('Flight Delay Categories')
        
        plt.tight_layout()
        plt.show()
    
    def plot_delays_by_time(self, df):
        """
        Plot delays by different time dimensions
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # By hour of day
        hourly_delays = df.groupby('dep_hour')['delayed'].mean() * 100
        axes[0, 0].plot(hourly_delays.index, hourly_delays.values, marker='o', color=self.colors[0])
        axes[0, 0].set_xlabel('Departure Hour')
        axes[0, 0].set_ylabel('Delay Rate (%)')
        axes[0, 0].set_title('Delay Rate by Departure Hour')
        axes[0, 0].grid(True, alpha=0.3)
        
        # By day of week
        dow_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_delays = df.groupby('day_of_week')['delayed'].mean() * 100
        axes[0, 1].bar(range(7), daily_delays.values, color=self.colors[1])
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Delay Rate (%)')
        axes[0, 1].set_title('Delay Rate by Day of Week')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(dow_labels, rotation=45)
        
        # By month
        monthly_delays = df.groupby('month')['delayed'].mean() * 100
        axes[1, 0].plot(monthly_delays.index, monthly_delays.values, marker='s', color=self.colors[2])
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Delay Rate (%)')
        axes[1, 0].set_title('Delay Rate by Month')
        axes[1, 0].grid(True, alpha=0.3)
        
        # By airline
        airline_delays = df.groupby('OP_CARRIER')['delayed'].mean().sort_values(ascending=False).head(10) * 100
        axes[1, 1].barh(range(len(airline_delays)), airline_delays.values, color=self.colors[3])
        axes[1, 1].set_xlabel('Delay Rate (%)')
        axes[1, 1].set_ylabel('Airline')
        axes[1, 1].set_title('Top 10 Airlines by Delay Rate')
        axes[1, 1].set_yticks(range(len(airline_delays)))
        axes[1, 1].set_yticklabels(airline_delays.index)
        
        plt.tight_layout()
        plt.show()
    
    def plot_airport_delays(self, df, top_n=15):
        """
        Plot delays by airport
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Origin airports
        origin_delays = df.groupby('ORIGIN')['delayed'].mean().sort_values(ascending=False).head(top_n) * 100
        ax1.barh(range(len(origin_delays)), origin_delays.values, color=self.colors[0])
        ax1.set_xlabel('Delay Rate (%)')
        ax1.set_ylabel('Origin Airport')
        ax1.set_title(f'Top {top_n} Origin Airports by Delay Rate')
        ax1.set_yticks(range(len(origin_delays)))
        ax1.set_yticklabels(origin_delays.index)
        
        # Destination airports
        dest_delays = df.groupby('DEST')['delayed'].mean().sort_values(ascending=False).head(top_n) * 100
        ax2.barh(range(len(dest_delays)), dest_delays.values, color=self.colors[1])
        ax2.set_xlabel('Delay Rate (%)')
        ax2.set_ylabel('Destination Airport')
        ax2.set_title(f'Top {top_n} Destination Airports by Delay Rate')
        ax2.set_yticks(range(len(dest_delays)))
        ax2.set_yticklabels(dest_delays.index)
        
        plt.tight_layout()
        plt.show()
    
    def plot_weather_impact(self, df):
        """
        Plot weather impact on delays
        """
        weather_cols = [col for col in df.columns if 'weather' in col or 'visibility' in col or 'wind' in col]
        
        if not weather_cols:
            print("No weather data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Adverse weather impact
        if 'origin_adverse_weather' in df.columns:
            weather_impact = df.groupby('origin_adverse_weather')['delayed'].mean() * 100
            labels = ['Normal Weather', 'Adverse Weather']
            axes[0, 0].bar(labels, weather_impact.values, color=[self.colors[0], self.colors[1]])
            axes[0, 0].set_ylabel('Delay Rate (%)')
            axes[0, 0].set_title('Impact of Adverse Weather on Delays (Origin)')
        
        # Visibility impact
        if 'origin_low_visibility' in df.columns:
            visibility_impact = df.groupby('origin_low_visibility')['delayed'].mean() * 100
            labels = ['Normal Visibility', 'Low Visibility']
            axes[0, 1].bar(labels, visibility_impact.values, color=[self.colors[2], self.colors[3]])
            axes[0, 1].set_ylabel('Delay Rate (%)')
            axes[0, 1].set_title('Impact of Low Visibility on Delays (Origin)')
        
        # Wind impact
        if 'origin_high_wind' in df.columns:
            wind_impact = df.groupby('origin_high_wind')['delayed'].mean() * 100
            labels = ['Normal Wind', 'High Wind']
            axes[1, 0].bar(labels, wind_impact.values, color=[self.colors[4], self.colors[0]])
            axes[1, 0].set_ylabel('Delay Rate (%)')
            axes[1, 0].set_title('Impact of High Wind on Delays (Origin)')
        
        # Precipitation impact
        if 'origin_precipitation_flag' in df.columns:
            precip_impact = df.groupby('origin_precipitation_flag')['delayed'].mean() * 100
            labels = ['No Precipitation', 'Precipitation']
            axes[1, 1].bar(labels, precip_impact.values, color=[self.colors[1], self.colors[2]])
            axes[1, 1].set_ylabel('Delay Rate (%)')
            axes[1, 1].set_title('Impact of Precipitation on Delays (Origin)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, df, figsize=(14, 10)):
        """
        Plot correlation heatmap of numerical features
        """
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remove some columns that are not meaningful for correlation
        exclude_cols = ['year', 'day_of_month', 'flight_index']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if len(numerical_cols) > 30:
            # If too many features, select most important ones
            important_cols = ['delayed', 'dep_hour', 'arr_hour', 'month', 'day_of_week',
                            'origin_departures_per_hour', 'dest_arrivals_per_hour',
                            'prev_flight_arr_delay', 'airline_avg_delay']
            important_cols = [col for col in important_cols if col in numerical_cols]
            numerical_cols = important_cols + [col for col in numerical_cols if 'weather' in col or 'wind' in col or 'visibility' in col][:10]
        
        corr_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def plot_interactive_delay_map(self, df, save_path="delay_map.html"):
        """
        Create interactive map showing delay rates by airport
        """
        # Airport coordinates (simplified)
        airport_coords = {
            'ATL': (33.6407, -84.4277), 'LAX': (33.9425, -118.4081), 'ORD': (41.9742, -87.9073),
            'DFW': (32.8998, -97.0403), 'DEN': (39.8561, -104.6737), 'JFK': (40.6413, -73.7781),
            'SFO': (37.6213, -122.3790), 'SEA': (47.4502, -122.3088), 'LAS': (36.0840, -115.1537),
            'MCO': (28.4312, -81.3081), 'EWR': (40.6895, -74.1745), 'CLT': (35.2144, -80.9473),
            'PHX': (33.4484, -112.0740), 'IAH': (29.9902, -95.3368), 'MIA': (25.7959, -80.2870),
            'BOS': (42.3656, -71.0096), 'MSP': (44.8848, -93.2223), 'LGA': (40.7769, -73.8740),
            'DTW': (42.2162, -83.3554), 'PHL': (39.8719, -75.2411)
        }
        
        # Calculate delay rates by origin airport
        delay_rates = df.groupby('ORIGIN').agg({
            'delayed': 'mean',
            'ORIGIN': 'count'
        }).rename(columns={'ORIGIN': 'flight_count'}).reset_index()
        
        # Filter for airports with coordinates and significant flight volume
        delay_rates = delay_rates[
            (delay_rates['ORIGIN'].isin(airport_coords.keys())) & 
            (delay_rates['flight_count'] >= 50)
        ]
        
        # Create map
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)  # Center on US
        
        for _, row in delay_rates.iterrows():
            airport = row['ORIGIN']
            delay_rate = row['delayed'] * 100
            flight_count = row['flight_count']
            
            if airport in airport_coords:
                lat, lon = airport_coords[airport]
                
                # Color based on delay rate
                if delay_rate < 15:
                    color = 'green'
                elif delay_rate < 25:
                    color = 'orange'
                else:
                    color = 'red'
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=min(flight_count / 100, 20),  # Size based on flight volume
                    popup=f"{airport}<br>Delay Rate: {delay_rate:.1f}%<br>Flights: {flight_count}",
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Save map
        m.save(save_path)
        print(f"Interactive delay map saved to {save_path}")
        return m
    
    def plot_model_comparison(self, results):
        """
        Plot model comparison results
        """
        model_names = [result['model_name'] for result in results]
        roc_scores = [result['roc_auc'] for result in results]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, roc_scores, color=self.colors[:len(model_names)])
        plt.ylabel('ROC-AUC Score')
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, roc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def create_summary_dashboard(self, df):
        """
        Create a comprehensive summary dashboard
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Delay Distribution', 'Delays by Hour', 'Delays by Airline', 
                          'Delays by Month', 'Weather Impact', 'Airport Performance'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Delay distribution
        fig.add_trace(
            go.Histogram(x=df['ARR_DELAY'], name='Delay Distribution', nbinsx=50),
            row=1, col=1
        )
        
        # Delays by hour
        hourly_delays = df.groupby('dep_hour')['delayed'].mean() * 100
        fig.add_trace(
            go.Scatter(x=hourly_delays.index, y=hourly_delays.values, 
                      mode='lines+markers', name='Hourly Delays'),
            row=1, col=2
        )
        
        # Top airlines by delay rate
        airline_delays = df.groupby('OP_CARRIER')['delayed'].mean().sort_values(ascending=False).head(8) * 100
        fig.add_trace(
            go.Bar(x=airline_delays.index, y=airline_delays.values, name='Airline Delays'),
            row=2, col=1
        )
        
        # Monthly delays
        monthly_delays = df.groupby('month')['delayed'].mean() * 100
        fig.add_trace(
            go.Scatter(x=monthly_delays.index, y=monthly_delays.values,
                      mode='lines+markers', name='Monthly Delays'),
            row=2, col=2
        )
        
        # Weather impact (if available)
        if 'origin_adverse_weather' in df.columns:
            weather_impact = df.groupby('origin_adverse_weather')['delayed'].mean() * 100
            fig.add_trace(
                go.Bar(x=['Normal', 'Adverse'], y=weather_impact.values, name='Weather Impact'),
                row=3, col=1
            )
        
        # Top airports by delay rate
        airport_delays = df.groupby('ORIGIN')['delayed'].mean().sort_values(ascending=False).head(8) * 100
        fig.add_trace(
            go.Bar(x=airport_delays.index, y=airport_delays.values, name='Airport Delays'),
            row=3, col=2
        )
        
        fig.update_layout(height=1200, showlegend=False, title_text="Flight Delay Analysis Dashboard")
        fig.show()

if __name__ == "__main__":
    print("Flight Delay Visualization Module")
    print("Create visualizer with: viz = FlightDelayVisualizer()")