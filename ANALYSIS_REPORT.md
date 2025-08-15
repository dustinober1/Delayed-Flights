# Flight Delay Analysis Report

**Analysis Date**: August 15, 2025  
**Dataset**: Bureau of Transportation Statistics - On-Time Performance (Jan-Mar 2023)  
**Sample Size**: 100,000 flights → 98,174 analyzed flights (after removing cancelled flights)

## Executive Summary

This analysis examines flight delay patterns in commercial aviation using official Bureau of Transportation Statistics data. The study reveals that **21.1% of flights experience delays exceeding 15 minutes**, with significant variations across time periods, airports, airlines, and routes.

## Dataset Overview

- **Total Records**: 100,000 flights
- **Analysis Sample**: 98,174 flights (excluded 1,826 cancelled/missing data)
- **Time Period**: January - March 2023
- **Features**: 110 original columns from BTS dataset
- **Target Variable**: Binary classification (delayed >15 minutes vs on-time)

## Key Findings

### 1. Overall Delay Statistics

| Metric | Value |
|--------|--------|
| Total Flights Analyzed | 98,174 |
| Delayed Flights (>15 min) | 20,697 (21.1%) |
| Average Delay | 6.97 minutes |
| Median Delay | -5.00 minutes |
| Cancellation Rate | 1.6% |

**Key Insight**: Despite 21% of flights being delayed, the median delay is negative, indicating most flights actually arrive early.

### 2. Temporal Patterns

#### Delays by Hour of Departure
Evening flights show significantly higher delay rates:

| Hour | Flights | Delay Rate |
|------|---------|------------|
| 21:00 (9 PM) | 3,350 | 27.7% |
| 19:00 (7 PM) | 5,665 | 27.6% |
| 20:00 (8 PM) | 4,268 | 27.3% |
| 18:00 (6 PM) | 5,942 | 26.6% |
| 17:00 (5 PM) | 5,939 | 26.3% |

#### Delays by Day of Week

| Day | Flights | Delay Rate |
|-----|---------|------------|
| Wednesday | 13,901 | 23.4% |
| Friday | 14,941 | 23.0% |
| Sunday | 14,338 | 21.6% |
| Thursday | 14,800 | 21.3% |
| Monday | 14,566 | 20.1% |
| Saturday | 11,800 | 19.6% |
| Tuesday | 13,828 | 18.3% |

**Key Insight**: Mid-week and end-of-week flights experience higher delays, likely due to increased business travel and weekend leisure traffic.

### 3. Airport Performance

#### Top 10 Airports by Delay Rate (minimum 500 flights)

| Airport | Flights | Delay Rate | Notes |
|---------|---------|------------|-------|
| FLL (Fort Lauderdale) | 1,369 | 29.4% | Leisure destination |
| LAS (Las Vegas) | 2,823 | 27.7% | High-traffic leisure hub |
| MCO (Orlando) | 2,493 | 27.4% | Major tourist destination |
| PBI (Palm Beach) | 504 | 27.4% | Seasonal leisure traffic |
| DEN (Denver) | 4,034 | 26.9% | Weather-sensitive hub |
| HNL (Honolulu) | 866 | 24.4% | Long-haul destination |
| MIA (Miami) | 1,547 | 24.4% | International hub |
| JFK (New York JFK) | 1,989 | 24.2% | Congested major hub |
| BOS (Boston) | 1,982 | 24.0% | Weather-sensitive |
| DFW (Dallas) | 3,799 | 23.5% | Major hub |

**Key Insight**: Leisure destinations and weather-sensitive locations show higher delay rates.

### 4. Airline Performance

#### Airlines Ranked by Delay Rate (minimum 1,000 flights)

| Airline | Code | Flights | Delay Rate | Avg Delay | Performance Tier |
|---------|------|---------|------------|-----------|------------------|
| Frontier Airlines | F9 | 2,495 | 32.0% | 18.76 min | Poor |
| Allegiant Air | G4 | 1,761 | 30.0% | 21.04 min | Poor |
| JetBlue Airways | B6 | 4,286 | 28.0% | 12.11 min | Below Average |
| Spirit Airlines | NK | 3,913 | 28.0% | 13.26 min | Below Average |
| Hawaiian Airlines | HA | 1,150 | 28.0% | 12.52 min | Below Average |
| American Airlines | AA | 13,488 | 23.0% | 10.78 min | Average |
| Alaska Airlines | AS | 3,414 | 22.0% | 5.38 min | Above Average |
| United Airlines | UA | 10,530 | 21.0% | 5.84 min | Above Average |
| Envoy Air | MQ | 3,319 | 20.0% | 5.03 min | Good |
| Southwest Airlines | WN | 20,144 | 20.0% | 4.45 min | Good |
| Delta Air Lines | DL | 13,772 | 19.0% | 5.38 min | Good |
| SkyWest Airlines | OO | 9,477 | 19.0% | 7.76 min | Good |
| Endeavor Air | 9E | 2,956 | 18.0% | 4.56 min | Excellent |
| PSA Airlines | OH | 2,858 | 14.0% | 0.53 min | Excellent |
| Republic Airways | YX | 4,611 | 14.0% | -2.58 min | Excellent |

**Key Insight**: Low-cost carriers generally show higher delay rates, while regional carriers often perform better.

### 5. Distance Analysis

| Distance Category | Flights | Delay Rate | Avg Delay |
|-------------------|---------|------------|-----------|
| Short (<500mi) | 34,507 | 18.8% | 5.26 min |
| Medium (500-1000mi) | 34,964 | 21.0% | 7.32 min |
| Long (1000-1500mi) | 16,577 | 23.7% | 9.52 min |
| Very Long (1500-2000mi) | 6,192 | 23.1% | 7.16 min |
| Ultra Long (>2000mi) | 5,934 | 25.1% | 7.57 min |

**Key Insight**: Longer flights show incrementally higher delay rates, with ultra-long flights having the highest delay probability.

### 6. Route Analysis

#### Top 10 Routes by Delay Rate (minimum 100 flights)

| Route | Flights | Delay Rate | Route Type |
|-------|---------|------------|------------|
| DEN-LAX | 123 | 33.3% | Transcontinental |
| LAS-DEN | 134 | 32.1% | Mountain/Desert |
| ORD-LGA | 170 | 31.2% | Hub to Hub |
| LAX-LAS | 184 | 30.4% | Leisure Route |
| FLL-ATL | 113 | 30.1% | Southeast Corridor |
| EWR-MCO | 107 | 29.9% | Northeast to Leisure |
| LAS-LAX | 175 | 29.7% | West Coast Leisure |
| MCO-ORD | 100 | 29.0% | Leisure to Hub |
| PHX-LAS | 114 | 28.9% | Southwest Corridor |
| MIA-LGA | 111 | 28.8% | Southeast to Northeast |

**Key Insight**: Routes involving leisure destinations and congested hubs show consistently higher delay rates.

### 7. Statistical Correlations

**Strongest correlations with flight delays:**

| Factor | Correlation | Strength |
|--------|-------------|----------|
| Departure Time | 0.118 | Weak Positive |
| Departure Hour | 0.118 | Weak Positive |
| Distance | 0.047 | Very Weak |
| Month | 0.024 | Very Weak |
| Day of Week | 0.013 | Very Weak |

**Key Insight**: Departure timing is the strongest predictor, but correlations are generally weak, suggesting complex multifactor causation.

## Actionable Insights

### For Airlines:
1. **Schedule Optimization**: Consider reducing evening departure slots or building in buffer time
2. **Route Management**: Focus improvement efforts on consistently problematic routes
3. **Hub Operations**: Implement better ground operations at high-delay airports

### For Passengers:
1. **Best Travel Times**: Book morning flights and avoid Wednesday/Friday travel when possible
2. **Airport Selection**: Consider alternative airports to avoid FLL, LAS, MCO during peak times
3. **Airline Choice**: Consider delay history when booking, especially for leisure travel

### For Airport Operations:
1. **Capacity Planning**: Evening hours require enhanced operations management
2. **Weather Preparedness**: Mountain and northern airports need robust weather contingency plans
3. **Ground Services**: Focus on turn-around efficiency during peak evening hours

## Methodology Notes

- **Data Source**: Bureau of Transportation Statistics Official Data
- **Sample Period**: January-March 2023 (representative winter/spring period)
- **Delay Definition**: >15 minutes arrival delay (industry standard)
- **Analysis Scope**: Domestic US flights only
- **Data Quality**: 98.2% completeness after removing cancelled flights

## Technical Details

### Data Processing:
- Raw files: 3 CSV files (~1.8M total records)
- Sampling: Random 100k sample for analysis
- Cleaning: Removed cancelled flights and missing arrival data
- Feature Engineering: Created temporal and categorical variables

### Files Generated:
- `data/processed/airline_sample.csv` - Full processed dataset
- `data/processed/airline_exploration.csv` - Analysis-ready dataset
- `run_exploration.py` - Data processing script
- `explore_data.py` - Analysis script

---

*Report generated automatically from flight delay analysis pipeline*