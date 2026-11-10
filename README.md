# 🅿️ Smart Parking Availability & Prediction System

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
The models are trained for the first time (approximately 20–30 seconds). The application can also run without uploading any file by using demo data.

Features

- Live Status: Shows the number of available parking spaces in each lot and predicts occupancy for the next 1, 2, and 3 hours.
- Future Planning: Predicts parking occupancy for any selected date and time, identifies the least crowded time slots, and suggests dynamic parking prices.
- Smart Recommendations: Recommends the best parking lot based on the user's destination and available parking spaces.
- Analytics: Analyzes weekday and weekend patterns, generates occupancy heatmaps, studies the effects of rain, events, and holidays, and performs KMeans-based parking lot clustering.
- Model Report: Compares different machine learning models, including Baseline, Linear Regression, Random Forest, and Gradient Boosting, using a time-based data split.

How to Upload Your Own Data

Go to Sidebar → Upload CSV.

Required columns:

- "timestamp"
- "lot_id"
- "occupied"

Optional columns:

- "capacity"
- "rain"
- "is_holiday"
- "event"
- "lat"
- "lon"

Project Files

- core.py: Contains all the data science logic, including data processing, model training, and predictions.
- app.py: Contains the Streamlit dashboard and user interface.