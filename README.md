# Facebook Ads Performance Analysis

## Steps Included

1. **Data Quality & Daily EDA**
   - Defines key terms (spend, transactions, CAC).
   - Identifies and analyzes "leftover lines" (days with transactions but zero spend).
   - Daily aggregation and spend vs. CAC correlation analysis.

2. **Weekly Aggregation**
   - Aggregates daily data into weekly metrics.
   - Visualizes weekly spend and CAC trends over time.
   - Analyzes weekly spend vs. CAC correlation.

3. **Weekly Segmentation**
   - Segments weekly data by campaign type (Retargeting, Prospecting, Blended, ASC, Other).
   - Compares weekly spend and CAC across segments.
   - Provides combined scatter plots for detailed segment comparison.

4. **Retargeting Drilldown**
   - Detailed analysis specifically for retargeting campaigns.
   - Truncates inactive campaigns after 30 days post-last spend to avoid noise.
   - Includes segmented scatter plots and detailed campaign analysis.

5. **Monthly Seasonality**
   - Aggregates and analyzes monthly data to identify seasonal patterns.
   - Year-over-year CAC and ROAS comparisons.
   - Naive budget recommender based on linear modeling.

6. **Outlier Analysis (Daily)**
   - Identifies outliers in daily spend or CAC using the IQR method.
   - Visualizes detected outliers clearly for easy interpretation.

7. **Outlier Days Deep-Dive**
   - Provides detailed breakdowns of outlier days.
   - Highlights segment-level and campaign-level spending anomalies.

8. **Zero-Spend Leftover Deep-Dive**
   - Explores transactions recorded on days without spend.
   - Analyzes day-lag from the last spending date to transaction date.
   - Generates cumulative leftover conversion charts.

## How to Run

1. Ensure the required Python libraries are installed:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn statsmodels altair openpyxl
```

2. Run the Streamlit app:

```bash
streamlit run your_script_name.py
```

Replace `your_script_name.py` with your actual Python script file name.

## Data Requirements

Place the data file (`facebook_historical_performance_eight_sleep.xlsx`) in the same directory as the script.

## Technology Stack

- Python
- Streamlit
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Altair

## Notes

- Adjust analysis thresholds and settings (e.g., outlier detection parameters) directly within the Streamlit interface for dynamic exploration.

