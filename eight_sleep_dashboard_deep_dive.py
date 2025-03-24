"""
- **Step 1**: Data Quality + Daily EDA  
- **Step 2**: Weekly Aggregation  
- **Step 3**: Weekly Segmentation  
- **Step 4**: Retargeting Drilldown  
- **Step 5**: Monthly Seasonality  
- **Step 6**: Outlier Analysis (Daily)  
- **Step 7**: Outlier Days Deep-Dive  
- **Step 8**: Zero-Spend Leftover Deep-Dive  
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import datetime
import matplotlib.dates as mdates
import altair as alt
import math
from math import erf, sqrt
import matplotlib.ticker as mticker

###########################
# Utility / Helper Funcs
###########################

@st.cache_data
def load_data():
    df = pd.read_excel("facebook_historical_performance_eight_sleep.xlsx", parse_dates=["date"])
    df = df[df["breakdown_platform_northbeam"]=="Facebook Ads"].copy()
    return df

def classify_segment(camp_name: str) -> str:
    low = camp_name.lower()
    if "retarget" in low or "rt-" in low:
        return "Retargeting"
    elif "prospect" in low or "pro-" in low:
        return "Prospecting"
    elif "blended" in low:
        return "Blended"
    elif "asc" in low:
        return "ASC"
    else:
        return "Other"

def fix_cac(spend, tx):
    if spend==0 and tx>0:
        return np.nan
    elif tx>0:
        return spend/tx
    else:
        return np.nan


###########################
# STEP 1: Data Quality + Daily EDA
###########################

def step1_data_quality_eda(df):
    """
    This function:
    1) Explains key terms (spend, tx, leftover lines, CAC).
    2) Summarizes leftover lines and their share of total lines, transactions, revenue. (Rows are assumed to be Leftover Lines when there is no spend attributed to the current date, but the campaign is determined)
    3) Shows bar charts for leftover row %, leftover tx %, leftover rev % by segment.
    4) Displays a daily aggregator and simple daily correlation of spend vs. CAC.
    """

    st.subheader("Step 1: Data Quality + Daily EDA")

    #######################
    # KEY TERMS
    #######################
    st.markdown("""
    **Key Terms**:
    - **Spend**: The budget or cost recorded for that campaign/day.
    - **tx**: Short for "**transactions**.
    - **CAC**: Cost per acquisition = Spend / tx (only valid if tx>0).
    - **Leftover lines**: Rows where *spend=0* but *tx>0*. 
      This often indicates a **lagged** attribution scenario 
      where conversions appear on days with no new spend.
    """)

    #######################
    # 1) Basic Stats
    #######################
    total_lines = len(df)
    tx_lines_df = df[df["transactions"] > 0]
    total_tx_lines = len(tx_lines_df)

    leftover_df = df[(df["spend"] == 0) & (df["transactions"] > 0)]
    leftover_rows = len(leftover_df)

    leftover_conversions = leftover_df["transactions"].sum()
    total_conversions = df["transactions"].sum()

    st.write(f"- **Total rows** in dataset: {total_lines}")
    st.write(f"- Rows with **tx>0**: {total_tx_lines}")
    st.write(f"- **Zero-spend leftover lines**: {leftover_rows}")
    st.write(f"   - {leftover_rows/total_tx_lines*100 if total_tx_lines>0 else 0:.2f}% of all tx>0 rows")
    st.write(f"   - {leftover_rows/total_lines*100 if total_lines>0 else 0:.2f}% of all rows in the dataset")

    leftover_conv_pct = (leftover_conversions / total_conversions * 100) if total_conversions > 0 else 0
    st.write(f"- Leftover lines represent **{leftover_conversions:.3f}** conversions "
             f"({leftover_conv_pct:.2f}% of total).")

    if "rev" in df.columns:
        leftover_revenue = leftover_df["rev"].sum()
        total_revenue = df["rev"].sum()
        leftover_rev_pct = (leftover_revenue / total_revenue * 100) if total_revenue>0 else 0
        st.write(f"- Leftover lines have **${leftover_revenue:,.2f}** revenue "
                 f"({leftover_rev_pct:.2f}% of total).")

    st.markdown("""
    **Interpretation**: 
    - A fairly large fraction of the *transaction-bearing* rows (spend=0, tx>0) 
      but they account for only a few percent of overall conversions/revenue. 
    """)

    st.markdown("---")

    #######################
    # 2) Detailed leftover vs. total breakdown by segment
    #######################

    leftover_df = leftover_df.copy()
    leftover_df["segment"] = leftover_df["campaign_name"].apply(classify_segment)
    leftover_seg = leftover_df.groupby("segment", as_index=False).agg({
        "transactions": "sum", 
        "rev": "sum" if "rev" in leftover_df.columns else np.sum,
        "campaign_name": "count"
    }).rename(columns={"campaign_name": "row_count"})

    df["segment"] = df["campaign_name"].apply(classify_segment)
    total_seg = df.groupby("segment", as_index=False).agg({
        "transactions": "sum", 
        "rev": "sum" if "rev" in df.columns else np.sum,
        "campaign_name": "count"
    }).rename(columns={"campaign_name": "row_count"})

    combined_seg = leftover_seg.merge(
        total_seg, on="segment", how="outer", suffixes=("_leftover", "_total")
    ).fillna(0)

    combined_seg["leftover_row_pct"] = np.where(
        combined_seg["row_count_total"] > 0,
        combined_seg["row_count_leftover"] / combined_seg["row_count_total"] * 100,
        0
    )
    combined_seg["leftover_tx_pct"] = np.where(
        combined_seg["transactions_total"] > 0,
        combined_seg["transactions_leftover"] / combined_seg["transactions_total"] * 100,
        0
    )
    if "rev_leftover" in combined_seg.columns:
        combined_seg["leftover_rev_pct"] = np.where(
            combined_seg["rev_total"] > 0,
            combined_seg["rev_leftover"] / combined_seg["rev_total"] * 100,
            0
        )

    st.markdown("**Detailed leftover vs. total breakdown by segment:**")
    st.dataframe(combined_seg)

    st.write("Below, we visualize **what fraction** of each segment's lines, conversions, and revenue are 'leftover.'")

    # leftover row %
    fig_bar_row, ax_bar_row = plt.subplots(figsize=(5,3))
    ax_bar_row.bar(combined_seg["segment"], combined_seg["leftover_row_pct"], color="tomato")
    ax_bar_row.set_title("Leftover Row % by Segment")
    ax_bar_row.set_ylabel("% leftover rows")
    ax_bar_row.set_xlabel("Segment")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig_bar_row)
    st.markdown("""
    **What this chart shows**: 
    - For each segment, the bar's height is the percent of that segment's *transaction-bearing lines*
      that have spend=0 (leftover lines).
    """)

    # leftover transaction %
    fig_bar_tx, ax_bar_tx = plt.subplots(figsize=(5,3))
    ax_bar_tx.bar(combined_seg["segment"], combined_seg["leftover_tx_pct"], color="green")
    ax_bar_tx.set_title("Leftover Conversions % by Segment")
    ax_bar_tx.set_ylabel("% leftover conversions")
    ax_bar_tx.set_xlabel("Segment")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig_bar_tx)
    st.markdown("""
    **What this chart shows**:
    - The percent of each segment's *total conversions* that come from leftover lines 
      (spend=0, tx>0). 
    """)

    # leftover revenue %
    if "leftover_rev_pct" in combined_seg.columns:
        fig_bar_rev, ax_bar_rev = plt.subplots(figsize=(5,3))
        ax_bar_rev.bar(combined_seg["segment"], combined_seg["leftover_rev_pct"], color="skyblue")
        ax_bar_rev.set_title("Leftover Revenue % by Segment")
        ax_bar_rev.set_ylabel("% leftover revenue")
        ax_bar_rev.set_xlabel("Segment")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig_bar_rev)
        st.markdown("""
        **What this chart shows**:
        - The percent of each segment's *total revenue* that is attributed to leftover lines. 
        """)

    st.markdown("---")

    #######################
    # 3) Daily Aggregator & correlation
    #######################
    daily = df.groupby("date", as_index=False).agg({"spend": "sum", "transactions": "sum"})
    daily["CAC"] = daily.apply(lambda r: fix_cac(r["spend"], r["transactions"]), axis=1)

    # --------------------------------------
    # ADDED LINES FOR DAILY ROAS (if 'rev' exists)
    # --------------------------------------
    if "rev" in df.columns:
        daily_rev = df.groupby("date", as_index=False).agg({"rev": "sum"})
        daily = pd.merge(daily, daily_rev, on="date", how="left")
        daily["ROAS"] = np.where(daily["spend"] > 0, daily["rev"] / daily["spend"], np.nan)
    else:
        daily["ROAS"] = np.nan
    # --------------------------------------

    st.write("**Daily Aggregator** (top 10 rows) after leftover fix => CAC=NaN if spend=0 & tx>0:")
    st.dataframe(daily.head(10))

    fig_line, ax_line = plt.subplots(figsize=(8,4))
    daily_sorted = daily.sort_values("date")
    ax_line.plot(daily_sorted["date"], daily_sorted["spend"], color="blue", label="Spend")
    ax_line.set_ylabel("Spend", color="blue")
    ax_line.set_xlabel("Date")
    ax_line2 = ax_line.twinx()
    ax_line2.plot(daily_sorted["date"], daily_sorted["CAC"], color="orange", label="CAC")
    ax_line2.set_ylabel("CAC", color="orange")
    ax_line.set_title("Daily Spend & CAC Over Time")
    st.pyplot(fig_line)
    st.markdown("""
    **Spend & CAC Over Time**:
    - The line in blue shows how daily spend varies.
    - The orange line shows daily cost per acquisition (CAC). 
      If spend=0 & tx>0, CAC=NaN (excluded).
    """)

    # correlation
    daily_valid = daily_sorted.dropna(subset=["CAC"])
    daily_valid = daily_valid[daily_valid["spend"] > 0]
    if len(daily_valid) > 2:
        corr_val = daily_valid[["spend","CAC"]].corr().iloc[0,1]
        lr = LinearRegression().fit(daily_valid[["spend"]], daily_valid["CAC"])
        slope = lr.coef_[0]
        intercept = lr.intercept_
        r2_lin = r2_score(daily_valid["CAC"], lr.predict(daily_valid[["spend"]]))
        st.write(f"**Daily correlation**(spend,CAC) => {corr_val:.4f}, R^2={r2_lin:.4f}, slope={slope:.6f}, intercept={intercept:.2f}")

        if abs(slope) > 1e-10:
            spend_500 = (500 - intercept) / slope
            if spend_500 > 0:
                st.write(f"For **$500 CAC** => daily spend=~${spend_500:.2f}")
            else:
                st.write("No feasible daily spend for $500 (negative or zero).")
    else:
        st.write("Not enough daily points for correlation after leftover fix.")



############################
# STEP 2: Weekly Aggregation
############################

def step2_weekly_agg(df):
    """
    Goals & Context:
    - We sum daily data by (iso_year, iso_week) to reduce day-level noise.
    - This helps unify leftover lines (spend=0, tx>0) with days that had real spend in the same week.
    - We then convert the ISO year/week to a real date (Monday) so the x-axis is a time axis.
    - Finally, we plot:
        1) A bar + line chart showing weekly spend vs. weekly CAC over time.
        2) A scatter plot of weekly spend vs. weekly CAC, plus a linear fit, 
           to see if there's a direct relationship.
    """

    st.subheader("Step 2: Weekly Aggregation & Overall Correlation")

    ############################################################################
    # 1) Introduction: Why Weekly Aggregation?
    ############################################################################
    st.markdown("""
    **Weekly Aggregation**:
    - After identifying leftover lines in Step 1, we now sum daily data **by week** 
      to see broader trends and possibly smooth out day-to-day fluctuations.
    - We'll compute a **weekly CAC** = total weekly spend / total weekly tx (when tx>0).
    - We then convert `(iso_year, iso_week)` to an actual date (the **Monday** of that ISO week),
      so the x-axis can be a true time axis. This avoids messy label overlap and 
      lets us see a timeline.
    - By looking at **weekly** results, we aim to see if there's a clearer correlation
      between spend and CAC with less noise.
    """)

    # 2) Create iso_year, iso_week
    df2 = df.copy()
    df2["iso_year"] = df2["date"].dt.isocalendar().year
    df2["iso_week"] = df2["date"].dt.isocalendar().week

    # 3) Aggregate daily -> weekly
    weekly = df2.groupby(["iso_year","iso_week"], as_index=False).agg({
        "spend":"sum",
        "transactions":"sum"
    })

    # 4) Calculate weekly CAC
    #    If transactions=0, CAC=NaN
    weekly["CAC"] = np.where(
        weekly["transactions"]>0,
        weekly["spend"]/weekly["transactions"],
        np.nan
    )

    st.markdown("**Weekly Aggregator sample (first 10 rows):**")
    st.dataframe(weekly.head(10))

    ########################################################################
    # 5) Convert iso_year/week -> Monday date, to fix x-axis labeling
    ########################################################################
    def iso_to_monday(yr, wk):
        # Monday = weekday=1
        return datetime.date.fromisocalendar(yr, wk, 1)

    weekly["week_start_date"] = [
        iso_to_monday(r.iso_year, r.iso_week) 
        for r in weekly.itertuples()
    ]

    # Sort by actual date
    weekly_sorted = weekly.sort_values("week_start_date")
    total_weeks = len(weekly_sorted)
    st.write(f"- Found **{total_weeks}** weekly records. (min date = {weekly_sorted['week_start_date'].min()}, max date = {weekly_sorted['week_start_date'].max()})")

    st.markdown("---")

    ########################################################################
    # 6) Bar+Line Chart: Weekly Spend (bar) & Weekly CAC (line)
    ########################################################################
    fig_w, ax_w = plt.subplots(figsize=(9,4))

    x_dates = weekly_sorted["week_start_date"]
    # Bar for spend
    ax_w.bar(x_dates, weekly_sorted["spend"], color="skyblue", label="Weekly Spend")
    ax_w.set_ylabel("Spend (bar)", color="blue")

    # Second axis for CAC
    ax_w2 = ax_w.twinx()
    ax_w2.plot(x_dates, weekly_sorted["CAC"], color="orange", marker="o", label="Weekly CAC")
    ax_w2.set_ylabel("CAC (line)", color="orange")

    # Format date axis to avoid overlap
    ax_w.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_w.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig_w.autofmt_xdate()

    ax_w.set_title("Weekly Spend & CAC Over Time")
    st.pyplot(fig_w)

    st.markdown("""
    **Interpretation**:
    - The x-axis uses the Monday date of each ISO week, so we have a real time axis.
    - Bars (blue) show the total weekly spend, lines (orange) show the average cost/acquisition that week.
    - Expected result: As we spend more in a week, the CAC line would be expected to increase, 
      which might hint at diminishing returns.
    """)

    st.markdown("---")

    ########################################################################
    # 7) Correlation: Weekly Spend vs. Weekly CAC (Scatter)
    ########################################################################
    weekly_valid = weekly_sorted.dropna(subset=["CAC"])
    weekly_valid = weekly_valid[weekly_valid["spend"]>0]
    if len(weekly_valid)<2:
        st.write("Not enough weekly data with spend>0 & transactions>0 to show correlation.")
        return

    fig_scatter, ax_scatter = plt.subplots(figsize=(6,4))
    ax_scatter.scatter(weekly_valid["spend"], weekly_valid["CAC"], color="purple", alpha=0.7)
    ax_scatter.set_xlabel("Weekly Spend")
    ax_scatter.set_ylabel("Weekly CAC")
    ax_scatter.set_title("Scatter: Weekly Spend vs. Weekly CAC")
    ax_scatter.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax_scatter.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    plt.xticks(rotation=45)

    # Linear fit
    lr = LinearRegression().fit(weekly_valid[["spend"]], weekly_valid["CAC"])

    x_range = np.linspace(weekly_valid["spend"].min(), weekly_valid["spend"].max(), 100)
    x_df = pd.DataFrame({"spend": x_range})  

    y_pred = lr.predict(x_df)              

    ax_scatter.plot(x_range, y_pred, color="red", label="Linear Fit")
    ax_scatter.legend()
    st.pyplot(fig_scatter)

    # stats
    corr_val = weekly_valid[["spend","CAC"]].corr().iloc[0,1]
    slope = lr.coef_[0]
    intercept = lr.intercept_
    r2_lin = r2_score(weekly_valid["CAC"], lr.predict(weekly_valid[["spend"]]))
    st.write(f"**Correlation(Weekly Spend, Weekly CAC)** = {corr_val:.4f}, R^2={r2_lin:.4f}")
    st.write(f"Slope={slope:.6f}, Intercept={intercept:.2f}")

    # Solve for $500
    if abs(slope)>1e-10:
        spend_500 = (500 - intercept)/slope
        if spend_500>0:
            st.write(f"For **$500 CAC** => weekly spend ~ ${spend_500:.2f}")
        else:
            st.write("No feasible weekly spend for $500 (negative or zero).")
    else:
        st.write("Slope is ~0 => can't solve for $500 CAC meaningfully.")

    st.markdown("""
**Analysis**:

1. **Scatter Plot**: Each point represents a single week. 
   - **x-axis** = total weekly spend  
   - **y-axis** = weekly CAC (cost per acquisition)  
   The purpose is to let us visually check if spending more in a week generally leads to a lower or higher CAC.

2. **Slope**:
   - A **negative slope** would suggest that as spend increases, CAC *decreases*, implying potential economies of scale or improved efficiency.
   - A **zero or positive slope** indicates no strong sign that higher spend lowers CAC; in fact, a positive slope means higher spend correlates with higher CAC (or random noise).
   - Currently using simple linear regression as data is limited. Some regression that favors diminishing returns with spend increases would likely be favored with more detailed or clean data, though.

3. **R² Value**:
   - R² measures how much of the variation in weekly CAC is explained by weekly spend in this **simple linear** model.
   - A **low R²** means spend alone doesn’t strongly predict CAC at the weekly level (Or the relationship isn’t linear).

4. **Spend for \\$500 CAC**:
   - We take the linear fit (CAC = intercept + slope × spend) and solve for CAC = \\$500.
   - If that solution is **negative or zero**, it implies there’s **no real positive spend** in this model’s logic that would yield a \\$500 CAC. 
   - In other words, the linear model can’t find a feasible weekly spend where we’d see \\$500 cost/acquisition.

5. **Next Steps**:
   - If the slope is near zero or the R² is small, we don’t see a clear “spend => CAC” relationship at the **weekly** level. 
   - In subsequent steps, we’ll explore **segmentation** (retargeting vs. prospecting) or **monthly trends** to refine our understanding. 
   - Sometimes, combining weekly data with additional variables (seasonality, promotions, etc.) or using a more sophisticated model can uncover relationships that aren’t visible in a simple linear scatter.

    """)





############################
# STEP 3: Weekly Segmentation
############################

def step3_segmentation(df):
    """
    1) Classify each row into segments ('Retargeting', 'Prospecting', etc.).
    2) Aggregate daily data -> weekly data per segment + iso_year + iso_week.
    3) Convert iso_year/week -> Monday date for a date-based time axis.
    4) For each segment, produce a bar+line chart of weekly spend & CAC (date-based).
    5) Show a combined color-coded scatter (spend vs. CAC) for all segments together.
    """

    st.subheader("Step 3: Weekly Campaign Segmentation")

    ########################################################################
    # INTRO: Why segment by campaign type?
    ########################################################################
    st.markdown("""
    **Segmentation**:
    - Different campaign types (e.g., Retargeting, Prospecting) may behave differently 
      in terms of spend efficiency.
    - By segmenting, we can see if one type tends to have lower or higher CAC at a given spend level.
    - We'll again use **weekly aggregation** (similar to Step 2), but do it *per segment* 
      and plot each segment's weekly spend & CAC over time.
    - Then we'll create a **combined scatter** for *all* segments, color-coded by segment,
      to quickly spot which segment outperforms at certain spend levels.
    """)

    ########################################################################
    # 1) Prepare data: classify segment, create iso_year & iso_week
    ########################################################################
    df2 = df.copy()
    df2["segment"] = df2["campaign_name"].apply(classify_segment)
    df2["iso_year"] = df2["date"].dt.isocalendar().year
    df2["iso_week"] = df2["date"].dt.isocalendar().week

    # Group by segment + iso-year + iso-week
    weekly_seg = df2.groupby(["segment","iso_year","iso_week"], as_index=False).agg({
        "spend": "sum",
        "transactions": "sum"
    })
    weekly_seg["CAC"] = np.where(
        weekly_seg["transactions"]>0,
        weekly_seg["spend"]/weekly_seg["transactions"],
        np.nan
    )

    st.markdown("**Weekly aggregator** (segment-based) (top 10 rows):")
    st.dataframe(weekly_seg.head(10))

    # Convert (iso_year, iso_week) -> Monday date, for a real time axis
    def iso_to_monday(yr, wk):
        return datetime.date.fromisocalendar(yr, wk, 1)

    weekly_seg["week_start_date"] = [
        iso_to_monday(r.iso_year, r.iso_week) for r in weekly_seg.itertuples()
    ]

    st.write("Segments found:", weekly_seg["segment"].unique().tolist())
    st.markdown("---")

    ########################################################################
    # 2) For each segment: Weekly Bar+Line chart (Spend & CAC)
    ########################################################################
    st.markdown("""
    ### Per-Segment Weekly Charts:
    - We'll show each segment's weekly spend (bar) and CAC (line) over time, 
      using a date-based axis so labels are neat.
    """)

    segments = weekly_seg["segment"].unique()
    for seg in segments:
        seg_data = weekly_seg[weekly_seg["segment"] == seg].copy()
        # filter valid rows
        seg_data = seg_data[seg_data["spend"]>0]
        seg_data.sort_values("week_start_date", inplace=True)
        
        if seg_data.empty:
            st.write(f"Segment '{seg}' => no weekly data with spend>0. Skipping.")
            continue

        st.markdown(f"#### Segment: {seg}")

        fig_seg, ax_seg = plt.subplots(figsize=(9,4))

        x_dates = seg_data["week_start_date"]
        # bar = spend
        ax_seg.bar(x_dates, seg_data["spend"], color="skyblue")
        ax_seg.set_ylabel("Weekly Spend", color="blue")
        ax_seg.set_xlabel("Date (Week Start)")

        # second axis for CAC
        ax_seg2 = ax_seg.twinx()
        ax_seg2.plot(x_dates, seg_data["CAC"], color="orange", marker="o")
        ax_seg2.set_ylabel("Weekly CAC", color="orange")

        # format x-axis as date
        ax_seg.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_seg.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig_seg.autofmt_xdate()

        ax_seg.set_title(f"Weekly Spend & CAC Over Time - Segment = {seg}")
        st.pyplot(fig_seg)

        # optional correlation
        valid_points = seg_data.dropna(subset=["CAC"])
        if len(valid_points) >= 2:
            corr_val = valid_points[["spend","CAC"]].corr().iloc[0,1]
            st.write(f"Correlation(Spend, CAC) for **{seg}** => {corr_val:.4f}")
        else:
            st.write("Not enough non-NaN CAC data to compute correlation here.")

        st.markdown("""
        **Analysis**:
        - Bars = total spend each iso-week, 
        - Orange line = cost/acquisition that week for this segment.
        - Observe if big spend weeks coincide with significantly lower or higher CAC.
        """)
        st.markdown("---")

    ########################################################################
    # 3) Combined Scatter: color-coded by segment
    ########################################################################
    st.markdown("""
    ### Combined Color-Coded Scatter (Weekly Spend vs. CAC)
    - Now we combine *all segments' weekly data* in one chart, 
      plotting each week's (spend, CAC) as a point. 
    - The color shows which segment the point belongs to.
    - This helps quickly see if certain segments cluster at lower CAC 
      for the same spend range, or if one segment can scale up spend 
      without spiking CAC, etc.
    """)

    valid_scatter = weekly_seg.dropna(subset=["CAC"])
    valid_scatter = valid_scatter[valid_scatter["spend"] > 0]
    if valid_scatter.empty:
        st.write("No valid weekly data with spend>0 & CAC => no combined scatter to show.")
        return

    fig_combined, ax_combined = plt.subplots(figsize=(7,5))
    sns.scatterplot(
        data=valid_scatter,
        x="spend",
        y="CAC",
        hue="segment",
        alpha=0.75,
        palette="Set2",  # or your choice
        ax=ax_combined
    )
    ax_combined.set_xlabel("Weekly Spend (USD)")
    ax_combined.set_ylabel("Weekly CAC (USD/acquisition)")
    ax_combined.set_title("Weekly Spend vs. CAC (Color-coded by Segment)")
    ax_combined.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig_combined.tight_layout()
    st.pyplot(fig_combined)

    st.markdown("""
    ### Extended Analysis & Next Steps

    **Observations from segment-level correlations**:

    - **ASC**: Correlation(spend, CAC) ~ 0.1158  
      This suggests a weak positive relationship: as spend increases, CAC might rise slightly, 
      or could be mostly noise since it’s still a low correlation.

    - **Blended**: Correlation ~ 0.0950  
      Very weak positive correlation. No strong indication that higher spend lowers CAC.

    - **Other**: Correlation ~ 0.3285  
      A moderate positive correlation. This could imply as we spend more in 'Other' segment, 
      CAC tends to go up as well. We may need more data or a deeper look (sub-segments exist here).

    - **Prospecting**: Correlation ~ 0.0183  
      Essentially near zero, suggesting no clear link between weekly spend and weekly CAC in 
      this segment’s data. Possibly it’s dominated by other factors (seasonality, campaign changes, etc.).

    - **Retargeting**: Correlation ~ 0.2573  
      Some positive correlation. If we read it literally, 
      spending more on Retargeting could lead to a higher CAC, 
      but we should confirm with a deeper drilldown.

    **Combined Scatter** (Color-coded by segment) insights:

    - If you see that **Retargeting** points cluster in a certain spend range but still achieve 
      moderate or low CAC, that might be good news (a stable, consistent place to invest). 
    - If **Prospecting** points scatter widely with no trend, the correlation might remain small. 
    - **One** or two segments may appear significantly lower on the y-axis (CAC) for the same 
      or higher spend, suggesting better cost-efficiency.

    **Relevance to Our Overall Goal**:

    - We want to find the relationship between **spend** and **CAC**.
    - At the segment level, we see mostly **positive or near-zero correlations**.
    - This could mean we need either:
      1) **More advanced modeling** (accounting for time lags, creative differences, or seasonality), 
      2) **Segmentation** within these segments (e.g., Retargeting might have sub-campaigns 
         that behave differently),
      3) A focus on **monthly** or multi-channel analysis to see if synergy or cross-effects exist.

    **Next Steps**:
    - **Step 4 (Retargeting Drilldown)**: 
      Since Retargeting has a moderate correlation (0.2573), we may want to look deeper 
      at sub-campaigns or different retargeting strategies. 
    - **Step 5 (Monthly Seasonality)**: 
      The overall data might be strongly seasonal (holidays, promotions). 
      Aggregating monthly and controlling for seasonal spikes might reveal clearer patterns. 
    - **Blended** or **Other** segments**: 
      If these lumps different campaign types together, 
      we could refine them further or treat them carefully in future modeling.
    """)





############################
# STEP 4: Retargeting Drilldown (Weekly)
############################

def step4_retargeting_drilldown(df):
    """
    Step 4: Retargeting Drilldown with:
      - Filter for retargeting campaigns (keyword-based).
      - Weekly aggregator per campaign.
      - Skip campaigns below a minimum spend threshold (e.g. $500).
      - Truncate each chart ~30 days after last real spend, so we don't see a long zero line.
      - Y-axes pinned at bottom=0 to avoid negative ticks.
      - Combined scatter color-coded by campaign.
      - Extended analysis & disclaimers about zero intervals.

    """

    st.subheader("Step 4: Retargeting Drilldown")

    #########################################################################
    # 1. Explanation / rationale
    #########################################################################
    st.markdown("""
    **Why Retargeting Drilldown?**
    - Retargeting likely differs from Prospecting or other segments in how spend scales with CAC.
    - We'll identify campaigns containing "retarget" or "rt-" in their name,
      aggregate to weekly, and focus on those above a certain total spend threshold (e.g. $500).
    - We'll also **truncate** each campaign's chart ~1 month after its last real spend 
      so we don't have huge zero lines for many months.
    - Finally, a combined scatter shows which retargeting sub-campaign might be more or less efficient.
    """)

    #########################################################################
    # 2. Filter for retargeting data & aggregator
    #########################################################################
    df2 = df.copy()
    # Mark retargeting
    df2["is_ret"] = df2["campaign_name"].str.lower().apply(
        lambda x: ("retarget" in x) or ("rt-" in x)
    )
    df_ret = df2[df2["is_ret"]].copy()
    if df_ret.empty:
        st.write("No retargeting data found. Possibly no campaigns have 'retarget' or 'rt-' in the name.")
        return

    df_ret["iso_year"] = df_ret["date"].dt.isocalendar().year
    df_ret["iso_week"] = df_ret["date"].dt.isocalendar().week

    weekly_camp = df_ret.groupby(["campaign_name","iso_year","iso_week"], as_index=False).agg({
        "spend":"sum",
        "transactions":"sum"
    })
    weekly_camp["CAC"] = np.where(
        weekly_camp["transactions"]>0,
        weekly_camp["spend"]/weekly_camp["transactions"],
        np.nan
    )

    # Convert iso_year/week -> real date (Monday)
    def iso_to_monday(yr, wk):
        return datetime.date.fromisocalendar(yr, wk, 1)
    weekly_camp["week_start_date"] = [
        iso_to_monday(r.iso_year, r.iso_week)
        for r in weekly_camp.itertuples()
    ]

    st.markdown("**Retargeting Weekly Aggregator** (sample):")
    st.dataframe(weekly_camp.head(10))

    #########################################################################
    # 3. Filter out minimal-spend campaigns & sort by spend desc
    #########################################################################
    MIN_SPEND_THRESHOLD = 500.0  # adjust as needed
    camp_spend = weekly_camp.groupby("campaign_name", as_index=False)["spend"].sum()
    camp_spend = camp_spend[camp_spend["spend"] >= MIN_SPEND_THRESHOLD]
    camp_spend.sort_values("spend", ascending=False, inplace=True)

    if camp_spend.empty:
        st.write(f"No retargeting campaigns with >= ${MIN_SPEND_THRESHOLD} total spend.")
        return

    st.markdown(f"**Retargeting campaigns with >= ${MIN_SPEND_THRESHOLD} total spend**:")
    st.dataframe(camp_spend)

    # pick top 5
    top5 = camp_spend.head(5)["campaign_name"].tolist()

    #########################################################################
    # 4. Weekly Spend & CAC Over Time (Truncated)
    #########################################################################
    st.markdown(f"### Weekly Spend & CAC Over Time (Top 5 Retargeting Campaigns above ${MIN_SPEND_THRESHOLD})")

    # how many days after last spend do we keep
    TRUNCATE_DAYS = 30

    for c in top5:
        df_c = weekly_camp[weekly_camp["campaign_name"] == c].copy()
        if df_c.empty:
            st.write(f"Campaign '{c}' => no data => skip.")
            continue

        # sort by date
        df_c.sort_values("week_start_date", inplace=True)

        # find last date with spend>0
        df_c_nonzero = df_c[df_c["spend"]>0]
        if df_c_nonzero.empty:
            st.write(f"Campaign '{c}' => all zero spend, skipping chart.")
            continue
        last_spend_date = df_c_nonzero["week_start_date"].max()
        # define cutoff date
        cutoff_date = pd.to_datetime(last_spend_date) + pd.Timedelta(days=TRUNCATE_DAYS)

        # also handle the date column as datetime for comparison
        df_c["wsd_datetime"] = pd.to_datetime(df_c["week_start_date"])
        df_c = df_c[df_c["wsd_datetime"] <= cutoff_date]

        st.markdown(f"**Campaign: {c}**")
        st.write(f"_Truncated chart after {cutoff_date.date()} (30 days post last spend date) to avoid long zero lines._")

        # plot
        fig, ax = plt.subplots(figsize=(9,4))
        x_dates = df_c["wsd_datetime"]
        ax.bar(x_dates, df_c["spend"], color="skyblue")
        ax.set_ylabel("Weekly Spend (bar)", color="blue")
        ax.set_xlabel("Date (Week Start)")

        ax2 = ax.twinx()
        ax2.plot(x_dates, df_c["CAC"], color="orange", marker="o")
        ax2.set_ylabel("Weekly CAC (line)", color="orange")

        # date formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()

        # avoid negative y-lims
        ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)

        ax.set_title(f"Retargeting Campaign: {c} — Weekly Spend & CAC")
        st.pyplot(fig)

        # correlation & model
        valid_pts = df_c.dropna(subset=["CAC"])
        valid_pts = valid_pts[valid_pts["spend"]>0]
        if len(valid_pts) < 2:
            st.write("Not enough points (spend>0 & CAC) to compute correlation.")
        else:
            corr_val = valid_pts[["spend","CAC"]].corr().iloc[0,1]
            lr = LinearRegression().fit(valid_pts[["spend"]], valid_pts["CAC"])
            slope = lr.coef_[0]
            intercept = lr.intercept_
            r2_lin = r2_score(valid_pts["CAC"], lr.predict(valid_pts[["spend"]]))
            st.write(f"Correlation(Spend, CAC) => {corr_val:.4f}, R^2={r2_lin:.4f}")
            if abs(slope) > 1e-10:
                spend_500 = (500 - intercept)/slope
                if spend_500>0:
                    st.write(f"For **$500 CAC** => weekly spend ~ ${spend_500:.2f}")
                else:
                    st.write("No feasible weekly spend for $500 (negative or zero).")
            else:
                st.write("Slope near zero => can't solve for $500 CAC meaningfully.")

        st.markdown("""
        **Analysis**:
        - The chart is truncated 30 days after the final nonzero spend, 
          so we don't see a long line of zero data.
        - Check if correlation is positive (CAC rises with more spend) or near zero/negative.
        - If there's a feasible solution for $500 CAC, that's a naive linear guess 
          for what weekly spend might produce that cost/acquisition.
        """)

        st.markdown("---")

    #########################################################################
    # 5. Combined Scatter for top retargeting campaigns
    #########################################################################
    st.markdown("""
    ### Combined Scatter: Retargeting Weekly Spend vs. CAC (Color by Campaign)
    - Now we combine all the truncated data from these top retargeting campaigns.
      Each point represents a valid (spend>0, CAC) weekly observation.
    """)

    # gather truncated data
    valid_scatter = weekly_camp[weekly_camp["campaign_name"].isin(top5)].copy()
    valid_scatter["wsd_datetime"] = pd.to_datetime(valid_scatter["week_start_date"])

    # for each row, define if it's after cutoff
    def get_cutoff_date_for_cam(cam):
        # last spend date
        sub = df_c_nonzero[df_c_nonzero["campaign_name"]==cam]
        if sub.empty:
            return None
        return pd.to_datetime(sub["week_start_date"].max()) + pd.Timedelta(days=TRUNCATE_DAYS)

    big_df_list = []
    for c in top5:
        chunk = valid_scatter[valid_scatter["campaign_name"]==c].copy()
        chunk_nonzero = chunk[chunk["spend"]>0]
        if chunk_nonzero.empty:
            continue
        last_spend_date = chunk_nonzero["week_start_date"].max()
        cutoff_date = pd.to_datetime(last_spend_date) + pd.Timedelta(days=TRUNCATE_DAYS)
        chunk["wsd_datetime"] = pd.to_datetime(chunk["week_start_date"])
        chunk = chunk[chunk["wsd_datetime"] <= cutoff_date]
        big_df_list.append(chunk)
    final_scatter = pd.concat(big_df_list, ignore_index=True)

    final_scatter = final_scatter.dropna(subset=["CAC"])
    final_scatter = final_scatter[final_scatter["spend"]>0]

    if final_scatter.empty:
        st.write("No valid retargeting data left after truncation => no scatter.")
        return

    fig_scat, ax_scat = plt.subplots(figsize=(7,5))
    sns.scatterplot(
        data=final_scatter,
        x="spend",
        y="CAC",
        hue="campaign_name",
        palette="tab10",
        alpha=0.7,
        ax=ax_scat
    )
    ax_scat.set_xlabel("Weekly Spend")
    ax_scat.set_ylabel("Weekly CAC")
    ax_scat.set_title("Retargeting Weekly Spend vs. CAC (Color by Campaign)")

    ax_scat.set_ylim(bottom=0)
    ax_scat.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Campaign")
    fig_scat.tight_layout()
    st.pyplot(fig_scat)

    #########################################################################
    # 6. Extended Analysis & Next Steps
    #########################################################################
    st.markdown("""
    ### Extended Analysis & Next Steps

    **1. Zero / paused intervals**:
    - We truncated each campaign's data 30 days after its last nonzero spend, 
      so the charts won't show a long line of zeros in later months.
    - If a campaign truly resumed months later, that data wouldn't appear in the truncated chart 
      (though you could adjust the code to handle multiple active intervals).

    **2. Correlation Observations**:
    - If a campaign has a moderate or high positive correlation, that suggests weekly CAC rises with spend.
    - A near-zero or negative correlation might mean we can scale without big cost inflation, 
      or it might reflect random noise if data is scarce.

    **3. $500 CAC Solutions**:
    - Only valid if the slope is negative or small enough that we find a positive spend level for $500. 
      If it's negative, more spend might reduce CAC (rare, but can happen at times).
    - If it's large positive, the implied spend might be infeasible or negative.

    **4. Next Steps**:
    - Step 5: Check monthly seasonality.
    - Possibly refine retargeting audiences or creative if correlation is strongly positive.
    - Compare retargeting vs. prospecting in a multi-channel view or 
      use advanced modeling for synergy & lag effects.

    In summary, retargeting sub-campaigns that pass the spend threshold can be individually assessed. 
    Where correlation is moderate or negative, there's potential to scale further 
    if $500 CAC is your target. Where correlation is strongly positive or no feasible solution, 
    further sub-segmentation or creative changes might help.
    """)



############################
# STEP 5: Monthly Seasonality (All Data)
############################

def label_month(m: int) -> str:
    """Categorize month => 'Holiday/Nov', 'Summer (May–Sep)', or 'Other Months'."""
    if m == 11:
        return "Holiday/Nov"
    elif 5 <= m <= 9:
        return "Summer (May–Sep)"
    else:
        return "Other Months"

def step5_monthly_seasonality(df):
    """
    - The majority of our analysis happens in Step 5. Seasonality for Summer and "Other" and "Promotional" months seem to be very important.
    - Ideally after determining the seasonality trends on the monthly, we would go back do weekly or daily to build a more accurate shorter term model
    - For the purposes of this exercise, we don't drill down much more than the monthly seasonality as the goal is to find a monthly spend for ~$500 CAC, 
    - Even though weekly projections summed up would likely be more accurate for the monthly CAC goal.
    """

    st.subheader("Step 5: Monthly Seasonality")

    #####################################################
    # 1) Monthly Aggregation
    #####################################################
    df2 = df.copy()
    df2["year"] = df2["date"].dt.year
    df2["month"] = df2["date"].dt.month

    monthly = df2.groupby(["year","month"], as_index=False).agg({
        "spend": "sum",
        "transactions": "sum"
    })

    # If 'rev' column exists, sum it
    if "rev" in df2.columns:
        rev_agg = df2.groupby(["year","month"], as_index=False)["rev"].sum()
        monthly = monthly.merge(rev_agg, on=["year","month"], how="left")
    else:
        monthly["rev"] = np.nan

    # Compute CAC & ROAS
    monthly["CAC"] = np.where(
        monthly["transactions"] > 0,
        monthly["spend"] / monthly["transactions"],
        np.nan
    )
    monthly["ROAS"] = np.where(
        monthly["spend"] > 0,
        monthly["rev"] / monthly["spend"],
        np.nan
    )

    # Sort & create "month_start_date"
    monthly.sort_values(["year","month"], inplace=True)
    monthly["month_start_date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-"
        + monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    st.markdown("**Monthly Aggregator (sample):**")
    st.dataframe(monthly.head(10))

    # Plot monthly spend & CAC over time
    st.markdown("### Monthly Spend & CAC Over Time")
    fig_cac, ax_cac = plt.subplots(figsize=(9,4))
    x_vals = monthly["month_start_date"]
    ax_cac.bar(x_vals, monthly["spend"], color="skyblue", label="Spend")
    ax_cac.set_ylabel("Monthly Spend (USD)", color="blue")
    ax_cac2 = ax_cac.twinx()
    ax_cac2.plot(x_vals, monthly["CAC"], color="orange", marker="o", label="CAC")
    ax_cac2.set_ylabel("Monthly CAC", color="orange")
    ax_cac.set_title("Monthly Spend (bar) & CAC (line)")
    ax_cac.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig_cac.autofmt_xdate()
    st.pyplot(fig_cac)

    # Plot monthly spend & ROAS over time
    st.markdown("### Monthly Spend & ROAS Over Time")
    fig_roas, ax_roas = plt.subplots(figsize=(9,4))
    ax_roas.bar(x_vals, monthly["spend"], color="skyblue")
    ax_roas.set_ylabel("Monthly Spend (USD)", color="blue")
    ax_roas2 = ax_roas.twinx()
    ax_roas2.plot(x_vals, monthly["ROAS"], color="green", marker="o")
    ax_roas2.set_ylabel("Monthly ROAS", color="green")
    ax_roas.set_title("Monthly Spend (bar) & ROAS (line)")
    ax_roas.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig_roas.autofmt_xdate()
    st.pyplot(fig_roas)

    st.markdown("---")

    #####################################################
    # 2) Year-over-Year Overlap (CAC & ROAS)
    #####################################################
    st.markdown("## Year-over-Year Overlap (CAC & ROAS)")

    yoy_cac = monthly.pivot(index="month", columns="year", values="CAC").sort_index()
    yoy_roas = monthly.pivot(index="month", columns="year", values="ROAS").sort_index()

    st.markdown("### CAC YOY Overlap")
    if yoy_cac.shape[1] < 2:
        st.write("Not enough distinct years for CAC YOY overlap.")
    else:
        fig_yoy_cac, ax_yoy_cac = plt.subplots(figsize=(6,4))
        yoy_cac.plot(marker="o", ax=ax_yoy_cac)
        ax_yoy_cac.set_xlabel("Month (1=Jan, 12=Dec)")
        ax_yoy_cac.set_ylabel("CAC")
        ax_yoy_cac.set_title("CAC Year-Over-Year by Month")
        st.pyplot(fig_yoy_cac)

    st.markdown("### ROAS YOY Overlap")
    if yoy_roas.shape[1] < 2:
        st.write("Not enough distinct years for ROAS YOY overlap.")
    else:
        fig_yoy_roas, ax_yoy_roas = plt.subplots(figsize=(6,4))
        yoy_roas.plot(marker="o", ax=ax_yoy_roas)
        ax_yoy_roas.set_xlabel("Month (1=Jan, 12=Dec)")
        ax_yoy_roas.set_ylabel("ROAS")
        ax_yoy_roas.set_title("ROAS Year-Over-Year by Month")
        st.pyplot(fig_yoy_roas)

    st.markdown("---")

    #####################################################
    # 3) Segment-Level Monthly Aggregator => Stacked Bar
    #####################################################
    st.markdown("## Segment-Level Monthly Aggregator")

    def classify_segment_local(campaign_name: str)->str:
        lw = campaign_name.lower()
        if "retarget" in lw or "rt-" in lw:
            return "Retargeting"
        elif "prospect" in lw or "pro-" in lw:
            return "Prospecting"
        elif "blended" in lw:
            return "Blended"
        elif "asc" in lw:
            return "ASC"
        else:
            return "Other"

    df2["segment"] = df2["campaign_name"].apply(classify_segment_local)
    df2["month_start_date"] = pd.to_datetime(
        df2["year"].astype(str) + "-"
        + df2["month"].astype(str).str.zfill(2) + "-01"
    )

    monthly_seg = df2.groupby(["segment","year","month","month_start_date"], as_index=False).agg({
        "spend":"sum","transactions":"sum"
    })
    if "rev" in df2.columns:
        seg_rev = df2.groupby(["segment","year","month","month_start_date"], as_index=False)["rev"].sum()
        monthly_seg = monthly_seg.merge(seg_rev, on=["segment","year","month","month_start_date"], how="left")
    else:
        monthly_seg["rev"] = np.nan

    monthly_seg["CAC"] = np.where(
        monthly_seg["transactions"]>0,
        monthly_seg["spend"]/monthly_seg["transactions"],
        np.nan
    )
    monthly_seg["ROAS"] = np.where(
        monthly_seg["spend"]>0,
        monthly_seg["rev"]/monthly_seg["spend"],
        np.nan
    )
    monthly_seg.sort_values(["year","month"], inplace=True)

    pivot_seg_spend = monthly_seg.pivot_table(
        index="month_start_date", columns="segment", values="spend", aggfunc="sum"
    ).fillna(0)

    if not pivot_seg_spend.empty:
        fig_stack, ax_stack = plt.subplots(figsize=(9,4))
        pivot_seg_spend.plot(kind="bar", stacked=True, ax=ax_stack, colormap="tab20", width=0.8)
        ax_stack.set_title("Monthly Spend by Segment (Stacked)")
        ax_stack.set_xlabel("Month Start")
        ax_stack.set_ylabel("Spend (USD)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_stack)

    st.markdown("---")

    #####################################################
    # 4) Scatter: Spend vs. CAC
    #####################################################
    st.markdown("## Scatter: Spend vs. CAC (Color by Season), with Regression Lines")

    monthly_valid_cac = monthly.dropna(subset=["CAC"]).copy()
    monthly_valid_cac = monthly_valid_cac[monthly_valid_cac["spend"]>0]
    monthly_valid_cac["season_label"] = monthly_valid_cac["month"].apply(label_month)

    if len(monthly_valid_cac) < 2:
        st.write("Not enough data for Altair scatter + regression lines.")
    else:
        base_cac = alt.Chart(monthly_valid_cac)
        points_cac = base_cac.mark_circle(size=80).encode(
            x=alt.X("spend:Q", axis=alt.Axis(title="Spend (USD)", format="~s"), scale=alt.Scale(zero=False)),
            y=alt.Y("CAC:Q", axis=alt.Axis(title="CAC"), scale=alt.Scale(zero=False)),
            color="season_label:N",
            tooltip=["year","month","spend","CAC"]
        )
        # separate regression lines by season_label
        lines_cac = base_cac.transform_regression("spend","CAC", groupby=["season_label"]).mark_line().encode(
            x="spend:Q", y="CAC:Q", color="season_label:N"
        )
        st.altair_chart((points_cac + lines_cac).interactive(), use_container_width=True)

    st.markdown("---")

    #####################################################
    # 5) Distribution Approach: 4 spend levels (±200k)
    #    × (Summer vs. Other), excluding November
    #####################################################
    st.markdown("## Distribution Approach: 4 Target Spend Levels (±$200k) × (Summer vs. Other), Excluding November")

    # Exclude November
    dist_df = monthly_valid_cac[monthly_valid_cac["month"] != 11].copy()

    # Define "summer vs other" (5..9 => Summer, else => Other)
    def summer_or_other(m):
        return "Summer" if (5 <= m <= 9) else "Other"

    dist_df["summer_cat"] = dist_df["month"].apply(summer_or_other)

    # 4 target spend levels, ±200k
    spend_targets = [300000, 500000, 700000]
    SPEND_TOL = 200000

    # We'll do a 4x2 subplot: row=target, col=(Summer, Other)
    fig_dist, axes_dist = plt.subplots(3, 2, figsize=(14,16))
    # Increase spacing
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    for i, target in enumerate(spend_targets):
        for j, cat in enumerate(["Summer","Other"]):
            ax = axes_dist[i, j]
            low_ = target - SPEND_TOL
            high_ = target + SPEND_TOL
            sub = dist_df[
                (dist_df["summer_cat"] == cat) &
                (dist_df["spend"] >= low_) &
                (dist_df["spend"] <= high_)
            ]
            if len(sub) < 2:
                ax.text(0.5, 0.5, "No/Limited Data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"${target//1000}k ±200k, {cat}")
                continue

            # KDE of CAC
            sns.kdeplot(data=sub, x="CAC", fill=True, alpha=0.3, ax=ax)

            # mean ±1σ
            mean_ = sub["CAC"].mean()
            std_  = sub["CAC"].std()
            ax.axvline(mean_, color="red", linestyle="--", label="mean")
            ax.axvline(mean_ + std_, color="orange", linestyle=":")
            ax.axvline(mean_ - std_, color="orange", linestyle=":")

            ax.set_title(f"Spend ~ {target//1000}k ±200k, {cat}")
            ax.set_xlabel("CAC")
            ax.set_ylabel("Density")
            ax.legend()

    fig_dist.suptitle("CAC Distributions by 4 Spend Targets (±200k) × (Summer vs. Other), Excluding November")
    st.pyplot(fig_dist)

    st.markdown("""
    **Interpretation**:
    - Each row = one target monthly spend level (300k, 500k, 700k), 
      allowing ±$200k around that number.
    - Larger figure size and increased spacing to avoid overlap.
    - Red dashed line = mean CAC, orange dotted lines = ±1σ.
    """)

    st.markdown("---")

    #####################################################
    # 6) Naive Budget Recommender (CAC ~ spend)
    #####################################################
    st.markdown("## Naive Budget Recommender (Linear Model: CAC ~ spend)")

    # Reuse monthly_valid_cac, which has .month, .season_label
    def subset_for_label(df_local, lbl):
        if lbl=="All months":
            return df_local
        else:
            return df_local[df_local["season_label"]==lbl]

    user_subset = st.selectbox(
        "Choose a season subset for the naive linear model:",
        ["All months","Summer (May–Sep)","Holiday/Nov","Other Months"]
    )
    subdf = subset_for_label(monthly_valid_cac, user_subset)
    subdf = subdf.dropna(subset=["CAC"])
    subdf = subdf[subdf["spend"]>0]

    if len(subdf) < 2:
        st.write(f"No data for subset={user_subset} => can't do a linear regression.")
    else:
        X = subdf[["spend"]]
        y = subdf["CAC"]
        lr = LinearRegression().fit(X,y)
        slope_ = lr.coef_[0]
        intercept_ = lr.intercept_
        r2_ = r2_score(y, lr.predict(X))

        st.write(f"**Linear Model**:  CAC = {intercept_:.2f} + {slope_:.6f} × spend,  R^2={r2_:.4f}")

        approach = st.radio("Pick approach:", ["Spend for desired CAC","CAC for given Spend"])

        if approach == "Spend for desired CAC":
            desired_cac = st.number_input("Desired CAC (USD)", 0.0, 500.0, step=50.0)
            if abs(slope_) < 1e-10:
                st.write("Slope ~ 0 => can't solve for spend.")
            else:
                spend_needed = (desired_cac - intercept_)/slope_
                if spend_needed>0:
                    st.write(f"For CAC=${desired_cac:.0f}, recommended spend ~ ${spend_needed:,.0f}")
                else:
                    st.write("No positive solution => can't achieve that CAC with this linear model.")
            
            # Probability approach with statsmodels
            if st.button("Check Probability for that CAC?"):
                if len(subdf) < 3:
                    st.write("Need >=3 data points for a more robust statsmodels approach.")
                else:
                    model = smf.ols("CAC ~ spend", data=subdf).fit()
                    user_spend_prob = st.number_input("At what monthly spend check Probability(CAC<desired)?", 0.0, 300000.0, step=50000.0)
                    pred_res = model.get_prediction(pd.DataFrame({"spend":[user_spend_prob]}))
                    df_pred = pred_res.summary_frame(alpha=0.05)
                    pred_mean = df_pred["mean"].iloc[0]
                    obs_ci_lower = df_pred["obs_ci_lower"].iloc[0]
                    obs_ci_upper = df_pred["obs_ci_upper"].iloc[0]
                    approx_std = (obs_ci_upper - obs_ci_lower)/(2.0*1.96)
                    if approx_std>0:
                        z_value = (desired_cac - pred_mean)/approx_std
                        prob = 0.5*(1 + erf(z_value/sqrt(2)))
                        st.write(f"Predicted mean CAC ~ {pred_mean:.2f}, 95% interval ~ [{obs_ci_lower:.2f}, {obs_ci_upper:.2f}]")
                        st.write(f"Probability(CAC < {desired_cac:.0f}) ~ {prob*100:.1f}%")
                    else:
                        st.write("Could not compute a valid stdev => probability approach not feasible.")

        else:
            # approach == "CAC for given Spend"
            user_spend_ = st.number_input("Monthly spend (USD)", 0.0, 300000.0, step=50000.0)
            pred_cac = slope_ * user_spend_ + intercept_
            st.write(f"For spend=${user_spend_:,.0f}, predicted CAC ~ ${pred_cac:.2f}")

            if st.button("Compute Probability of hitting a target?"):
                target_cac = st.number_input("Target CAC?", 0.0, 500.0, step=50.0)
                if len(subdf) < 3:
                    st.write("Need >=3 data points for statsmodels approach.")
                else:
                    model = smf.ols("CAC ~ spend", data=subdf).fit()
                    pred_res = model.get_prediction(pd.DataFrame({"spend":[user_spend_]}))
                    df_pred = pred_res.summary_frame(alpha=0.05)
                    pred_mean = df_pred["mean"].iloc[0]
                    obs_ci_lower = df_pred["obs_ci_lower"].iloc[0]
                    obs_ci_upper = df_pred["obs_ci_upper"].iloc[0]
                    approx_std = (obs_ci_upper - obs_ci_lower)/(2.0*1.96)
                    if approx_std>0:
                        z_value = (target_cac - pred_mean)/approx_std
                        prob = 0.5*(1 + erf(z_value/sqrt(2)))
                        st.write(f"Predicted mean CAC={pred_mean:.2f}, 95% interval~[{obs_ci_lower:.2f}, {obs_ci_upper:.2f}]")
                        st.write(f"Probability(CAC < {target_cac:.2f}) ~ {prob*100:.1f}%")
                    else:
                        st.write("No valid stdev => probability approach not feasible.")

    st.markdown("""
    **Note**:

    - Using ±\\$200k around each spend target yields narrower bins than ±\\$200k 
      but might still produce 'No Data' if usage is low.
    """)

    st.markdown("---")

    st.markdown("### Extended Analysis\n")

    st.markdown("""
    - We aggregated monthly data, made time-series for CAC/ROAS, 
      showed YOY overlaps, did a segment-level stacked bar, 
      created a Spend–CAC scatter with linear regression, 
      and created a distribution approach (±$200k, 4 targets, 
      Summer vs. Other (excl. Nov)) in a bigger subplot layout.

    - We determined seasonality is extremely important and summer / other months 
      should be treated separately. CAC seems to decrease at the same time ROAS increases (April), 
      despite increasing spend.

    - We've determined **\\$500 CAC** is possible in summer months around **\\$882,224** monthly spend. 
      This might be skewed by the May 2024 data, but we would still suggest more aggressive spending 
      than we've seen in previous summer months. June and July of 2024 appear to have underspent 
      at first glance if our goal was to remain around \\$500 CAC, but we would have to dive deeper 
      into those months' performance to say for certain.

    - We've also determined there is no reliable way to achieve a \\$500 CAC on winter months. 
      For winter months, using our simple OLS & standard deviation model, 
      we have a **37.8%** chance at achieving CAC < \\$500.
    """)




############################
# STEP 6: Outlier Analysis (Daily)
############################

def step6_outlier_analysis(df):
    """
      1. Explains why we look for outliers in daily data (promo spikes or data errors).
      2. Lets user pick which metric to detect outliers on: 'Daily Spend' or 'Daily CAC'.
      3. Uses an IQR-based approach (factor commonly 1.5) to detect outliers.
      4. Marks & visualizes outlier days on a time-series plot.
      5. Provides interpretive notes, but does NOT recalculate correlation without outliers.
    """

    st.subheader("Step 6: Outlier Analysis (Daily) - IQR Method")

    st.markdown("""
    **Why Outlier Analysis?**
    - Large daily spikes in spend (or extremely high/low CAC) might be:
      1) **Real** special promo days or sudden changes in ad strategy,
      2) **Data anomalies** or pipeline errors.
    - These outliers can heavily affect correlation or linear models. 
      If they're valid (like big promotions), we keep them (but interpret carefully).
      If they're errors, we might remove or winsorize them in a subsequent QA process.
    """)

    # 1) Aggregate daily
    daily = df.groupby("date", as_index=False).agg({"spend":"sum","transactions":"sum"})
    daily["CAC"] = daily.apply(lambda r: fix_cac(r["spend"], r["transactions"]), axis=1)

    st.markdown("**Daily aggregator (first 5 rows):**")
    st.dataframe(daily.head(5))

    # Let user pick which metric to detect outliers:
    metric = st.selectbox("Select metric to detect outliers on:", ["spend", "CAC"])

    # If user picks "CAC", we only use rows with a valid CAC
    if metric == "CAC":
        detect_df = daily.dropna(subset=["CAC"]).copy()
        st.write(f"Rows with valid CAC: {len(detect_df)} / {len(daily)} total.")
        if detect_df.empty:
            st.write("No valid CAC rows => can't do outlier detection on CAC.")
            return
    else:
        detect_df = daily.copy()

    # 2) IQR approach, factor slider
    factor = st.slider("IQR factor (commonly 1.5)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

    Q1 = detect_df[metric].quantile(0.25)
    Q3 = detect_df[metric].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + factor * IQR
    lower_bound = Q1 - factor * IQR

    # For CAC, we might avoid negative lower bounds
    if metric == "CAC":
        lower_bound = max(0, lower_bound)

    # Identify outliers
    outlier_mask = (detect_df[metric] > upper_bound) | (detect_df[metric] < lower_bound)
    outliers = detect_df[outlier_mask].copy()

    st.write(f"Outlier detection on **{metric}** => lower < {lower_bound:.2f}, upper > {upper_bound:.2f}")
    st.write(f"Outlier rows count: {len(outliers)}")

    if len(outliers) > 0:
        st.markdown("**Sample outlier rows:**")
        st.dataframe(outliers.head(10))

    st.markdown("---")

    # 3) Plot daily time series with outliers in red
    st.markdown(f"### Daily Time Series of {metric.title()} with Outliers Marked")
    daily_sorted = daily.sort_values("date")

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(daily_sorted["date"], daily_sorted[metric], label=metric, color="blue")

    # We only have outliers from detect_df; if user picked CAC, detect_df might be a subset
    outlier_dates = outliers["date"].unique() if len(outliers)>0 else []
    outlier_daily = daily_sorted[daily_sorted["date"].isin(outlier_dates)]

    ax.scatter(outlier_daily["date"], outlier_daily[metric], color="red", label="Outlier Days")

    ax.set_xlabel("Date")
    ax.set_ylabel(metric.title())
    ax.set_title(f"Daily {metric.title()} Time Series (IQR-based Outliers in Red)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    st.pyplot(fig)

    st.markdown(f"""
    **Interpretation**:
    - Red points are the days flagged as outliers based on the chosen IQR factor of **{factor}**.
    - Large positive outliers could be big promos or random data anomalies.
    - When increasing IQR factor for spend, we notice outlier spend days tend to happen in May or November.
    - When increasing IQR factor for CAC, we see outlier CAC days tend to happen in the winter.
    """)



############################
# STEP 7: Outlier Days Deep Dive
############################

def step7_outlier_deepdive(df):
    st.subheader("Step 7: Outlier Days Deep-Dive (Enhanced)")

    # Rebuild daily aggregator from Step 6
    daily = df.groupby("date", as_index=False).agg({"spend":"sum","transactions":"sum"})
    daily["CAC"] = daily.apply(lambda r: fix_cac(r["spend"], r["transactions"]), axis=1)

    # We'll assume we already have an 'outlier_mask' from Step 6 in the session, 
    # but let's for demo do IQR on spend again. 
    Q1 = daily["spend"].quantile(0.25)
    Q3 = daily["spend"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5*IQR
    outlier_days = daily[daily["spend"]>upper_bound].copy()

    if outlier_days.empty:
        st.write("No outlier days found with this approach => skipping deep-dive.")
        return

    # Merge leftover lines info
    leftover_daily = (
        df[df["spend"]==0]
        .groupby("date", as_index=False)["transactions"]
        .sum()
        .rename(columns={"transactions":"leftover_tx"})
    )
    outlier_days = outlier_days.merge(leftover_daily, on="date", how="left")
    outlier_days["leftover_tx"] = outlier_days["leftover_tx"].fillna(0)

    # Merge # segments
    # first ensure we have a 'segment' column
    if "segment" not in df.columns:
        df["segment"] = df["campaign_name"].apply(classify_segment)

    seg_count = df.groupby("date")["segment"].nunique().reset_index(name="num_segments")
    outlier_days = outlier_days.merge(seg_count, on="date", how="left")

    # maybe compare to average daily spend
    mean_spend = daily["spend"].mean()
    outlier_days["mult_of_avg_spend"] = outlier_days["spend"] / mean_spend

    st.write("**All Outlier Days** (with leftover tx, #segments, multiplier of avg spend):")
    st.dataframe(outlier_days[["date","spend","transactions","CAC","leftover_tx","num_segments","mult_of_avg_spend"]])

    st.markdown("""
    **Observations**:
    - leftover_tx: how many conversions came on zero-spend lines that day.
    - num_segments: how many distinct segments contributed spend that day (maybe 1 big retargeting campaign or many).
    - mult_of_avg_spend: if it's e.g. 3.2 => that day spent 3.2× the average daily spend across the entire dataset.
    """)

    # Let user pick a day
    chosen_day = st.selectbox(
        "Pick an outlier date to see breakdown by segment/campaign:",
        outlier_days["date"].astype(str).tolist()
    )

    day_obj = pd.to_datetime(chosen_day)
    df_day = df[df["date"]==day_obj].copy()

    st.markdown(f"### Outlier Day: {chosen_day}")

    # Segment breakdown
    df_day["segment"] = df_day["campaign_name"].apply(classify_segment)
    seg_g = df_day.groupby("segment", as_index=False).agg({
        "spend":"sum","transactions":"sum"
    })
    seg_g["CAC"] = seg_g.apply(lambda r: fix_cac(r["spend"], r["transactions"]), axis=1)
    seg_g.sort_values("spend", ascending=False, inplace=True)

    st.write("**Segment breakdown** for chosen outlier day:")
    st.dataframe(seg_g)


    # Bar chart: segment spend
    fig_seg, ax_seg = plt.subplots(figsize=(6,4))
    sns.barplot(data=seg_g, x="segment", y="spend", ax=ax_seg, color="steelblue")
    ax_seg.set_title(f"Segment Spend on {chosen_day}")
    ax_seg.set_xlabel("Segment")
    ax_seg.set_ylabel("Spend")
    ax_seg.tick_params(axis='x', rotation=30)
    st.pyplot(fig_seg)

    # Bar chart: segment transactions or CAC
    choice_metric = st.selectbox("Choose second bar chart metric:", ["transactions","CAC"])
    fig_seg2, ax_seg2 = plt.subplots(figsize=(6,4))
    sns.barplot(data=seg_g, x="segment", y=choice_metric, ax=ax_seg2, color="darkorange")
    ax_seg2.set_title(f"Segment {choice_metric.capitalize()} on {chosen_day}")
    ax_seg2.set_xlabel("Segment")
    ax_seg2.set_ylabel(choice_metric.capitalize())
    ax_seg2.tick_params(axis='x', rotation=30)
    st.pyplot(fig_seg2)

    # Check leftover lines specifically
    leftover_lines = df_day[(df_day["spend"]==0)&(df_day["transactions"]>0)]
    if not leftover_lines.empty:
        st.markdown("**Zero-spend leftover lines (this outlier day):**")
        st.dataframe(leftover_lines)

    # Top campaigns
    camp_g = df_day.groupby("campaign_name", as_index=False).agg({
        "spend":"sum","transactions":"sum"
    })
    camp_g["CAC"] = camp_g.apply(lambda r: fix_cac(r["spend"], r["transactions"]), axis=1)
    camp_g.sort_values("spend", ascending=False, inplace=True)

    st.write("**Campaign-level breakdown** (top 10) for the chosen day:")
    st.dataframe(camp_g.head(10))

    # Quick bar chart
    fig_camp, ax_camp = plt.subplots(figsize=(6,4))
    top_c = camp_g.head(10)
    sns.barplot(data=top_c, x="spend", y="campaign_name", ax=ax_camp, color="purple")
    ax_camp.set_title(f"Top 10 Campaigns by Spend on {chosen_day}")
    ax_camp.set_xlabel("Spend")
    ax_camp.set_ylabel("Campaign Name")
    st.pyplot(fig_camp)

    st.markdown("""
    **Interpretation**:
    - Not really a lot additional to learn from this deep dive.
    """)




############################
# STEP 8: Zero-Spend Leftover Deep-Dive
############################

def step8_zero_leftover_deepdive(df):
    """
    Step 8: Zero-Spend Leftover Deep-Dive (Enhanced)
    
    - Focus on leftover lines: (spend=0 & transactions>0).
    - Clamp negative day-lags to 0 (if any) to avoid confusion on the chart.
    - Bin day-lags (0–3, 4–7, 8–14, 15–30, 30+).
    - Display a cumulative leftover curve to see how quickly post-view conversions accumulate.
    - Show segment/campaign-level stats (median day-lag, total leftover conversions).
    - Optional campaign filter to view leftover lines for a specific campaign.
    - Provide interpretive insights on what a large or small day-lag might mean.
    """


    st.subheader("Step 8: Zero-Spend Leftover Deep-Dive (Enhanced)")

    # 1) Identify leftover lines
    leftover = df[(df["spend"] == 0) & (df["transactions"] > 0)].copy()
    if leftover.empty:
        st.write("No leftover lines => none to deep-dive.")
        return

    total_lines = len(leftover)
    st.write(f"**Total leftover lines**: {total_lines}")

    # Ensure we have a 'segment' column
    if "segment" not in leftover.columns:
        leftover["segment"] = leftover["campaign_name"].apply(classify_segment)

    #-----------------------------------------------
    # 2) Calculate day-lag from last nonzero spend
    #-----------------------------------------------
    # Build a map of campaign -> last spend date
    df_spend = df[df["spend"]>0]
    last_spend_map = (
        df_spend.groupby("campaign_name")["date"].max()
        .to_dict()
    )
    # Add to leftover
    leftover["last_nonzero_spend"] = leftover["campaign_name"].map(last_spend_map)

    # days_since_spend
    leftover["days_since_spend"] = leftover.apply(
        lambda r: (r["date"] - r["last_nonzero_spend"]).days 
                  if pd.notnull(r["last_nonzero_spend"]) else np.nan,
        axis=1
    )

    # CLAMP negative day-lag to 0
    leftover["days_since_spend"] = np.where(
        leftover["days_since_spend"] < 0,
        0,
        leftover["days_since_spend"]
    ).astype(float)

    # Summarize leftover conversions
    total_conversions = leftover["transactions"].sum()
    st.write(f"**Total leftover conversions** = {total_conversions:.2f}")

    #-----------------------------------------------
    # 3) Bin day-lags
    #-----------------------------------------------
    def bin_day_lag(d):
        if pd.isna(d):
            return "No last spend found"
        if d < 4:
            return "0–3 days"
        elif d < 8:
            return "4–7 days"
        elif d < 15:
            return "8–14 days"
        elif d < 31:
            return "15–30 days"
        else:
            return "30+ days"

    leftover["day_lag_bin"] = leftover["days_since_spend"].apply(bin_day_lag)

    lag_bins = leftover.groupby("day_lag_bin", as_index=False)["transactions"].sum()
    # Sort bins in a custom order
    order_map = {
        "0–3 days":1, 
        "4–7 days":2, 
        "8–14 days":3, 
        "15–30 days":4, 
        "30+ days":5,
        "No last spend found":6
    }
    lag_bins["sort_order"] = lag_bins["day_lag_bin"].map(order_map)
    lag_bins.sort_values("sort_order", inplace=True)

    st.markdown("**Leftover lines grouped by day-lag bins:**")
    st.dataframe(lag_bins[["day_lag_bin","transactions"]])

    fig_bin, ax_bin = plt.subplots(figsize=(6,4))
    ax_bin.bar(lag_bins["day_lag_bin"], lag_bins["transactions"], color="purple")
    ax_bin.set_title("Leftover Conversions by Day-Lag Bin")
    ax_bin.set_xlabel("Day Lag Bin")
    ax_bin.set_ylabel("Sum of transactions")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig_bin)

    st.markdown("""
    **Interpretation**:
    - This shows how many leftover conversions fall into each day-lag range.
    - If most leftover lines appear in 0–3 days, it might be typical short-lag post-view.
    - If a large portion is 30+ days, that could reflect a very long attribution window or suspicious data.
    """)

    #-----------------------------------------------
    # 4) Cumulative leftover curve
    #-----------------------------------------------
    leftover_lag = leftover.dropna(subset=["days_since_spend"]).copy()
    leftover_lag.sort_values("days_since_spend", inplace=True)
    leftover_lag["cumulative_tx"] = leftover_lag["transactions"].cumsum()
    leftover_lag["pct_of_leftover"] = leftover_lag["cumulative_tx"] / total_conversions * 100

    fig_cum, ax_cum = plt.subplots(figsize=(7,4))
    ax_cum.plot(leftover_lag["days_since_spend"], leftover_lag["pct_of_leftover"], color="teal")
    ax_cum.set_title("Cumulative % of leftover conversions vs. day-lag")
    ax_cum.set_xlabel("Days since last spend (clamped at 0 if negative)")
    ax_cum.set_ylabel("Cumulative % of leftover conversions")
    st.pyplot(fig_cum)

    st.markdown("""
    **Cumulative leftover chart**:
    - X-axis: day-lag from last spend (0 if negative).
    - Y-axis: cumulative % of leftover conversions.
    - If 80% happen by day 5, that suggests a short post-view window for most leftover lines.
    """)

    #-----------------------------------------------
    # 5) Segment/campaign-level stats
    #-----------------------------------------------
    st.markdown("### Segment/ Campaign-Level Stats on Leftover Lines")

    # Segment-level
    seg_lag = leftover.groupby("segment", as_index=False).agg({
        "days_since_spend":"median",
        "transactions":"sum"
    })
    seg_lag.rename(columns={"days_since_spend":"median_day_lag"}, inplace=True)
    seg_lag.sort_values("transactions", ascending=False, inplace=True)

    st.markdown("**Median day-lag & leftover conversions by segment:**")
    st.dataframe(seg_lag)
    st.markdown("""
    Some segments might have a higher median day-lag, implying they keep getting conversions
    well after the last spend date.
    """)

    # Campaign-level
    camp_lag = leftover.groupby("campaign_name", as_index=False).agg({
        "days_since_spend":"median",
        "transactions":"sum"
    })
    camp_lag.rename(columns={"days_since_spend":"median_day_lag"}, inplace=True)
    camp_lag.sort_values("transactions", ascending=False, inplace=True)

    st.markdown("**Top campaigns by leftover transactions (with median day-lag):**")
    st.dataframe(camp_lag.head(20))

    #-----------------------------------------------
    # 6) (Optional) Campaign filter
    #-----------------------------------------------
    all_camps = camp_lag["campaign_name"].unique().tolist()
    chosen_camp = st.selectbox("Pick a campaign for leftover detail:", ["(None)"] + all_camps)
    if chosen_camp != "(None)":
        cdf = leftover[leftover["campaign_name"] == chosen_camp].copy()
        st.markdown(f"**Leftover lines** for campaign = {chosen_camp}, total {len(cdf)}")
        st.dataframe(cdf.head(30))

    #-----------------------------------------------
    # Final Decision Points
    #-----------------------------------------------
    st.markdown("""
    ### Decision Points
    - If day-lag is extremely large for a campaign, you might question the validity of 
      that post-view attribution or confirm your ad platform's extended windows.
    - For this assignment, this analysis isn't very valuable, but it is important to understand
      attribution to campaigns and if there might be data quality issues.
    """)




############################
# MAIN
############################

def main():
    st.title("Eight Sleep: Take Home Assignment (FB & CAC Relationship)")

    df = load_data()

    steps = [
        "Step 1: Data Quality + Daily EDA",
        "Step 2: Weekly Aggregation",
        "Step 3: Weekly Segmentation",
        "Step 4: Retargeting Drilldown",
        "Step 5: Monthly Seasonality",
        "Step 6: Outlier Analysis (Daily)",
        "Step 7: Outlier Days Deep-Dive",
        "Step 8: Zero-Spend Leftover Deep-Dive"
    ]
    choice = st.sidebar.selectbox("Choose a step", steps)

    if choice == steps[0]:
        step1_data_quality_eda(df)
    elif choice == steps[1]:
        step2_weekly_agg(df)
    elif choice == steps[2]:
        step3_segmentation(df)
    elif choice == steps[3]:
        step4_retargeting_drilldown(df)
    elif choice == steps[4]:
        step5_monthly_seasonality(df)
    elif choice == steps[5]:
        step6_outlier_analysis(df)
    elif choice == steps[6]:
        step7_outlier_deepdive(df)
    elif choice == steps[7]:
        step8_zero_leftover_deepdive(df)

    st.write("---")
    st.write("End of Step. Use the sidebar to switch steps.")

if __name__=="__main__":
    main()
