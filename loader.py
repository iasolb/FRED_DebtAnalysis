from fredapi import Fred
import pandas as pd
import os
from dotenv import load_dotenv


def load_fred_master() -> pd.DataFrame:
    load_dotenv()
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    # ---------------------------------------------------------------------------
    # Series catalog — FRED ID : (friendly_name, native_frequency)
    #
    # Frequency key:  D = daily, W = weekly, M = monthly,
    #                 Q = quarterly, A = annual, SEP = irregular
    #
    # D/W series → resampled via MEAN  (captures full month's behavior)
    # M and lower → resampled via LAST (point-in-time snapshot)
    # All gaps forward-filled after merge.
    # ---------------------------------------------------------------------------

    SERIES = {
        # ── Prices & Inflation ────────────────────────────────────────────────
        "CPIAUCSL": ("Headline_CPI", "M"),
        "CPILFESL": ("Core_CPI", "M"),
        "PCEPI": ("PCE_Price_Index", "M"),
        "PCEPILFE": ("Core_PCE", "M"),
        "CPIUFDSL": ("CPI_Food", "M"),
        "CPIENGSL": ("CPI_Energy", "M"),
        "MICH": ("UMich_Inflation_Expectations", "M"),
        "T5YIE": ("Breakeven_Inflation_5Y", "D"),
        "T10YIE": ("Breakeven_Inflation_10Y", "D"),
        # ── Output & Growth ───────────────────────────────────────────────────
        "GDP": ("Nominal_GDP", "Q"),
        "GDPC1": ("Real_GDP", "Q"),
        "A939RX0Q048SBEA": ("Real_GDP_Per_Capita", "Q"),
        "INDPRO": ("Industrial_Production", "M"),
        "TOTALSA": ("Total_Vehicle_Sales", "M"),
        # ── Labor Market ──────────────────────────────────────────────────────
        "UNRATE": ("Unemployment_Rate", "M"),
        "U6RATE": ("U6_Underemployment", "M"),
        "PAYEMS": ("Nonfarm_Payrolls", "M"),
        "ICSA": ("Initial_Jobless_Claims", "W"),
        "AWHAETP": ("Avg_Weekly_Hours", "M"),
        "CES0500000003": ("Avg_Hourly_Earnings", "M"),
        "JTSJOL": ("JOLTS_Job_Openings", "M"),
        "CIVPART": ("Labor_Force_Participation", "M"),
        # ── Interest Rates & Yields ───────────────────────────────────────────
        "FEDFUNDS": ("FedFunds_Rate", "M"),
        "DFF": ("FedFunds_Daily", "D"),
        "DGS2": ("Treasury_2Y", "D"),
        "DGS10": ("Treasury_10Y", "D"),
        "DGS30": ("Treasury_30Y", "D"),
        "T10Y2Y": ("Yield_Curve_10Y_2Y", "D"),
        "T10Y3M": ("Yield_Curve_10Y_3M", "D"),
        "BAMLH0A0HYM2": ("HY_OAS_Spread", "D"),
        "BAMLC0A0CM": ("IG_OAS_Spread", "D"),
        "MORTGAGE30US": ("Mortgage_Rate_30Y", "W"),
        # ── Money Supply & Credit ─────────────────────────────────────────────
        "M2SL": ("M2_Money_Stock", "M"),
        "TOTRESNS": ("Total_Reserves", "M"),
        "BUSLOANS": ("Commercial_Loans", "M"),
        "TOTALSL": ("Consumer_Credit_Total", "M"),
        "REVOLSL": ("Consumer_Credit_Revolving", "M"),
        # ── Housing ───────────────────────────────────────────────────────────
        "CSUSHPISA": ("Case_Shiller_Home_Price", "M"),
        "HOUST": ("Housing_Starts", "M"),
        "PERMIT": ("Building_Permits", "M"),
        "EXHOSLUSM495S": ("Existing_Home_Sales", "M"),
        "MSACSR": ("Months_Supply_Housing", "M"),
        # ── Consumer & Sentiment ──────────────────────────────────────────────
        "UMCSENT": ("UMich_Consumer_Sentiment", "M"),
        "RSAFS": ("Retail_Sales", "M"),
        "PCE": ("Personal_Consumption_Expenditures", "M"),
        "PSAVERT": ("Personal_Savings_Rate", "M"),
        # ── Trade & Dollar ────────────────────────────────────────────────────
        "BOPGSTB": ("Trade_Balance", "M"),
        "DTWEXBGS": ("Trade_Weighted_USD_Broad", "D"),
        # ── Financial Conditions & Volatility ─────────────────────────────────
        "VIXCLS": ("VIX", "D"),
        "SP500": ("SP500", "D"),
        "NFCI": ("Chicago_Fed_Financial_Conditions", "W"),
        "STLFSI2": ("StL_Fed_Financial_Stress", "W"),
        # ── Fiscal ────────────────────────────────────────────────────────────
        "GFDEBTN": ("Federal_Debt_Total", "Q"),
        "FYFSD": ("Federal_Surplus_Deficit", "A"),
        "MTSDS133FMS": ("Monthly_Treasury_Statement_Deficit", "M"),
        # ── Inequality ────────────────────────────────────────────────────────
        "SIPOVGINIUSA": ("Gini_Index_USA", "A"),
        # ── SEP Projections (median, dot-plot era onward) ─────────────────────
        "FEDTARMD": ("SEP_FedFunds_Median", "SEP"),
        "FEDTARMDLR": ("SEP_FedFunds_Median_LongerRun", "SEP"),
        "GDPC1MD": ("SEP_RealGDP_Median", "SEP"),
        "UNRATEMED": ("SEP_Unemployment_Median", "SEP"),
        "PCECTPIMD": ("SEP_PCE_Inflation_Median", "SEP"),
        "PCECTPILFEMD": ("SEP_CorePCE_Median", "SEP"),
    }

    # Daily series get averaged into weekly buckets (captures intra-week moves).
    # Everything else (W/M/Q/A/SEP) gets resampled via last, then forward-filled.
    _MEAN_FREQS = {"D"}

    # Weekly resampling anchor: W-FRI gives you weeks ending Friday,
    # aligning with most financial data conventions.
    RESAMPLE_RULE = "W-FRI"

    def pull_all(
        series_dict: dict,
        start: str = "1990-01-01",
        freq: str = RESAMPLE_RULE,
    ) -> pd.DataFrame:
        """
        Pull every series from FRED, resample to weekly (W-FRI), forward-fill.

        Daily series  → weekly MEAN  (captures full week's behavior)
        All others    → weekly LAST  (point-in-time, then ffill fills the gaps)

        Returns a single wide DataFrame indexed by week-ending date.
        """
        frames = {}
        failed = []

        for series_id, (name, native_freq) in series_dict.items():
            try:
                s = fred.get_series(series_id, observation_start=start)
                s.name = name

                if native_freq in _MEAN_FREQS:
                    s = s.resample(freq).mean()
                else:
                    s = s.resample(freq).last()

                frames[name] = s
                agg = "mean" if native_freq in _MEAN_FREQS else "last"
                print(f"  ✓ {name:<40s} ({series_id:<18s} {native_freq} → {agg})")
            except Exception as e:
                failed.append((series_id, name, str(e)))
                print(f"  ✗ {name:<40s} ({series_id}) — {e}")

        df = pd.DataFrame(frames)
        df = df.ffill()
        df.index.name = "date"

        if failed:
            print(f"\n⚠  {len(failed)} series failed:")
            for sid, nm, err in failed:
                print(f"    {nm} ({sid}): {err}")

        print(
            f"\nLoaded {len(frames)} series  |  {df.shape[0]} weeks  |  {df.columns.size} columns"
        )
        return df

    df = pull_all(SERIES)

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv("data/fred_master.csv")

    print(f"\nSaved → fred_master.csv")
    return df
