"""
Macro Scoring Module
====================

Standalone module that takes the weekly FRED master DataFrame (from loader.py)
and attaches scored columns useful as independent variables in policy reaction
function models.

Three categories of scores:

    1. DERIVED SERIES — raw transforms of the underlying data that many of the
       scores depend on (YoY inflation, real rates, etc.).  These are
       attached first and are useful in their own right.

    2. REGIME & ANOMALY FLAGS — binary/categorical tags for weeks where
       observed conditions contradict standard macro theory or cross
       well-known thresholds.

    3. CONTINUOUS SCORES — signed, continuous measures of tension,
       conflict, or imbalance across the dual mandate, fiscal-monetary
       interaction, and expectations anchoring.

    4. LEAD/LAG PREDICTIVE SIGNALS — forward-shifted columns that encode
       the empirical lead structure in the data (M2→CPI, claims→UE, etc.)
       for use as simulation inputs.

Usage:
    from loader import load_fred_master
    from macro_scores import score

    df = load_fred_master()
    scored = score(df)

All scored columns are returned alongside the original data.
Nothing is dropped; nothing is mutated in place.

Compatible with ResearchHandler — the returned DataFrame can be passed
directly to ResearchHandler() as the source, and individual scored
columns can be registered as independents, controls, or used with
Simulation.from_spec().
"""

import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — all magic numbers in one place for easy calibration
# ═══════════════════════════════════════════════════════════════════════════

# Lookback windows (in weeks)
_W52 = 52  # 1 year
_W26 = 26  # 6 months
_W13 = 13  # 3 months (≈ quarterly)
_W78 = 78  # 18 months

# Inflation target
_PI_STAR = 2.0  # percent, Fed's stated target

# NAIRU / natural rate of unemployment (time-varying in reality,
# but a fixed anchor is standard in Taylor-type rules;
# CBO's recent estimates center around 4.0–4.4%)
_U_STAR = 4.0

# Sahm rule threshold
_SAHM_THRESHOLD = 0.50

# Taylor rule coefficients
_TAYLOR_ALPHA = 0.5  # weight on inflation gap
_TAYLOR_BETA = 0.5  # weight on unemployment gap (Okun-style)

# Deanchoring thresholds (in percentage points)
_DEANCHOR_MILD = 1.0
_DEANCHOR_SEVERE = 2.0

# M2 growth anomaly bounds (historical normal ≈ 4–6% YoY)
_M2_NORMAL_LOW = 2.0
_M2_NORMAL_HIGH = 8.0


# ═══════════════════════════════════════════════════════════════════════════
# 1. DERIVED SERIES
# ═══════════════════════════════════════════════════════════════════════════


def _attach_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core year-over-year rates, real rates, and ratios that the scoring
    functions depend on.  All are useful as standalone regressors too.
    """
    # ── Year-over-year inflation rates ────────────────────────────────
    df["CPI_YoY"] = df["Headline_CPI"].pct_change(_W52) * 100
    df["Core_CPI_YoY"] = df["Core_CPI"].pct_change(_W52) * 100
    df["Core_PCE_YoY"] = df["Core_PCE"].pct_change(_W52) * 100
    df["PCE_YoY"] = df["PCE_Price_Index"].pct_change(_W52) * 100
    df["CPI_Food_YoY"] = df["CPI_Food"].pct_change(_W52) * 100
    df["CPI_Energy_YoY"] = df["CPI_Energy"].pct_change(_W52) * 100

    # ── Inflation momentum (3-month annualized vs 12-month) ──────────
    # If 3m >> 12m, inflation is reaccelerating; if 3m << 12m, decelerating.
    df["Core_PCE_3m_ann"] = df["Core_PCE"].pct_change(_W13) * (52 / 13) * 100
    df["Inflation_Momentum"] = df["Core_PCE_3m_ann"] - df["Core_PCE_YoY"]

    # ── Money supply growth ───────────────────────────────────────────
    df["M2_YoY"] = df["M2_Money_Stock"].pct_change(_W52) * 100
    df["M2_6m_ann"] = df["M2_Money_Stock"].pct_change(_W26) * (52 / 26) * 100

    # ── Real fed funds rate (ex-post) ─────────────────────────────────
    df["Real_FFR_CPI"] = df["FedFunds_Rate"] - df["CPI_YoY"]
    df["Real_FFR_PCE"] = df["FedFunds_Rate"] - df["Core_PCE_YoY"]

    # ── Real 10-year yield ────────────────────────────────────────────
    df["Real_10Y_CPI"] = df["Treasury_10Y"] - df["CPI_YoY"]

    # ── Labor market derived ──────────────────────────────────────────
    # Nonfarm payroll monthly change (approx: 4-week diff on weekly data)
    df["Payroll_Change_4w"] = df["Nonfarm_Payrolls"].diff(4)

    # Sahm rule components
    df["UE_3m_avg"] = df["Unemployment_Rate"].rolling(_W13, min_periods=8).mean()
    df["UE_12m_min"] = df["Unemployment_Rate"].rolling(_W52, min_periods=26).min()
    df["Sahm_Indicator"] = df["UE_3m_avg"] - df["UE_12m_min"]

    # JOLTS / unemployed (labor market tightness)
    # Rough labor force ≈ 160M; unemployed ≈ UE_rate * LF
    _labor_force = 160_000  # thousands
    df["Unemployed_Approx"] = df["Unemployment_Rate"] / 100 * _labor_force
    df["JOLTS_UE_Ratio"] = df["JOLTS_Job_Openings"] / df["Unemployed_Approx"]

    # ── Fiscal derived ────────────────────────────────────────────────
    # Debt / GDP (approximate — quarterly GDP annualized)
    # GFDEBTN is in millions, GDP is in billions (SAAR — already annualized)
    df["Debt_to_GDP"] = (df["Federal_Debt_Total"] / 1000) / df["Nominal_GDP"] * 100

    # MTSDS133FMS is in millions, GDP is in billions (SAAR)
    df["Deficit_GDP_Pct"] = (
        (df["Monthly_Treasury_Statement_Deficit"] * 12 / 1000) / df["Nominal_GDP"] * 100
    )

    # ── SEP-based derived ─────────────────────────────────────────────
    # Forward-fill SEP projections to make them available every week
    df["SEP_LR_FFR"] = df["SEP_FedFunds_Median_LongerRun"].ffill()
    df["SEP_r_star"] = df["SEP_LR_FFR"] - _PI_STAR

    # Policy stance relative to the SEP longer-run neutral
    df["Policy_Stance_SEP"] = df["FedFunds_Rate"] - df["SEP_LR_FFR"]

    # Term premium proxy: 10Y minus expected short rate path (use SEP median)
    df["SEP_FFR_Med"] = df["SEP_FedFunds_Median"].ffill()
    df["Term_Premium_Proxy"] = df["Treasury_10Y"] - df["SEP_FFR_Med"]

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. REGIME & ANOMALY FLAGS
# ═══════════════════════════════════════════════════════════════════════════


def _attach_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary (0/1) and categorical flags for theoretically anomalous or
    regime-relevant conditions.
    """

    # ── Yield curve inversion flags ───────────────────────────────────
    # Classic recession signal: 10Y-2Y < 0
    df["Flag_Curve_Inverted_10Y2Y"] = (df["Yield_Curve_10Y_2Y"] < 0).astype(int)
    df["Flag_Curve_Inverted_10Y3M"] = (df["Yield_Curve_10Y_3M"] < 0).astype(int)

    # ── Sahm rule triggered ───────────────────────────────────────────
    df["Flag_Sahm_Triggered"] = (df["Sahm_Indicator"] >= _SAHM_THRESHOLD).astype(int)

    # ── Taylor rule deviation flag ────────────────────────────────────
    # Flag weeks where actual FFR is >100bp below the Taylor-prescribed rate
    # (i.e. policy is "too loose" relative to the rule).  Uses the PCE-based
    # Taylor rule computed in the continuous scores section — so this flag
    # is computed later in _attach_continuous and back-filled here as a
    # placeholder.  We set it properly in _attach_continuous.
    df["Flag_Taylor_Below_100bp"] = 0  # placeholder
    df["Flag_Taylor_Above_100bp"] = 0  # placeholder

    # ── FCI contradiction: loose financial conditions during tightening ─
    # Chicago Fed NFCI < 0 means looser than average.  If the Fed is
    # explicitly restrictive (FFR > SEP neutral) but FCI is still loose,
    # monetary policy isn't transmitting normally.
    df["Flag_FCI_Contradiction"] = (
        (df["Chicago_Fed_Financial_Conditions"] < -0.2)
        & (df["Policy_Stance_SEP"] > 0.5)
    ).astype(int)

    # ── Inflation expectations deanchoring ────────────────────────────
    # UMich 1-year expectations diverge from trailing realized CPI
    df["Expect_Gap_UMich"] = df["UMich_Inflation_Expectations"] - df["CPI_YoY"]
    df["Flag_Deanchor_Mild"] = (df["Expect_Gap_UMich"].abs() > _DEANCHOR_MILD).astype(
        int
    )
    df["Flag_Deanchor_Severe"] = (
        df["Expect_Gap_UMich"].abs() > _DEANCHOR_SEVERE
    ).astype(int)

    # Direction of deanchoring: positive = expectations running hot
    df["Flag_Deanchor_Hot"] = (df["Expect_Gap_UMich"] > _DEANCHOR_MILD).astype(int)
    df["Flag_Deanchor_Cold"] = (df["Expect_Gap_UMich"] < -_DEANCHOR_MILD).astype(int)

    # ── M2 growth anomaly ─────────────────────────────────────────────
    df["Flag_M2_Excess"] = (df["M2_YoY"] > _M2_NORMAL_HIGH).astype(int)
    df["Flag_M2_Contraction"] = (df["M2_YoY"] < _M2_NORMAL_LOW).astype(int)

    # ── Labor market signals contradicting each other ─────────────────
    # Falling unemployment BUT falling JOLTS/UE ratio — the labor market
    # is cooling structurally even though the headline looks fine
    df["JOLTS_UE_Ratio_chg"] = df["JOLTS_UE_Ratio"].diff(_W13)
    df["UE_chg_13w"] = df["Unemployment_Rate"].diff(_W13)

    df["Flag_Labor_Divergence"] = (
        (df["UE_chg_13w"] > 0.2) & (df["JOLTS_UE_Ratio_chg"] < -0.1)
        | (df["UE_chg_13w"] < -0.2) & (df["JOLTS_UE_Ratio_chg"] > 0.1)
    ).astype(int)

    # ── Negative real rate during above-target inflation ──────────────
    df["Flag_Neg_Real_Rate_HighInflation"] = (
        (df["Real_FFR_PCE"] < 0) & (df["Core_PCE_YoY"] > _PI_STAR)
    ).astype(int)

    # ── Credit spread stress ──────────────────────────────────────────
    # HY spread above 5% historically signals stress
    df["Flag_HY_Stress"] = (df["HY_OAS_Spread"] > 5.0).astype(int)

    # ── Fiscal dominance warning ──────────────────────────────────────
    # Rising debt/GDP + rising interest costs + Fed cutting.
    # This is a rolling directional check.
    df["Debt_GDP_chg_26w"] = df["Debt_to_GDP"].diff(_W26)
    df["FFR_chg_26w"] = df["FedFunds_Rate"].diff(_W26)
    df["Flag_Fiscal_Dominance_Risk"] = (
        (df["Debt_GDP_chg_26w"] > 0)
        & (df["FFR_chg_26w"] < -0.25)
        & (df["Core_PCE_YoY"] > _PI_STAR)
    ).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONTINUOUS SCORES
# ═══════════════════════════════════════════════════════════════════════════


def _attach_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signed, continuous measures of policy tension, mandate conflict,
    and macro imbalance.  All are designed so that:

        > 0  →  one direction of pressure (typically inflationary / hawkish)
        < 0  →  other direction (typically recessionary / dovish)
        ≈ 0  →  balanced / benign

    This makes them natural regressors in reaction function models where
    the dependent variable is FFR changes or forward-rate moves.
    """

    # ── Taylor Rule Gap ───────────────────────────────────────────────
    # Classic Taylor (1993): r_t = r* + π_t + α(π_t - π*) + β(u* - u_t)
    #
    # Uses SEP-implied r* when available, else the HLW estimate.
    # Positive gap = policy is too loose (Taylor says raise).
    # Negative gap = policy is too tight (Taylor says cut).
    r_star = df["SEP_r_star"].combine_first(df["HLW_r_star"])
    pi = df["Core_PCE_YoY"]
    u_gap = _U_STAR - df["Unemployment_Rate"]  # positive when UE below NAIRU

    df["Taylor_Prescribed"] = (
        r_star + pi + _TAYLOR_ALPHA * (pi - _PI_STAR) + _TAYLOR_BETA * u_gap
    )
    df["Taylor_Gap"] = df["Taylor_Prescribed"] - df["FedFunds_Rate"]
    # Positive = Taylor says policy is too loose (should be higher)

    # Now back-fill the Taylor flags from section 2
    df["Flag_Taylor_Below_100bp"] = (df["Taylor_Gap"] > 1.0).astype(int)
    df["Flag_Taylor_Above_100bp"] = (df["Taylor_Gap"] < -1.0).astype(int)

    # ── Dual Mandate Tension Index ────────────────────────────────────
    # Measures the degree to which the two mandates are pulling in
    # opposite directions.
    #
    #   inflation_pressure = Core PCE YoY - target  (+ means too hot)
    #   employment_pressure = UE rate - NAIRU        (+ means too weak)
    #
    # When both are positive → stagflationary tension (mandates conflict)
    # When both are negative → overheating (both say tighten — aligned)
    # When signs differ → mandates are aligned on direction
    #
    # Score = inflation_pressure × employment_pressure
    #   > 0 → mandates conflict (stagflation or goldilocks overheating)
    #   < 0 → mandates agree on direction
    #
    # Magnitude captures severity.
    inflation_pressure = df["Core_PCE_YoY"] - _PI_STAR
    employment_pressure = df["Unemployment_Rate"] - _U_STAR

    df["Inflation_Pressure"] = inflation_pressure
    df["Employment_Pressure"] = employment_pressure

    df["Mandate_Tension"] = inflation_pressure * employment_pressure
    # Positive = stagflationary tension OR goldilocks overrun
    # Negative = mandates aligned

    # Signed version that distinguishes stagflation from goldilocks:
    # Stagflation: inflation hot AND UE high → both positive → product > 0
    # Goldilocks overrun: inflation cold AND UE low → both negative → product > 0
    # We want to separate these, so also provide a categorical label
    def _classify_regime(row):
        ip = row["Inflation_Pressure"]
        ep = row["Employment_Pressure"]
        if pd.isna(ip) or pd.isna(ep):
            return np.nan
        if ip > 0.3 and ep > 0.3:
            return 3  # stagflationary
        elif ip < -0.3 and ep < -0.3:
            return -3  # overheating (goldilocks overshoot)
        elif ip > 0.3 and ep < -0.3:
            return 1  # hot inflation, tight labor → tighten clearly
        elif ip < -0.3 and ep > 0.3:
            return -1  # cold inflation, slack labor → ease clearly
        else:
            return 0  # near target on both

    df["Mandate_Regime"] = df.apply(_classify_regime, axis=1)

    # ── Fiscal-Monetary Conflict Score ────────────────────────────────
    # Captures the degree to which fiscal policy is working against
    # monetary policy.
    #
    # When the Fed is restrictive (FFR > neutral) but the government
    # is running large deficits, fiscal expansion offsets monetary
    # tightening — and vice versa.
    #
    # Score = policy_stance × deficit_as_pct_GDP (inverted sign on deficit
    # since deficit is negative in the data convention)
    #
    # Positive = fiscal and monetary pulling in same direction (aligned)
    # Negative = fiscal offsetting monetary (conflict)
    df["Fiscal_Monetary_Conflict"] = df["Policy_Stance_SEP"] * df["Deficit_GDP_Pct"]
    # When stance > 0 (tight) and deficit < 0 (big deficit), product < 0 → conflict
    # When stance < 0 (loose) and deficit < 0 (big deficit), product > 0 → aligned (both stimulative)

    # ── Expectations Anchoring Score ──────────────────────────────────
    # Composite: average absolute deviation of inflation expectations
    # from target across multiple horizons.
    # Low = well-anchored.  High = drifting.
    umich_dev = (df["UMich_Inflation_Expectations"] - _PI_STAR).abs()
    be5y_dev = (df["Breakeven_Inflation_5Y"] - _PI_STAR).abs()
    be10y_dev = (df["Breakeven_Inflation_10Y"] - _PI_STAR).abs()

    # Average available measures (handles NaN gracefully)
    df["Expect_Anchor_Score"] = pd.concat(
        [umich_dev, be5y_dev, be10y_dev], axis=1
    ).mean(axis=1)

    # Signed version: positive = expectations running hot vs target
    umich_signed = df["UMich_Inflation_Expectations"] - _PI_STAR
    be5y_signed = df["Breakeven_Inflation_5Y"] - _PI_STAR
    be10y_signed = df["Breakeven_Inflation_10Y"] - _PI_STAR

    df["Expect_Anchor_Signed"] = pd.concat(
        [umich_signed, be5y_signed, be10y_signed], axis=1
    ).mean(axis=1)

    # ── Financial Conditions Transmission Score ───────────────────────
    # Measures how effectively rate changes are reaching the real economy.
    # If the Fed hikes but FCI stays loose, transmission is weak.
    #
    # Score = policy_stance × (-FCI)
    #   Large positive = tight policy is transmitting (FCI also tight)
    #   Near zero or negative = policy not transmitting as expected
    df["FCI_Transmission"] = df["Policy_Stance_SEP"] * (
        -df["Chicago_Fed_Financial_Conditions"]
    )
    # Positive = restrictive stance + tight FCI (normal transmission)
    # Negative = restrictive stance + loose FCI (transmission failure)
    #         or accommodative stance + tight FCI (also unusual)

    # ── Credit Impulse ────────────────────────────────────────────────
    # Second derivative of credit: acceleration/deceleration of lending.
    # Positive impulse = credit expanding faster → stimulative.
    credit_growth = df["Consumer_Credit_Total"].pct_change(_W13) * (52 / 13) * 100
    df["Credit_Impulse"] = credit_growth.diff(_W13)

    # ── Housing Affordability Pressure ────────────────────────────────
    # Composite of mortgage rate level and home price growth.
    # Both high = severe affordability pressure.
    hp_yoy = df["Case_Shiller_Home_Price"].pct_change(_W52) * 100
    df["Home_Price_YoY"] = hp_yoy
    df["Housing_Pressure"] = (
        df["Mortgage_Rate_30Y"]
        - df["Mortgage_Rate_30Y"].rolling(_W52 * 5, min_periods=52).mean()
    ) + (hp_yoy - hp_yoy.rolling(_W52 * 5, min_periods=52).mean())

    # ── Real Activity Momentum ────────────────────────────────────────
    # Composite z-score of several real-activity indicators relative
    # to their own trailing means.  Captures broad economic momentum
    # beyond just the headline UE rate.
    def _trailing_z(series, window=_W52 * 3):
        mu = series.rolling(window, min_periods=_W52).mean()
        sigma = series.rolling(window, min_periods=_W52).std()
        return (series - mu) / sigma.replace(0, np.nan)

    activity_components = []
    for col in [
        "Industrial_Production",
        "Retail_Sales",
        "Personal_Consumption_Expenditures",
        "Nonfarm_Payrolls",
    ]:
        if col in df.columns:
            z = _trailing_z(df[col])
            activity_components.append(z)

    if activity_components:
        df["Activity_Momentum"] = pd.concat(activity_components, axis=1).mean(axis=1)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. LEAD / LAG PREDICTIVE SIGNALS
# ═══════════════════════════════════════════════════════════════════════════


def _attach_leads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-looking columns that encode the empirical lead structure.

    Convention: `_Lead_NNw` means the VALUE of that indicator NN weeks
    in the FUTURE has been pulled back to the current row.  This is the
    "target" you're trying to predict.

    Convention: `_Lag_NNw` means the VALUE of that indicator NN weeks
    AGO has been placed on the current row.  These are lagged predictors
    ("what did M2 growth look like 18 months ago?").

    For simulation inputs, the _Lag columns are most useful — they
    represent information available at time t that predicts outcomes
    at time t+N.
    """

    # ── M2 → Inflation (18-month lead) ────────────────────────────────
    # M2 growth today predicts CPI ~18 months later.
    # Lag version: "what was M2 growth 78 weeks ago?" → useful as a
    # predictor of current inflation.
    df["M2_YoY_Lag_78w"] = df["M2_YoY"].shift(_W78)

    # Lead version: "what will CPI be 78 weeks from now?" → useful as
    # a simulation target.
    df["CPI_YoY_Lead_78w"] = df["CPI_YoY"].shift(-_W78)
    df["Core_PCE_YoY_Lead_52w"] = df["Core_PCE_YoY"].shift(-_W52)

    # ── Initial Claims → Unemployment (13-26 week lead) ───────────────
    df["Claims_Lag_13w"] = df["Initial_Jobless_Claims"].shift(_W13)
    df["Claims_Lag_26w"] = df["Initial_Jobless_Claims"].shift(_W26)
    df["UE_Lead_26w"] = df["Unemployment_Rate"].shift(-_W26)

    # ── Yield Curve → Recession (52-week lead) ────────────────────────
    df["Curve_10Y2Y_Lag_52w"] = df["Yield_Curve_10Y_2Y"].shift(_W52)
    df["Curve_10Y3M_Lag_52w"] = df["Yield_Curve_10Y_3M"].shift(_W52)

    # ── Breakeven → Realized Inflation (26-52 week lead) ──────────────
    df["Breakeven_5Y_Lag_26w"] = df["Breakeven_Inflation_5Y"].shift(_W26)
    df["Breakeven_5Y_Lag_52w"] = df["Breakeven_Inflation_5Y"].shift(_W52)

    # ── Housing → CPI Shelter (long lag, ~18 months) ──────────────────
    df["Home_Price_YoY_Lag_78w"] = df["Home_Price_YoY"].shift(_W78)

    # ── Credit conditions → Consumption (13-26 week lead) ─────────────
    df["Credit_Impulse_Lag_13w"] = df["Credit_Impulse"].shift(_W13)
    df["Credit_Impulse_Lag_26w"] = df["Credit_Impulse"].shift(_W26)

    # ── FFR changes → Unemployment (52-week lead) ─────────────────────
    # Rate hikes today hit the labor market ~1 year later
    df["FFR_Chg_52w"] = df["FedFunds_Rate"].diff(_W52)
    df["FFR_Chg_52w_Lag_52w"] = df["FFR_Chg_52w"].shift(_W52)

    # ── Sentiment → Consumption/Activity ──────────────────────────────
    df["Sentiment_Lag_13w"] = df["UMich_Consumer_Sentiment"].shift(_W13)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════


def score(
    df: pd.DataFrame,
    *,
    pi_star: float = _PI_STAR,
    u_star: float = _U_STAR,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Apply the full scoring layer to a FRED master DataFrame.

    Args:
        df:       weekly DataFrame from loader.py (or fred_master.csv)
        pi_star:  inflation target override (default 2.0)
        u_star:   NAIRU override (default 4.0)
        copy:     if True, operate on a copy (default).  Set False to
                  mutate in place for memory efficiency on large frames.

    Returns:
        DataFrame with all original columns plus ~60 scored columns.
    """
    global _PI_STAR, _U_STAR
    _PI_STAR = pi_star
    _U_STAR = u_star

    if copy:
        df = df.copy()

    # Ensure date is parsed
    if df.index.name == "date":
        pass  # already indexed
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    # Apply in dependency order
    df = _attach_derived(df)
    df = _attach_flags(df)
    df = _attach_continuous(df)
    df = _attach_leads(df)

    # Reset index so 'date' is a column again (matches loader.py output)
    df = df.reset_index()

    return df


def list_scored_columns() -> dict[str, list[str]]:
    """
    Return a catalog of all scored column names grouped by category.
    Useful for documentation and for programmatically selecting
    subsets of scores to register as independents in ResearchHandler.
    """
    return {
        "derived": [
            "CPI_YoY",
            "Core_CPI_YoY",
            "Core_PCE_YoY",
            "PCE_YoY",
            "CPI_Food_YoY",
            "CPI_Energy_YoY",
            "Core_PCE_3m_ann",
            "Inflation_Momentum",
            "M2_YoY",
            "M2_6m_ann",
            "Real_FFR_CPI",
            "Real_FFR_PCE",
            "Real_10Y_CPI",
            "Payroll_Change_4w",
            "Sahm_Indicator",
            "UE_3m_avg",
            "UE_12m_min",
            "JOLTS_UE_Ratio",
            "Unemployed_Approx",
            "Debt_to_GDP",
            "Deficit_GDP_Pct",
            "SEP_LR_FFR",
            "SEP_r_star",
            "Policy_Stance_SEP",
            "SEP_FFR_Med",
            "Term_Premium_Proxy",
        ],
        "flags": [
            "Flag_Curve_Inverted_10Y2Y",
            "Flag_Curve_Inverted_10Y3M",
            "Flag_Sahm_Triggered",
            "Flag_Taylor_Below_100bp",
            "Flag_Taylor_Above_100bp",
            "Flag_FCI_Contradiction",
            "Flag_Deanchor_Mild",
            "Flag_Deanchor_Severe",
            "Flag_Deanchor_Hot",
            "Flag_Deanchor_Cold",
            "Flag_M2_Excess",
            "Flag_M2_Contraction",
            "Flag_Labor_Divergence",
            "Flag_Neg_Real_Rate_HighInflation",
            "Flag_HY_Stress",
            "Flag_Fiscal_Dominance_Risk",
        ],
        "continuous": [
            "Taylor_Prescribed",
            "Taylor_Gap",
            "Inflation_Pressure",
            "Employment_Pressure",
            "Mandate_Tension",
            "Mandate_Regime",
            "Fiscal_Monetary_Conflict",
            "Expect_Anchor_Score",
            "Expect_Anchor_Signed",
            "Expect_Gap_UMich",
            "FCI_Transmission",
            "Credit_Impulse",
            "Housing_Pressure",
            "Home_Price_YoY",
            "Activity_Momentum",
        ],
        "leads_lags": [
            "M2_YoY_Lag_78w",
            "CPI_YoY_Lead_78w",
            "Core_PCE_YoY_Lead_52w",
            "Claims_Lag_13w",
            "Claims_Lag_26w",
            "UE_Lead_26w",
            "Curve_10Y2Y_Lag_52w",
            "Curve_10Y3M_Lag_52w",
            "Breakeven_5Y_Lag_26w",
            "Breakeven_5Y_Lag_52w",
            "Home_Price_YoY_Lag_78w",
            "Credit_Impulse_Lag_13w",
            "Credit_Impulse_Lag_26w",
            "FFR_Chg_52w",
            "FFR_Chg_52w_Lag_52w",
            "Sentiment_Lag_13w",
        ],
    }
