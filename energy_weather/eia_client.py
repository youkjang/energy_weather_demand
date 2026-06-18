"""Client helpers for EIA hourly electricity demand data."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests


EIA_REGION_DATA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


class EIAClientError(RuntimeError):
    """Raised when the EIA API response cannot be fetched or parsed."""


def fetch_eia_hourly_demand(
    region: str,
    start_date: str,
    end_date: str,
    api_key: str,
    length: int = 5000,
) -> pd.DataFrame:
    """Fetch hourly actual electricity demand from the EIA API.

    Parameters
    ----------
    region:
        EIA respondent code, such as ``"PJM"``.
    start_date, end_date:
        Date or datetime strings accepted by the EIA API.
    api_key:
        EIA API key. The key is supplied by the caller and is never hard-coded.
    length:
        Number of rows requested per page. EIA caps this endpoint at 5,000.

    Returns
    -------
    pandas.DataFrame
        Raw hourly demand records returned by the EIA API.
    """
    if not api_key:
        raise ValueError("api_key is required.")
    if not region:
        raise ValueError("region is required.")
    if length <= 0 or length > 5000:
        raise ValueError("length must be between 1 and 5000.")

    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None

    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "facets[type][]": "D",
            "start": start_date,
            "end": end_date,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": length,
        }

        try:
            response = requests.get(EIA_REGION_DATA_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise EIAClientError(f"EIA API request failed with HTTP status {status}.") from exc
        except requests.RequestException as exc:
            raise EIAClientError(f"EIA API request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise EIAClientError("EIA API response was not valid JSON.") from exc

        api_response = payload.get("response")
        if not isinstance(api_response, dict):
            raise EIAClientError("EIA API response is missing the 'response' field.")

        page_data = api_response.get("data")
        if page_data is None:
            raise EIAClientError("EIA API response is missing the 'response.data' field.")
        if not isinstance(page_data, list):
            raise EIAClientError("EIA API field 'response.data' must be a list.")

        if total is None and api_response.get("total") is not None:
            try:
                total = int(api_response["total"])
            except (TypeError, ValueError) as exc:
                raise EIAClientError("EIA API field 'response.total' must be numeric.") from exc

        rows.extend(page_data)

        if len(page_data) < length:
            break
        offset += length
        if total is not None and offset >= total:
            break

    if not rows:
        raise EIAClientError(
            f"No EIA hourly demand records found for region {region!r} "
            f"from {start_date!r} to {end_date!r}."
        )

    return pd.DataFrame(rows)


def clean_eia_demand(df: pd.DataFrame, timezone: str = "America/New_York") -> pd.DataFrame:
    """Clean raw EIA hourly demand records for analysis.

    The returned frame contains UTC and local timestamps, local dates, region
    labels, and numeric demand in megawatt-hours.
    """
    if df.empty:
        raise ValueError("df is empty.")
    if "period" not in df.columns:
        raise ValueError("df must include a 'period' column.")
    if "value" not in df.columns and "demand_mwh" not in df.columns:
        raise ValueError("df must include a 'value' or 'demand_mwh' column.")

    cleaned = df.copy()

    if "value" in cleaned.columns:
        cleaned = cleaned.rename(columns={"value": "demand_mwh"})

    if "region" not in cleaned.columns and "respondent" in cleaned.columns:
        cleaned["region"] = cleaned["respondent"]
    if "region_name" not in cleaned.columns:
        if "respondent-name" in cleaned.columns:
            cleaned["region_name"] = cleaned["respondent-name"]
        elif "region-name" in cleaned.columns:
            cleaned["region_name"] = cleaned["region-name"]
        else:
            cleaned["region_name"] = pd.NA

    if "region" not in cleaned.columns:
        cleaned["region"] = pd.NA

    cleaned["datetime_utc"] = pd.to_datetime(cleaned["period"], errors="coerce", utc=True)
    if cleaned["datetime_utc"].isna().any():
        raise ValueError("df contains one or more invalid 'period' timestamps.")

    cleaned["datetime_local"] = cleaned["datetime_utc"].dt.tz_convert(timezone)
    cleaned["date_local"] = cleaned["datetime_local"].dt.date
    cleaned["demand_mwh"] = pd.to_numeric(cleaned["demand_mwh"], errors="coerce")

    columns = [
        "datetime_utc",
        "datetime_local",
        "date_local",
        "region",
        "region_name",
        "demand_mwh",
    ]
    return (
        cleaned[columns]
        .sort_values("datetime_utc")
        .drop_duplicates(subset=["datetime_utc"], keep="first")
        .reset_index(drop=True)
    )


def summarize_hourly_demand(df: pd.DataFrame) -> dict[str, Any]:
    """Return basic quality and range checks for hourly demand data."""
    if "demand_mwh" not in df.columns:
        raise ValueError("df must include a 'demand_mwh' column.")

    timestamp_col = "datetime_utc" if "datetime_utc" in df.columns else None
    duplicate_count = 0
    start_time = None
    end_time = None

    if timestamp_col is not None:
        duplicate_count = int(df[timestamp_col].duplicated().sum())
        start_time = df[timestamp_col].min()
        end_time = df[timestamp_col].max()

    return {
        "rows": int(len(df)),
        "start_time": start_time,
        "end_time": end_time,
        "missing_demand_count": int(df["demand_mwh"].isna().sum()),
        "duplicate_timestamp_count": duplicate_count,
        "minimum_demand": df["demand_mwh"].min(),
        "maximum_demand": df["demand_mwh"].max(),
        "mean_demand": df["demand_mwh"].mean(),
    }


def to_daily_peak_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily peak hourly demand from cleaned EIA demand data."""
    required_columns = {"date_local", "demand_mwh"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"df must include required columns: {missing_list}.")

    return (
        df.groupby("date_local", as_index=False)["demand_mwh"]
        .max()
        .rename(columns={"demand_mwh": "daily_peak_demand_mwh"})
        .sort_values("date_local")
        .reset_index(drop=True)
    )

