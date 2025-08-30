import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


def copy_charging() -> pd.DataFrame:
    """Read the usage data, copy charging data back to previous months, and shift from 12-3am to 12-3pm.

    Timestamp,kW
    8/13/2025 11:00 PM,0.92
    """
    df = pd.read_csv(
        "data/2024-2025.csv", parse_dates=["Timestamp"], dtype={"kW": float}
    )
    df.to_csv("data/actual.csv", index=False)
    df["hour"] = pd.to_datetime(df["Timestamp"]).dt.hour
    df["yyyymm"] = pd.to_datetime(df["Timestamp"]).dt.strftime("%Y-%m")

    # copy ev charging from 2025-04+ backwards
    df = add_ev(df)

    df["hour"] = pd.to_datetime(df["Timestamp"]).dt.hour
    for hour in range(0, 4):
        df.loc[df["hour"] == hour, "Timestamp"] += pd.Timedelta(hours=12)
        df.loc[df["hour"] == hour + 12, "Timestamp"] -= pd.Timedelta(hours=12)
    # reset hour
    df["hour"] = pd.to_datetime(df["Timestamp"]).dt.hour
    # sort by Timestamp
    df = df.sort_values(by="Timestamp").reset_index(drop=True)
    df.to_csv("data/charging.csv", index=False)
    return df


def pge_export():
    """Reformat PG&E export data file.

    Load giant PG&E data file, keep only generation rates, convert UTC to PST/PDT,
    and write to output/pge-export.csv

    RateLookupID Column:
        If the RateLookupID includes “USCA-PGXX” that indicates Delivery Export Rates.
        If the RateLookupID includes “USCA-XXPG” that indicate Generation Export Rates*.
        *Generation Export Rates are only applicable to SBP customers that receive
        bundled generation service from PG&E. Customers that receive generation service from a Community Choice Aggregator (CCA) or a Direct Access (DA) provider should refer to that generation service provider for more information about the generation export pricing available to them.

    RateName Column:
        The number that follows “NBT” represents the legacy pricing for that year.
        Please see first paragraph for a description.

    Dates and Times Columns:
        DateStart, TimeStart, DateEnd and TimeEnd values are in
        Coordinated Universal Time (UTC).

    DayStart, DayEnd and ValueName are in Pacific Prevailing Time:
        -	These fields indicate the effective day-type categories of the
            rate factor
        -	Monday through Sunday are represented as 1-7. Holidays are listed
            as number “8” in the DayStart and DayEnd columns.
        -	ValueName Column indicates the month and weekday hour or weekend
        hour starting value for the rate factor.

    Value and Unit Columns:
        This represents the dollar amount pricing per export kWh.
    """
    df = pd.read_csv("data/pge-export.csv")
    # keep only generation rates
    df = df[df["RIN"].str.contains("USCA-XXPG")]
    cols = ["DateStart", "TimeStart", "Value"]
    df = df[cols]
    df["DateTime"] = pd.to_datetime(df["DateStart"] + " " + df["TimeStart"])
    # convert DateTime from UTC to PST/PDT
    df["DateTime"] = (
        df["DateTime"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
    )
    # drop timezone from DateTime
    df["DateTime"] = df["DateTime"].dt.tz_localize(None)
    filename = "data/pge-export-pt.csv"
    df = df[["DateTime", "Value"]]
    df.to_csv(filename, index=False)
    print(f"wrote {filename}")
    # get rows where Datetime.hour is 0
    plt.plot(df["DateTime"], df["Value"])
    plt.title("PG&E Export Rates")
    plt.xlabel("Date")
    plt.ylabel("Value ($/kWh)")
    plt.grid()
    filename = "output/pge-export-dates.png"
    plt.savefig(filename)
    plt.close()
    # create chart with a histogram of the Value column
    # set bins to 0.10, 0.20, 0.30, etc.
    bins = np.arange(0, df["Value"].max() + 0.1, 0.1)
    plt.hist(df["Value"], bins=bins)
    plt.title("PG&E Export Rates")
    plt.xlabel("Value ($/kWh)")
    plt.ylabel("Count")
    plt.grid()
    filename = "output/pge-export-histogram.png"
    plt.savefig(filename)
    plt.close()
    over_off_peak = df[df["Value"] > 0.40]
    fraction = len(over_off_peak) / len(df)
    print(f"Hours over off-peak rate: {fraction:.2%}")
    over_base = df[df["Value"] > 0.03]
    fraction = len(over_base) / len(df)
    print(f"Hours over 0.03 {fraction:.2%}")
    return df


def copy_charging(df: pd.DataFrame, from_month: date, to_month: date) -> pd.DataFrame:
    """Copy usage from 0-3 from from_month to to_month"""
    end = from_month + relativedelta(months=1)
    from_dt = from_month
    to_dt = to_month
    while from_dt < end:
        for hour in range(0, 4):
            from_ts = pd.Timestamp(from_dt.year, from_dt.month, from_dt.day, hour)
            to_ts = pd.Timestamp(to_dt.year, to_dt.month, to_dt.day, hour)
            print(
                f"from {from_ts} to {to_ts}: {df.loc[df['Timestamp'] == from_ts, 'kW'].values[0]}"
            )
            df.loc[df["Timestamp"] == to_ts, "kW ev"] = df.loc[
                df["Timestamp"] == from_ts, "kW"
            ].values[0]
        from_dt += timedelta(days=1)
        to_dt += timedelta(days=1)
    return df


def add_ev(df: pd.DataFrame) -> pd.DataFrame:
    """Charging starts 2025-04-01. Copy charging from 2025-04+ backwards.

    summer: 2025-06, 2025-07 to 2024-08, 2024-09
    winter: 2025-04, 2025-05 to 2025-03, 2025-02, 2025-01, 2024-12, 2025-11, 2025-10
    """
    df["kW orig"] = df["kW"]
    # winter with charging
    df = copy_charging(df, date(2025, 5, 1), date(2025, 3, 1))
    df = copy_charging(df, date(2025, 4, 1), date(2025, 2, 1))
    df = copy_charging(df, date(2025, 5, 1), date(2025, 1, 1))
    df = copy_charging(df, date(2025, 5, 1), date(2024, 12, 1))
    df = copy_charging(df, date(2025, 4, 1), date(2024, 11, 1))
    df = copy_charging(df, date(2025, 5, 1), date(2024, 10, 1))
    # summer with charging
    df = copy_charging(df, date(2025, 6, 1), date(2024, 9, 1))
    df = copy_charging(df, date(2025, 7, 1), date(2024, 8, 1))
    return df


def main():
    # copy_charging()
    pge_export()


if __name__ == "__main__":
    main()
