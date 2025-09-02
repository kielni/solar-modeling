import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from util import RATE_VALUES


def daily_arbitrage_targets(tariff: str) -> pd.DataFrame:
    df = pd.read_csv("data/pge-export-pt.csv", parse_dates=["DateTime"])
    # get the max rate for each day
    df["date"] = pd.to_datetime(df["DateTime"]).dt.date
    df["total credit"] = df["generation credit"] + df["delivery credit"]
    print(f"{len(df)} total hours")
    # first drop everything below off peak
    df = df[df["total credit"] > 0.40]
    print(f"{len(df)} credit above off peak")
    df = df.loc[
        df.groupby("date")["total credit"]
        .apply(lambda x: x.nlargest(2).index)
        .explode()
    ]
    df["month"] = pd.to_datetime(df["DateTime"]).dt.month
    df["date"] = pd.to_datetime(df["DateTime"]).dt.date
    df["hour"] = pd.to_datetime(df["DateTime"]).dt.hour
    # set date to date portion of DateTime
    # set season to summer if month is between 6 and 9 else winter
    df["season"] = "winter"
    df.loc[df["month"].between(6, 9), "season"] = "summer"
    winter_max = RATE_VALUES[tariff]["winter peak"].total()
    summer_max = RATE_VALUES[tariff]["summer peak"].total()
    # set target to max rate for the season
    df["target"] = winter_max
    df.loc[df["season"] == "summer", "target"] = summer_max
    df = df[df["total credit"] > df["target"]]
    df = df.sort_values(by="date")
    # DateTime,generation credit,delivery credit,date,total credit,month,season,target
    print(f"{len(df)} arbitrage target days")
    filename = "data/arbitrage_targets.csv"
    print(f"writing arbitrage targets to {filename}")
    df.to_csv(filename, index=False)
    df = df[["DateTime", "date", "hour"]]
    return df


def copy_charging_orig(df: pd.DataFrame, from_month: date, to_month: date) -> pd.DataFrame:
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
    df = df[["DateStart", "TimeStart", "Value", "RIN"]]
    df["DateTime"] = pd.to_datetime(df["DateStart"] + " " + df["TimeStart"])
    # convert DateTime from UTC to PST/PDT
    df["DateTime"] = (
        df["DateTime"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
    )
    # drop timezone from DateTime
    df["DateTime"] = df["DateTime"].dt.tz_localize(None)

    df_gen = df[df["RIN"].str.contains("USCA-XXPG")]
    df_gen["generation credit"] = df_gen["Value"]
    df_tnd = df[df["RIN"].str.contains("USCA-PGXX")]
    df_tnd["delivery credit"] = df_tnd["Value"]

    # merge on DateTime
    df = pd.merge(
        df_gen[["DateTime", "generation credit"]],
        df_tnd[["DateTime", "delivery credit"]],
        on="DateTime",
    )
    df = df[["DateTime", "generation credit", "delivery credit"]]
    filename = "data/pge-export-pt.csv"
    df.to_csv(filename, index=False)
    print(f"wrote {filename}")
    df["total credit"] = df["generation credit"] + df["delivery credit"]
    # get rows where Datetime.hour is 0
    plt.plot(df["DateTime"], df["generation credit"], label="generation credit")
    plt.plot(df["DateTime"], df["delivery credit"], label="delivery credit")
    plt.title("PG&E Export Rates")
    plt.xlabel("Date")
    plt.ylabel("Value ($/kWh)")
    plt.grid()
    filename = "output/pge-export-dates.png"
    plt.savefig(filename)
    plt.close()
    # create chart with a histogram of the Value column
    # set bins to 0.10, 0.20, 0.30, etc.
    bins = np.arange(0, df["total credit"].max() + 0.1, 0.1)
    plt.hist(df["total credit"], bins=bins)
    plt.title("PG&E Export Rates")
    plt.xlabel("Value ($/kWh)")
    plt.ylabel("Count")
    plt.grid()
    filename = "output/pge-export-histogram.png"
    plt.savefig(filename)
    plt.close()
    over_off_peak = df[df["total credit"] > 0.40]
    fraction = len(over_off_peak) / len(df)
    print(f"Hours over off-peak rate: {fraction:.2%}")
    over_base = df[df["total credit"] > 0.03]
    fraction = len(over_base) / len(df)
    print(f"Hours over 0.03 {fraction:.2%}")
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


def main(op: str, tariff: str):
    if op == "copy_charging":
        copy_charging()
    elif op == "arbitrage_targets":
        daily_arbitrage_targets(tariff)
    elif op == "pge_export":
        pge_export()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("op", type=str)
    parser.add_argument("--tariff", type=str, default="ELEC", nargs="?")
    args = parser.parse_args()

    main(args.op, args.tariff)
