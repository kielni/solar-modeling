import argparse
import csv
from datetime import date, datetime
import os

import gspread
from gspread import Spreadsheet
from gspread.utils import ValueInputOption
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import numpy as np
import numpy_financial
import pandas as pd
from pydantic import BaseModel, Field
import yaml

from util import COLORS

# https://www.pge.com/assets/pge/docs/account/rate-plans/residential-electric-rate-plan-pricing.pdf
# https://www.pge.com/tariffs/assets/pdf/tariffbook/ELEC_SCHEDS_E-ELEC.pdf

"""
solar panels
https://connect.soligent.net/site/Item%20Documents/7739_110-3006%20Data%20Sheet.pdf
- 1.3–1.8 kWh/day
First year degradation: 1%
· Linear warranty after initial year:
 with 0.4% annual degradation,
 87.4% is guaranteed up to 30 years
"""
"""
https://www.franklinwh.com/products/apower2-home-battery-backup/
15 year warranty
2x  15kWh batteries = 27,300
$910 per kWh
"""
"""
What happens in January
- runs of solar output below daily use
- battery drains to zero
- grid use during peak hours

options
  - charge battery from grid? think so, need to confirm
  - larger battery (now: 3x daily peak use)
  - more solar: won't help when there is no sun; already sometimes get battery full
  - use only during peak (ie constantly fiddle with settings)
"""

"""
https://www.franklinwh.com/support/overview/grid-charge--export/
Note: In states such as California, regulations prohibit batteries from discharging to the grid if the grid was used to charge the batteries.
"""
# 0.4% degredation per year
SOLAR_DEGRADATION_FACTOR = 0.004

# capacity value used to generate hourly output
# from https://www.renewables.ninja
BASE_SOLAR_CAPACITY = 10
# scale to parameter value
SOLAR_CAPACITY = BASE_SOLAR_CAPACITY

BASE_BATTERY_CAPACITY = 30
# scale to parameter value
BATTERY_CAPACITY = BASE_BATTERY_CAPACITY

# average peak usage
PEAK_USAGE = 10

# use battery in off peak if it's at least this level
# might need all daily solar production to offset peak usage
BASE_MIN_BATTERY_WINTER = PEAK_USAGE
MIN_BATTERY_WINTER = BASE_MIN_BATTERY_WINTER
# in summer, let battery go lower since it will likely refill
BASE_MIN_BATTERY_SUMMER = PEAK_USAGE / 4
MIN_BATTERY_SUMMER = BASE_MIN_BATTERY_SUMMER

ANNUAL_RATE_INCREASE = 0.04

DEFAULT_USAGE = "data/charging.csv"

FILL_OFF_PEAK = False

OUTPUT_DIR = "output"

DAILY_USE = 43

SHEET_ID = "1vSV4EjU8OsduAFK0HzzNbDCx9E8KZpzjp8XyiYHbehQ"

RATE_VALUES = {
    "ELEC": {
        "winter off peak": 0.35,
        "winter peak": 0.38,
        "winter part peak": 0.36,
        "summer off peak": 0.40,
        "summer peak": 0.61,
        "summer part peak": 0.45,
    },
    "EV2-A": {
        "winter off peak": 0.31,
        "winter peak": 0.62,
        "winter part peak": 0.51,
        "summer off peak": 0.31,
        "summer peak": 0.50,
        "summer part peak": 0.48,
    },
}
FIRST_PEAK_HOUR = 15

FORECAST_TARIFF = "ELEC"

"""
E-ELEC

Winter Season: October-May
0.35 Off-Peak: 15 hours per day: 12 a.m.-3 p.m.
0.38 Peak: 4-9 p.m.
0.36 Partial-Peak: 3-4 p.m., 9 p.m.–12 a.m.

Summer Season: June-September
0.40 Off-Peak: 15 hours per day: 12 a.m.-3 p.m.
0.61 Peak: 4-9 p.m.
0.45 Partial-Peak: 3-4 p.m., 9 p.m.–12 a.m.

EV2-A
Winter Season: October-May
0.31 Off-peak hours are 12 midnight to 3 p.m.
0.62 Peak hours (4-9 p.m.): electricity is more expensive
0.51 Partial-peak (3-4 p.m. and 9 p.m. - 12 midnight)

Summer Season: June-September
0.31 Off-peak hours are 12 midnight to 3 p.m.
0.50 Peak hours (4-9 p.m.): electricity is more expensive
0.48 Partial-peak (3-4 p.m. and 9 p.m. - 12 midnight)
"""


MONTH_COLORS = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]


def month_label(month: date) -> str:
    return month.strftime("%m")


def set_params(config: dict):
    global OUTPUT_DIR
    global BATTERY_CAPACITY
    global SOLAR_CAPACITY
    global FILL_OFF_PEAK
    global MIN_BATTERY_WINTER
    global MIN_BATTERY_SUMMER
    global ANNUAL_RATE_INCREASE
    OUTPUT_DIR = config["output"]
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    BATTERY_CAPACITY = BASE_BATTERY_CAPACITY * config["battery"] / BASE_BATTERY_CAPACITY
    SOLAR_CAPACITY = BASE_SOLAR_CAPACITY * config["solar"] / BASE_SOLAR_CAPACITY
    FILL_OFF_PEAK = config.get("fill_off_peak", 0) > 0
    MIN_BATTERY_WINTER = config.get("min_battery_winter", MIN_BATTERY_WINTER)
    MIN_BATTERY_SUMMER = config.get("min_battery_summer", MIN_BATTERY_SUMMER)
    ANNUAL_RATE_INCREASE = config.get("annual_rate_increase", ANNUAL_RATE_INCREASE)


def set_rates(df: pd.DataFrame, tariff: str, rates: dict[str, float]) -> pd.DataFrame:
    """Set rate per kWh from period and tariff"""
    for period, rate in rates.items():
        df.loc[df["period"] == period, tariff] = rate
    df_export = load_export()
    # rename Value to credit per kWh
    df_export = df_export.rename(columns={"Value": "credit per kWh"})
    df_export["credit per kWh"] = df_export["credit per kWh"]
    # drop credit per kWh from df if it exists; replace with new value
    if "credit per kWh" in df.columns:
        df = df.drop(columns=["credit per kWh"])
    df = pd.merge(df, df_export, left_on="Timestamp", right_on="DateTime", how="left")
    # drop spring DST hour from PG&E sheet
    # drop row where kW is NaN
    df = df[df["kW"].notna()]
    df["credit per kWh"] = df["credit per kWh"].fillna(0.0)
    df["service charge"] = 0.0
    # ELEC has a service charge of $15 per billing period
    # approximate hourly to make the calculations easier
    df["service charge"] = (15 * 12) / (365 * 24)
    return df


def load_output() -> dict[str, float]:
    """Load hourly output estimate from output.csv

    day,kW
    01-01,49.18
    from https://www.renewables.ninja
    """
    output: dict[str, float] = {}
    with open("data/output.csv") as f:
        for line in f.readlines():
            if not line:
                continue
            md, kw = line.strip().split(",")
            try:
                output[md] = float(kw) * SOLAR_CAPACITY / BASE_SOLAR_CAPACITY
            except ValueError:
                continue
    return output


def chart_solar_hourly_by_month(df: pd.DataFrame, max_y: float):
    """Group by day. Plot one line per day with hour on x-axis and electricity on y-axis."""
    # get min, max, and median
    per_day = df.groupby("day").agg({"electricity": "sum"})
    min_val = round(per_day["electricity"].min())
    max_val = round(per_day["electricity"].max())
    median_val = round(per_day["electricity"].median())
    days_over_use = per_day[per_day["electricity"] >= DAILY_USE].shape[0]
    over_pct = round(days_over_use / len(per_day) * 100)
    # initialize the plot
    plt.figure(figsize=(10, 6))
    stats_text = (
        f"{min_val} - {max_val} kWh\n"
        f"median  {median_val} kWh\n"
        f"{days_over_use} days ({over_pct})% ≥ {DAILY_USE} kWh"
    )
    plt.text(
        0.01,
        0.95,
        stats_text,
        ha="left",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black" if median_val >= DAILY_USE else "red",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    plt.ylim(0, max_y)
    cmap = plt.cm.Blues
    day_count = len(df["day"].unique())
    colors = [cmap(0.3 + 0.7 * i / 30) for i in range(day_count)]
    for day, group in sorted(df.groupby("day")):
        plt.plot(
            group["hour"],
            group["electricity"],
            linewidth=2,
            color=colors[day - 1],
        )
    plt.xticks(range(0, 24, 1))  # set x ticks
    plt.title("Average hourly solar output")
    plt.xlabel("Hour")
    plt.ylabel("kW")
    highlight_peak(plt.gca())
    row = df.iloc[0]
    month_name = row["month_label"]
    month_num = row["month"]
    month = date(2025, month_num, 1)
    # add month name as title
    plt.title(f"Hourly solar output {month_name}:  {round(SOLAR_CAPACITY)} kW")
    filename = f"{get_output_dir()}/solar_hourly_{month_label(month)}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {filename}")


def chart_solar_hourly():
    """Chart solar output by hour for all months."""
    df = pd.read_csv(
        "data/output_hourly.csv",
        parse_dates=["local_time"],
        dtype={"electricity": float},
    )
    columns = ["local_time", "electricity"]
    df = df[columns]
    df["electricity"] = df["electricity"] * SOLAR_CAPACITY / BASE_SOLAR_CAPACITY
    # get first row
    df["month"] = df["local_time"].dt.month
    df["month_label"] = df["local_time"].dt.strftime("%B")
    df["day"] = df["local_time"].dt.day
    df["hour"] = df["local_time"].dt.hour
    # get max y value for all months
    max_y = df["electricity"].max() * 1.1
    plt.figure(figsize=(10, 6))  # create figure once
    by_month = pd.DataFrame()
    label_colors = []
    # chart: average by hour, one line per month
    for i, (month, group) in enumerate(sorted(df.groupby("month"))):
        chart_solar_hourly_by_month(group, max_y)
        days = len(group["day"].unique())
        daily = round(group["electricity"].sum() / days)
        label_colors.append("black" if daily >= DAILY_USE else "red")
        group = group.groupby("hour").agg({"electricity": "mean"}).reset_index()
        by_month = pd.concat([by_month, group], ignore_index=True)
        group = group.sort_values(by="hour")
        month_name = datetime(2025, month, 1).strftime("%B")
        color = MONTH_COLORS[i % len(MONTH_COLORS)]
        plt.plot(
            group["hour"],
            group["electricity"],
            label=f"{month_name} ({daily})",
            color=color,
            linewidth=2,
        )
    plt.xticks(range(0, 24, 1))
    plt.title(f"Average hourly solar output: {round(SOLAR_CAPACITY)} kW system")
    plt.xlabel("Hour")
    plt.ylabel("kW")
    plt.legend(labelcolor=label_colors)
    output_dir = get_output_dir()
    # Set legend text color to red for under_months
    filename = f"{output_dir}/solar_all_months.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {filename}")
    # write data for use by other charts
    by_month.to_csv(f"{output_dir}/solar_hourly_by_month.csv", index=False)
    # create grid of all months
    os.system(
        f"cd {output_dir} ; montage solar_hourly_*.png -tile 3x4 -geometry +2+2 solar_monthly.png; cd .."
    )


def load_export():
    df = pd.read_csv("data/pge-export-pt.csv", parse_dates=["DateTime"])
    return df


def load_hourly_output() -> dict[str, float]:
    """Load hourly output estimate from output_hourly.csv

    time,local_time,electricity,day,month
    2019-01-01 0:00,2018-12-31 16:00,1.34,01-01,01 January,,,,,,,,,,,
    from https://www.renewables.ninja
    return a dict of hour: kWh
    """
    output: dict[str, float] = {}
    reader = csv.DictReader(open("data/output_hourly.csv"))
    for row in reader:
        # 2018-12-31 16:00
        hour = row["local_time"][5:]
        output[hour] = float(row["electricity"]) * SOLAR_CAPACITY / BASE_SOLAR_CAPACITY
    return output


def set_use(df: pd.DataFrame, period_type: str, battery_level: float) -> float:
    """For a subset of rows of period_type, use battery if available, otherwise use grid.

    Return total battery use.
    """
    rows = df[df["period_type"] == period_type]
    total_battery_use = 0
    current_battery_level = battery_level
    for _, row in rows.iterrows():
        use = row["kW ev"]
        battery_use = min(use, current_battery_level)
        df.loc[row.name, f"kW battery use {period_type}"] = battery_use
        df.loc[row.name, "kW grid use"] = use - battery_use
        total_battery_use += battery_use
        current_battery_level -= battery_use
        df.loc[row.name, "battery percent"] = current_battery_level / BATTERY_CAPACITY
    return total_battery_use


class Flow(BaseModel):
    timestamp: datetime
    demand: float = Field(default=0.0)
    start_battery_level: float = Field(default=0.0)
    end_battery_level: float = Field(default=0.0)
    solar_generation: float = Field(default=0.0)
    # sources: solar, battery, grid
    from_solar: float = Field(default=0.0)
    from_battery: float = Field(default=0.0)
    from_grid: float = Field(default=0.0)
    # destinations: battery, grid
    to_battery: float = Field(default=0.0)
    to_grid: float = Field(default=0.0)


def min_battery_level(month: int) -> float:
    """Return minimum battery level to maintain during off peak hours."""
    if month >= 6 and month <= 9:
        return MIN_BATTERY_SUMMER
    return MIN_BATTERY_WINTER


"""
https://sunwatts.com/content/manual/franklin_System_User_Manual.pdf

 the homeowner can select the Time of Use mode to customize the on-peak and
off-peak times according to the electricity rate. The FranklinWH system will select solar
and aPower battery power during peak rate periods. During the off-peak periods, the
system will use power from the grid, the PV system, and the batteries in balance according
to household loads

One or two time periods may be set for every 24
hours, to achieve the most economical use of electricity for Smart Circuits.
"""


def perfect_arbitrage(
    df: pd.DataFrame, hourly_output: dict[str, float], suffix: str = ""
):
    """
    highest price is 7pm each day
    fill the battery at 2pm if needed
    do not use battery until max hour
    at max hour, dump battery
    """
    #  initial battery level for the model
    battery_level = BATTERY_CAPACITY / 2
    max_export_hour = 19  # 7pm
    rows = []
    for _, row in df.iterrows():
        flow = Flow(
            timestamp=row["Timestamp"].to_pydatetime(),
            # need this much
            demand=row["kW"],
            # available solar generation
            solar_generation=hourly_output.get(
                row["Timestamp"].strftime("%m-%d %H:%M"), 0.0
            )
            * row["solar_efficiency"],
            start_battery_level=battery_level,
        )
        min_battery = min_battery_level(flow.timestamp.month)
        period_type = row["period_type"]
        demand = flow.demand
        # from solar first
        flow.from_solar = min(demand, flow.solar_generation)
        demand -= flow.from_solar
        hour = flow.timestamp.hour
        # save battery to dump at max export hour
        preserve_battery = 15 <= hour <= max_export_hour
        if period_type in ["peak", "part peak"]:
            # then from battery
            if not preserve_battery:
                flow.from_battery = min(demand, flow.start_battery_level)
            demand -= flow.from_battery
            if demand:
                # then from grid
                flow.from_grid = demand
            else:
                # to battery
                excess = flow.solar_generation - flow.from_solar
                flow.to_battery = min(
                    BATTERY_CAPACITY - flow.start_battery_level, excess
                )
        else:  # off peak
            # from battery if battery is high enough
            if (
                demand
                and not preserve_battery
                and flow.start_battery_level > min_battery
            ):
                flow.from_battery = min(demand, flow.start_battery_level - min_battery)
                demand -= flow.from_battery
            if demand:
                flow.from_grid = demand
            else:
                # to battery
                excess = flow.solar_generation - flow.from_solar
                flow.to_battery = min(
                    excess, BATTERY_CAPACITY - flow.start_battery_level
                )
            # if last off peak hour, fill battery
            if flow.timestamp.hour == (FIRST_PEAK_HOUR - 1):
                battery_level = flow.start_battery_level + flow.to_battery
                charge = BATTERY_CAPACITY - battery_level
                flow.to_battery += charge
                flow.from_grid += charge
        # to grid
        if flow.timestamp.hour == max_export_hour:
            # send all battery to grid at max export hour
            battery_level = flow.start_battery_level + flow.to_battery
            flow.to_grid = battery_level
            flow.end_battery_level = 0
        else:
            flow.to_grid = max(
                flow.solar_generation - flow.from_solar - flow.to_battery, 0
            )
            flow.end_battery_level = (
                flow.start_battery_level - flow.from_battery + flow.to_battery
            )

        battery_level = flow.end_battery_level
        # print(flow)
        rows.append(flow.model_dump(mode="json"))
    df_flows = pd.DataFrame(rows)
    # add pd.Timestamp column to df_flows from timestamp
    df_flows["Timestamp"] = pd.to_datetime(df_flows["timestamp"])
    df = df.merge(df_flows, on="Timestamp", how="left")
    filename = f"{get_output_dir()}/flows_v3{suffix}-arbitrage.csv"
    df.to_csv(f"{filename}", index=False)
    print(f"wrote flows to {filename}")
    return df


def calculate_v3(
    df: pd.DataFrame,
    hourly_output: dict[str, float],
    fill_off_peak: float,
    suffix: str = "",
) -> pd.DataFrame:
    rows = []
    # initial battery level for the model
    battery_level = BATTERY_CAPACITY / 2
    for _, row in df.iterrows():
        flow = Flow(
            timestamp=row["Timestamp"].to_pydatetime(),
            # need this much
            demand=row["kW"],
            # available solar generation
            solar_generation=hourly_output.get(
                row["Timestamp"].strftime("%m-%d %H:%M"), 0.0
            )
            * row["solar_efficiency"],
            start_battery_level=battery_level,
        )
        min_battery = min_battery_level(flow.timestamp.month)
        period_type = row["period_type"]
        demand = flow.demand
        # from solar first
        flow.from_solar = min(demand, flow.solar_generation)
        demand -= flow.from_solar
        if period_type in ["peak", "part peak"]:
            # then from battery
            flow.from_battery = min(demand, flow.start_battery_level)
            demand -= flow.from_battery
            if demand:
                # then from grid
                flow.from_grid = demand
            else:
                # to battery
                excess = flow.solar_generation - flow.from_solar
                flow.to_battery = min(
                    BATTERY_CAPACITY - flow.start_battery_level, excess
                )
        else:  # off peak
            # from battery if battery is high enough
            if demand and flow.start_battery_level > min_battery:
                flow.from_battery = min(demand, flow.start_battery_level - min_battery)
                demand -= flow.from_battery
            if demand:
                flow.from_grid = demand
            else:
                # to battery
                excess = flow.solar_generation - flow.from_solar
                flow.to_battery = min(
                    excess, BATTERY_CAPACITY - flow.start_battery_level
                )
            # if last off peak hour and battery is less than fill_off_peak, fill battery
            if (
                fill_off_peak
                and flow.timestamp.hour == (FIRST_PEAK_HOUR - 1)
                and battery_level < fill_off_peak
            ):
                battery_level = flow.start_battery_level + flow.to_battery
                charge = fill_off_peak - battery_level
                flow.to_battery += charge
                flow.from_grid += charge
        # to grid
        flow.to_grid = flow.solar_generation - flow.from_solar - flow.to_battery
        # update battery level
        flow.end_battery_level = (
            flow.start_battery_level - flow.from_battery + flow.to_battery
        )
        battery_level = flow.end_battery_level
        # print(flow)
        rows.append(flow.model_dump(mode="json"))
    df_flows = pd.DataFrame(rows)
    # add pd.Timestamp column to df_flows from timestamp
    df_flows["Timestamp"] = pd.to_datetime(df_flows["timestamp"])
    df = df.merge(df_flows, on="Timestamp", how="left")
    filename = f"{get_output_dir()}/flows_v3{suffix}.csv"
    df.to_csv(f"{filename}", index=False)
    print(f"wrote flows to {filename}")
    return df


def chart_demand(df: pd.DataFrame, ax):
    """Chart demand, solar generation, and sources."""
    ax.plot(
        df["hour_label"],
        df["solar_generation_mean"],
        label="solar generation",
        color=COLORS["solar_generation"],
    )
    ax.fill_between(
        df["hour_label"],
        df["solar_generation_25"],
        df["solar_generation_75"],
        color=COLORS["solar_generation"],
        alpha=0.4,
    )
    # draw dashed line at solar_generation_25 and solar_generation_75
    ax.plot(
        df["hour_label"],
        df["solar_generation_25"],
        color=COLORS["solar_generation"],
        linestyle="--",
        linewidth=1,
    )
    ax.plot(
        df["hour_label"],
        df["solar_generation_75"],
        color=COLORS["solar_generation"],
        linestyle="--",
        linewidth=1,
    )
    # bar for solar
    ax.bar(
        df["hour_label"],
        df["from_solar_mean"],
        label="from solar",
        color=COLORS["from_solar"],
    )
    # bar for battery, on top of solar
    ax.bar(
        df["hour_label"],
        df["from_battery_mean"],
        label="from battery",
        bottom=df["from_solar_mean"],
        color=COLORS["from_battery"],
    )
    # bar for from grid, on top of battery
    ax.bar(
        df["hour_label"],
        df["from_grid_mean"],
        label="from grid",
        bottom=df["from_solar_mean"] + df["from_battery_mean"],
        color=COLORS["from_grid"],
    )
    # demand line
    ax.plot(df["hour_label"], df["demand_mean"], label="demand", color="black")
    ax.legend()


def chart_excess(df: pd.DataFrame, ax):
    """To battery and to grid are negative."""
    # to battery
    ax.bar(
        df["hour_label"],
        -df["to_battery_mean"],
        label="to battery",
        color=COLORS["to_battery"],
        bottom=0,
    )
    # to grid, on top of to_battery
    ax.bar(
        df["hour_label"],
        -df["to_grid_mean"],
        label="to grid",
        bottom=-df["to_battery_mean"],
        color=COLORS["to_grid"],
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_axisbelow(True)
    ax.legend()


def chart_battery(df: pd.DataFrame, ax):
    ax.plot(
        df["hour_label"],
        df["start_battery_level_25"],
        color=COLORS["battery_level"],
        linestyle="--",
    )
    ax.plot(
        df["hour_label"],
        df["start_battery_level_75"],
        color=COLORS["battery_level"],
        linestyle="--",
    )
    # shade space between 25th and 75th
    ax.fill_between(
        df["hour_label"],
        df["start_battery_level_25"],
        df["start_battery_level_75"],
        color=COLORS["battery_level"],
        alpha=0.5,
    )
    ax.plot(
        df["hour_label"],
        df["start_battery_level_mean"],
        label="battery",
        color=COLORS["battery_level"],
    )
    ax.plot(df["hour_label"], df["demand_mean"], label="demand", color="black")
    ax.legend()


def chart_monthly_sources(df: pd.DataFrame):
    """Sum flows by month.

    Group by month and sum from_solar, from_battery, from_grid, to_grid.
    Create one stacked bar per month.
    """
    print("charting monthly sources")
    fig, ax = plt.subplots(figsize=(12, 6))
    df_monthly = df.groupby("month").agg(
        {
            "from_solar": "sum",
            "from_battery": "sum",
            "from_grid": "sum",
            "to_grid": "sum",
        }
    )
    df_monthly = df_monthly.reset_index()
    df_monthly["month_label"] = df_monthly["month"].apply(
        lambda x: datetime(2025, x, 1).strftime("%b")
    )
    # stack bars for from_solar, from_battery, from_grid
    ax.bar(
        df_monthly["month_label"],
        df_monthly["from_solar"],
        label="from solar",
        color=COLORS["from_solar"],
    )
    ax.bar(
        df_monthly["month_label"],
        df_monthly["from_battery"],
        label="from battery",
        bottom=df_monthly["from_solar"],
        color=COLORS["from_battery"],
    )
    ax.bar(
        df_monthly["month_label"],
        df_monthly["from_grid"],
        label="from grid",
        bottom=df_monthly["from_solar"] + df_monthly["from_battery"],
        color=COLORS["from_grid"],
    )
    # draw to_grid as negative bar
    ax.bar(
        df_monthly["month_label"],
        -df_monthly["to_grid"],
        label="to grid (export)",
        color=COLORS["to_grid"],
        bottom=None,
    )
    ax.legend()
    # label y axis as kWh
    ax.set_ylabel("kWh")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(
        f"Sources by month:  {round(SOLAR_CAPACITY)} kW solar, {round(BATTERY_CAPACITY)} kWh battery"
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    # draw horizontal line at y=0
    # add title: Sources by month:  10 kW solar, 30 kWh battery
    # draw horizontal grid lines
    plt.tight_layout()
    fig.savefig(f"{get_output_dir()}/monthly_sources.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def highlight_peak(ax: plt.Axes):
    ax.set_xlim(-0.5, 23.5)
    # part peak: with light yellow color for hour=15, hour=21-23
    ax.axvspan(15, 24, color=COLORS["part peak"], alpha=1.0)
    # peak: full height bar with light red color for hour=16-20
    ax.axvspan(16, 20, color=COLORS["peak"], alpha=0.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="lightgray", alpha=0.7)


def chart_flows(df: pd.DataFrame, month: date, max_y: float):
    # positive: from_solar + from_battery + from_grid = demand
    # negative: to_battery + to_grid = excess
    # solar generation, demand
    # battery level
    print(f"charting flows for {month.strftime('%B')}")
    cols = [
        "from_solar",
        "from_battery",
        "from_grid",
        "demand",
        "solar_generation",
        "to_battery",
        "to_grid",
        "start_battery_level",
    ]
    df_hourly = (
        df.groupby("hour")[cols]
        .quantile([0.25, 0.5, 0.75])  # 25th, median, 75th
        .unstack(level=-1)  # move quantiles into columns
    )
    df_hourly.columns = [f"{col}_{int(q * 100)}" for col, q in df_hourly.columns]
    df_hourly["hour_label"] = df_hourly.index.astype(str)

    # get mean by hour
    df_hourly_mean = df.groupby("hour").agg({col: "mean" for col in cols})
    df_hourly_mean.columns = [f"{col}_mean" for col in cols]
    df_hourly_mean["hour_label"] = df_hourly_mean.index.astype(str)
    df_hourly = pd.merge(df_hourly, df_hourly_mean, on="hour_label", how="left")

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    for ax in axs:
        highlight_peak(ax)
    chart_battery(df_hourly, axs[1])
    min_y = (df_hourly["to_grid_mean"] + df_hourly["to_battery_mean"]).max() * 1.05
    axs[0].set_ylim(-min_y, max_y)
    chart_demand(df_hourly, axs[0])
    chart_excess(df_hourly, axs[0])
    axs[1].set_ylim(0, BATTERY_CAPACITY * 1.05)
    plt.tight_layout()
    plt.suptitle(f"{month.strftime('%B')} hourly: {label_from_params()}", y=1.00)
    fig.savefig(
        f"{get_output_dir()}/flow_{month_label(month)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def open_spreadsheet() -> Spreadsheet:
    gc = gspread.auth.service_account(filename="google.json")
    return gc.open_by_key(SHEET_ID)


def write_to_sheet(df: pd.DataFrame, sheet_name: str):
    # use gspread to write to Google Sheet
    print("writing to Google Sheet")
    """
    ['Timestamp', 'kW', 'month', 'hour', 'yyyymm', 'ymd', 'season', 'period', 'period_type', 'timestamp', 
    'demand', 'start_battery_level', 'end_battery_level', 'solar_generation', 'from_solar', 'from_battery', 'from_grid', 'to_battery', 'to_grid', 
    'ELEC', 'credit per kWh', 'service charge', 'pge cost ELEC', 'pge credit', 'grid cost ELEC', 'net cost ELEC', 'savings ELEC', 
    'EV2-A', 'pge cost EV2-A', 'grid cost EV2-A', 'net cost EV2-A', 'savings EV2-A']
    """
    cols = [
        "Timestamp",
        "season",
        "period",
        "period_type",
        "demand",
        "solar_generation",
        "start_battery_level",
        "end_battery_level",
        "from_solar",
        "from_battery",
        "from_grid",
        "to_battery",
        "to_grid",
        "credit per kWh",
        "service charge",
        "pge credit",
        "ELEC",
        "pge cost ELEC",
        "grid cost ELEC",
        "EV2-A",
        "pge cost EV2-A",
        "grid cost EV2-A",
        "net cost ELEC",
        "savings ELEC",
        "net cost EV2-A",
        "savings EV2-A",
    ]
    # convert Timestamp to string
    df["Timestamp"] = df["Timestamp"].astype(str)
    # keep only cols
    df = df[cols]
    print(df.columns.values.tolist())
    df.to_csv(f"{get_output_dir()}/flows-model.csv", index=False)
    ss = open_spreadsheet()
    # create worksheet if it doesn't exist
    if sheet_name not in [worksheet.title for worksheet in ss.worksheets()]:
        worksheet = ss.add_worksheet(
            sheet_name, rows=len(df.index), cols=len(df.columns)
        )
    else:
        worksheet = ss.worksheet(sheet_name)
    print(f"writing to {sheet_name}")
    worksheet.update(
        [df.columns.values.tolist()] + df.values.tolist(),
        value_input_option=ValueInputOption.user_entered,
    )


def add_costs(
    df: pd.DataFrame, tariffs: dict[str, dict[str, float]], suffix: str = ""
) -> pd.DataFrame:
    for tariff in tariffs:
        df = set_rates(df, tariff, tariffs[tariff])
        # cost without solar: demand * rate
        col = f"pge cost {tariff}"
        df[col] = df["demand"] * df[tariff] + df["service charge"]
        # cost with solar: from_grid * rate - to_grid * credit per kWh
        credit_col = "pge credit"
        df[credit_col] = df["to_grid"] * df["credit per kWh"]
        grid_col = f"grid cost {tariff}"
        df[grid_col] = df["from_grid"] * df[tariff]
        if tariff == "ELEC":
            df[grid_col] += df["service charge"]
        net_col = f"net cost {tariff}"
        df[net_col] = df[grid_col] - df[credit_col]
        df[f"savings {tariff}"] = df[col] - df[net_col]
    df.to_csv(f"{get_output_dir()}/costs{suffix}.csv", index=False)
    return df


def label_periods(df: pd.DataFrame) -> pd.DataFrame:
    # date helpers
    df["month"] = pd.to_datetime(df["Timestamp"]).dt.month
    df["hour"] = pd.to_datetime(df["Timestamp"]).dt.hour
    df["day"] = pd.to_datetime(df["Timestamp"]).dt.day
    df["yyyymm"] = pd.to_datetime(df["Timestamp"]).dt.strftime("%Y-%m")
    df["ymd"] = pd.to_datetime(df["Timestamp"]).dt.strftime("%Y-%m-%d")

    # season
    df["season"] = "winter"
    df.loc[df["month"].between(6, 9), "season"] = "summer"

    # periods
    df.loc[(df["season"] == "winter") & (df["hour"] < 15), "period"] = "winter off peak"
    df.loc[(df["season"] == "winter") & (df["hour"] == 15), "period"] = (
        "winter part peak"
    )
    df.loc[(df["season"] == "winter") & (df["hour"].between(16, 20)), "period"] = (
        "winter peak"
    )
    df.loc[(df["season"] == "winter") & (df["hour"] >= 21), "period"] = (
        "winter part peak"
    )

    df.loc[(df["season"] == "summer") & (df["hour"] < 15), "period"] = "summer off peak"
    df.loc[(df["season"] == "summer") & (df["hour"] == 15), "period"] = (
        "summer part peak"
    )
    df.loc[(df["season"] == "summer") & (df["hour"].between(16, 20)), "period"] = (
        "summer peak"
    )
    df.loc[(df["season"] == "summer") & (df["hour"] >= 21), "period"] = (
        "summer part peak"
    )
    df["period_type"] = "peak"
    df.loc[df["period"].str.contains("off peak"), "period_type"] = "off peak"
    df.loc[df["period"].str.contains("part peak"), "period_type"] = "part peak"
    return df


def initial_setup(filename: str):
    """Read the usage data

    Timestamp,kW
    """
    df = pd.read_csv(filename, parse_dates=["Timestamp"], dtype={"kW": float})
    # fill DST hour with 0.0
    return df.fillna(0.0)


def get_output_dir():
    return OUTPUT_DIR


def print_summary(df: pd.DataFrame, config: dict, payback_period: float, irr: float):
    lines = []
    lines.append(f"## {label_from_params()}\n")
    lines.append(f"solar capacity\t{round(SOLAR_CAPACITY)} kW")
    lines.append(f"battery capacity\t{round(BATTERY_CAPACITY)} kWh")
    lines.append(f"fill off peak\t{config['fill_off_peak']} kWh")
    lines.append(
        f"min battery level\tsummer={round(MIN_BATTERY_SUMMER)} kWh\twinter={round(MIN_BATTERY_WINTER)} kWh"
    )
    # average daily use by period type
    lines.append("## Average daily use by period\n")
    daily = df.groupby(["ymd", "period_type"]).agg({"kW": "sum"}).reset_index()
    average = daily.groupby("period_type")["kW"].mean()
    for period_type, avg_kw in average.items():
        lines.append(f"{period_type}\t{round(avg_kw)} kWh")
    lines.append("\n")
    # pge cost ELEC, net cost ELEC,
    # pge cost EV2-A, net cost EV2-A
    lines.append("\ntariff\tPG&E\tnet cost\tsavings")
    row = {
        "scenario": f"{label_from_params()} {ANNUAL_RATE_INCREASE:.0%} rate increase"
    }
    for tariff in RATE_VALUES.keys():
        pge_cost = df[f"pge cost {tariff}"].sum()
        net_cost = df[f"net cost {tariff}"].sum()
        savings = df[f"savings {tariff}"].sum()
        lines.append(f"{tariff}\t${pge_cost:,.0f}\t${net_cost:,.0f}\t${savings:,.0f}")
        row[f"pge cost {tariff}"] = pge_cost
        row[f"net cost {tariff}"] = net_cost
        row[f"savings {tariff}"] = savings
    lines.append("\n")
    lines.append("## Annual values\n")
    # demand, solar production, to grid
    demand = df["demand"].sum()
    from_grid = df["from_grid"].sum()
    solar = df["solar_generation"].sum()
    to_grid = df["to_grid"].sum()
    export_df = df[df["to_grid"] > 0]
    export_df["credit"] = export_df["to_grid"] * export_df["credit per kWh"]
    credit = export_df["credit"].sum()
    credit_per_kwh = credit / to_grid if to_grid else 0.0
    lines.append(f"demand\t{demand:,.0f} kWh")
    lines.append(f"solar\t{solar:,.0f} kWh")
    lines.append(f"to grid\t{to_grid:,.0f} kWh")
    lines.append(f"credits\t${credit:,.0f}")
    lines.append(f"credit per kWh\t${credit_per_kwh:,.2f}")
    lines.append(f"annual rate increase\t{ANNUAL_RATE_INCREASE:.0%}")
    lines.append(f"payback period\t{payback_period:.1f} years")
    lines.append(f"IRR\t{irr:.2%}")
    row["demand"] = demand
    row["solar"] = solar
    row["from grid"] = from_grid
    row["to grid"] = to_grid
    row["to grid rate"] = credit_per_kwh
    row["credit"] = credit
    row["annual rate increase"] = ANNUAL_RATE_INCREASE
    row["payback period"] = payback_period
    row["IRR"] = irr
    print("\n".join(lines))
    filename = f"{get_output_dir()}/summary.md"
    with open(filename, "w") as f:
        f.write("\n\n".join(lines))
    print(f"wrote summary to {filename}")
    df_summary = pd.DataFrame([row])
    df_summary.to_csv(f"{get_output_dir()}/summary.csv", index=False)


def source_label(df: pd.DataFrame, col: str) -> int:
    percent = round(df[col].sum() / df["demand"].sum() * 100)
    return f"({percent}%)"


def chart_daily(df: pd.DataFrame, month: date):
    print(f"writing daily chart for {month.strftime('%B')}")
    # chart 1: battery, solar, demand, excess
    # sort by Timestamp
    df = df.sort_values(by="Timestamp")
    # demand = df["demand"].sum()
    cols = [
        "demand",
        "solar_generation",
        "to_grid",
        "from_solar",
        "from_battery",
        "from_grid",
        "to_battery",
    ]
    df_daily = df.groupby("day").agg({col: "sum" for col in cols}).reset_index()
    # group df by day and get first value of start_battery_level
    df_battery = df.groupby("day")["start_battery_level"].first()
    df_daily = df_daily.merge(df_battery, on="day", how="left")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 31.5)
    # lines for battery level, demand, solar generation
    ax.plot(
        df_daily["day"],
        df_daily["start_battery_level"],
        label=f"battery level",
        color=COLORS["battery_level"],
        linewidth=2,
    )
    ax.plot(
        df_daily["day"],
        df_daily["solar_generation"],
        label="solar generation",
        color=COLORS["solar_generation"],
        linewidth=2,
    )
    # bars for sources
    ax.bar(
        df_daily["day"],
        df_daily["from_solar"],
        label=f"from solar {source_label(df, 'from_solar')}",
        color=COLORS["from_solar"],
    )
    ax.bar(
        df_daily["day"],
        df_daily["from_battery"],
        label=f"from battery {source_label(df, 'from_battery')}",
        color=COLORS["from_battery"],
        bottom=df_daily["from_solar"],
    )
    ax.bar(
        df_daily["day"],
        df_daily["from_grid"],
        label=f"from grid {source_label(df, 'from_grid')}",
        color=COLORS["from_grid"],
        bottom=df_daily["from_solar"] + df_daily["from_battery"],
    )
    # show to_grid as negative
    total_to_grid = df_daily["to_grid"].sum()
    ax.bar(
        df_daily["day"],
        -df_daily["to_grid"],
        label=f"to grid {total_to_grid:,.0f} kWh",
        color=COLORS["to_grid"],
        bottom=None,
    )
    # label y axis as kWh
    ax.set_ylabel("kWh")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"{month.strftime('%B')}: {label_from_params()}")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    plt.tight_layout()
    # plt.show()
    ax.legend()
    fig.savefig(
        f"{get_output_dir()}/daily_{month_label(month)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def label_from_params():
    s = f"{round(SOLAR_CAPACITY)} kW solar, {round(BATTERY_CAPACITY)} kWh battery"
    if FILL_OFF_PEAK:
        s += ", fill off peak"
    return s


def cost_chart_setup(ax, months: list[str]):
    x = np.arange(1, 13)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend()
    # add grid lines
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0.5, 12.5)


def chart_costs(df: pd.DataFrame):
    """Create chart with side by side bars for pge cost ELEC and net cost ELEC

    Add total savings for the year as a text box.
    """
    tariff = FORECAST_TARIFF
    # group by month and sum pge cost {tariff} and net cost {tariff}
    df = df.sort_values(by="Timestamp")
    df_monthly = (
        df.groupby("month")
        .agg(
            {
                f"pge cost {tariff}": "sum",
                f"net cost {tariff}": "sum",
                "pge credit": "sum",
                f"to_grid": "sum",
            }
        )
        .reset_index()
        .sort_values(by="month")
    )

    months = [date(2025, month, 1).strftime("%b") for month in df_monthly["month"]]
    x = np.arange(len(months))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        df_monthly[f"pge cost {tariff}"],
        width,
        label="PG&E",
        color=COLORS["from_grid"],
        alpha=0.5,
    )
    ax.bar(
        x + width / 2,
        df_monthly[f"net cost {tariff}"],
        width,
        label="solar + battery",
        color=COLORS["system"],
    )
    ax.plot(
        x,
        df_monthly["pge credit"],
        label="export credit",
        color=COLORS["from_grid"],
    )
    cost_chart_setup(ax, months)
    ax.set_ylabel("cost $")
    total_savings = df[f"savings {tariff}"].sum()
    savings_pct = round(
        df[f"net cost {tariff}"].sum() / df[f"pge cost {tariff}"].sum() * 100
    )
    tariff = FORECAST_TARIFF
    # group by month, sum net cost {tariff}
    by_month = df.groupby("month").agg({f"net cost {tariff}": "sum"})
    monthly_bill = by_month[f"net cost {tariff}"].mean()
    ax.set_title(f"{tariff} cost: {label_from_params()}")

    text = (
        f"${total_savings:,.0f} annual savings\n"
        f"${monthly_bill:,.0f} average monthly bill\n"
        f"{savings_pct}% of PG&E cost\n"
    )
    plt.text(
        0.5,
        0.95,
        text,
        ha="center",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=12,
        color="black",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    max_credit = df_monthly["pge credit"].max()
    plt.tight_layout()
    fig.savefig(f"{get_output_dir()}/costs.png", dpi=300, bbox_inches="tight")
    print(f"wrote costs to {get_output_dir()}/costs.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    # plot monthly pge credit as bars
    df_credits = df[df["to_grid"] > 0]
    df_credits = df_credits[["month", "credit per kWh", "to_grid", "pge credit"]]
    df_credits.to_csv(f"{get_output_dir()}/credits.csv", index=False)
    total_credit = df_credits["pge credit"].sum()

    df_monthly["credit per kWh"] = df_monthly["pge credit"] / df_monthly["to_grid"]
    df_monthly["credit per kWh"] = df_monthly["credit per kWh"].fillna(0.0)
    ax.set_title(f"{label_from_params()} export credit: ${total_credit:,.0f}")
    ax.set_ylabel("credit $")
    ax.set_ylim(0, max_credit * 1.1)
    ax.bar(
        df_monthly["month"],
        df_monthly["pge credit"],
        color=COLORS["from_grid"],
    )
    cost_chart_setup(ax, months)
    # add value of df_monthly["to_grid"] as text above each bar
    for _, row in df_monthly.iterrows():
        plt.text(
            row["month"],
            row["pge credit"],
            (f"{row['to_grid']:.0f} kWh\n$" f"{row['credit per kWh']:.2f}/kWh"),
            fontsize=10,
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    fig.savefig(f"{get_output_dir()}/credits.png", dpi=300, bbox_inches="tight")
    print(f"wrote credits to {get_output_dir()}/credits.png")
    plt.close(fig)


def chart_roi(df: pd.DataFrame, payback_period: float, irr: float):
    """Chart cumulative savings line"""
    df["year"] = pd.to_datetime(df["yyyymm"]).dt.year
    # Convert yyyymm strings to datetime for proper x-axis plotting
    df["date"] = pd.to_datetime(df["yyyymm"])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        df["date"],
        df["cumulative savings"],
        color=COLORS["savings"],
        label="savings",
        linestyle="--",
    )
    ax.plot(df["date"], df["cumulative pge"], color=COLORS["from_grid"], label="PG&E")
    ax.plot(
        df["date"],
        df["cumulative net"],
        color=COLORS["system"],
        label="solar + battery",
    )
    ax.set_title(f"Cumulative: {label_from_params()}")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(df["date"].min(), df["date"].max())
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${int(x):,}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    text = f"payback period: {payback_period:.1f} years\n" f"IRR: {irr:.2%}"
    plt.text(
        0.99,
        0.05,
        text,
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    plt.tight_layout()
    filename = f"{get_output_dir()}/roi.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"wrote {filename}")
    plt.close(fig)


def calculate_roi(df: pd.DataFrame, config: dict):
    """
    Timestamp,kW,hour,yyyymm,month,day,ymd,season,period,period_type
    """
    years = 15
    hourly_output = load_hourly_output()
    df_years = pd.DataFrame()
    rates = {}
    rates.update(RATE_VALUES)
    solar_degradation_factor = SOLAR_DEGRADATION_FACTOR
    solar_efficiency = 1.0
    for year in range(years):
        if year > 0:
            # reduce the solar generation by the degradation factor
            solar_efficiency = (1 - solar_degradation_factor) ** year
            # increase the per kwH cost by 4%
            for tariff in rates:
                for period in rates[tariff]:
                    rates[tariff][period] *= 1 + ANNUAL_RATE_INCREASE
        print(
            f"year {year} solar efficiency: {solar_efficiency} off peak rate: {rates[FORECAST_TARIFF]['winter off peak']}"
        )
        df_year = df.copy()
        df_year["Timestamp"] = df["Timestamp"] + pd.Timedelta(days=365 * year)
        df_year["solar_efficiency"] = solar_efficiency
        suffix = f"_year_{year:02d}"
        df_year = calculate_v3(
            df_year,
            hourly_output,
            config["fill_off_peak"],
            suffix,
        )
        df_year = add_costs(df_year, rates, suffix)
        df_years = pd.concat([df_years, df_year])
    # set yyyymm as the year and month
    df_years["yyyymm"] = df_years["Timestamp"].dt.strftime("%Y-%m")
    # pge cost FORECAST_TARIFF, net cost FORECAST_TARIFF
    df_years = (
        df_years.groupby("yyyymm")
        .agg(
            {
                f"savings {FORECAST_TARIFF}": "sum",
                f"net cost {FORECAST_TARIFF}": "sum",
                f"pge cost {FORECAST_TARIFF}": "sum",
            }
        )
        .reset_index()
    )
    df_years = df_years.sort_values(by="yyyymm")
    df_years["cumulative savings"] = (
        df_years[f"savings {FORECAST_TARIFF}"].cumsum() - config["system_cost"]
    )
    df_years["cumulative pge"] = df_years[f"pge cost {FORECAST_TARIFF}"].cumsum()
    df_years["cumulative net"] = df_years[f"net cost {FORECAST_TARIFF}"].cumsum()

    # payback period: first month where cumulative savings is positive
    payback_month = df_years[df_years["cumulative savings"] > 0]["yyyymm"].iloc[0]
    year, month = payback_month.split("-")
    payback_dt = date(int(year), int(month), 1)
    min_timestamp = df["Timestamp"].min().date()
    years = (payback_dt - min_timestamp).days / 365
    print(f"payback period: {payback_dt} ({years:.1f} years)")
    irr = numpy_financial.irr(
        [-config["system_cost"]] + df_years[f"savings {FORECAST_TARIFF}"].to_list()
    )
    print(f"IRR: {irr:.2%}")
    df_years.to_csv(f"{get_output_dir()}/roi.csv", index=False)
    return df_years, years, irr


def main(config: dict):
    df_initial = initial_setup(config.get("usage", DEFAULT_USAGE))
    df_initial = label_periods(df_initial)
    df_initial["solar_efficiency"] = 1.0
    df_initial.to_csv(f"{get_output_dir()}/initial.csv", index=False)
    hourly_output = load_hourly_output()

    # get data for Sep 2025
    df_sep = df_initial[df_initial["month"] == 9]
    df_sep = perfect_arbitrage(df_sep, hourly_output)
    df_sep = add_costs(df_sep, RATE_VALUES, suffix="_arbitrage")
    df_sep.to_csv(f"{get_output_dir()}/arbitrage.csv", index=False)
    write_to_sheet(df_sep, "arbitrage")
    # sum net cost ELEC, savings ELEC, pge credit

    # year 1 flows
    df = calculate_v3(df_initial, hourly_output, config["fill_off_peak"])
    df = add_costs(df, RATE_VALUES)
    write_to_sheet(df, config["output"])
    max_y = df[["demand", "solar_generation"]].max().max() * 1.1
    df = df.sort_values(by="Timestamp")
    chart_costs(df)
    chart_monthly_sources(df)
    # chart for each month
    for month, group in df.groupby(df["month"]):
        chart_flows(group, date(2025, month, 1), max_y=max_y)
        chart_daily(group, date(2025, month, 1))
    chart_solar_hourly()
    df_roi, payback_period, irr = calculate_roi(df_initial, config)
    chart_roi(df_roi, payback_period, irr)

    print_summary(df, config, payback_period, irr)


def load_config(filename: str):
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, default="solar-10-battery-30-actual.yml", nargs="?"
    )
    args = parser.parse_args()
    _config = load_config(args.config)
    set_params(_config)
    main(_config)
