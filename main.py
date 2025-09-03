import argparse
import csv
from datetime import date, datetime
import os
from typing import Any
import calendar

import gspread
from gspread import Spreadsheet
from gspread.utils import ValueInputOption
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import numpy as np
import numpy_financial
import pandas as pd
from pydantic import BaseModel, Field
import yaml

from util import (
    COLORS,
    RATE_VALUES,
    Charges,
    RATE_VALUES_MONTHLY,
    RATE_VALUES_DAILY,
    BONUS_CREDIT,
    BONUS_END_YEAR,
)

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

Credits only apply to generation:
April bill: 435.52
163.65 generation (38%) 271.87 T&D

generation fraction = 0.33 from recent bills
"""
START_YEAR = 2026

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
# max 10 kW per 15 kWh battery
BATTERY_DISCHARGE_RATIO = 10 / 15

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

OUTPUT_DIR = "output"

DAILY_USE = 43

SHEET_ID = "1vSV4EjU8OsduAFK0HzzNbDCx9E8KZpzjp8XyiYHbehQ"

FIRST_PEAK_HOUR = 15

FORECAST_TARIFF = "ELEC"


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
    # TODO: less global config
    global OUTPUT_DIR
    global BATTERY_CAPACITY
    global SOLAR_CAPACITY
    global MIN_BATTERY_WINTER
    global MIN_BATTERY_SUMMER
    global ANNUAL_RATE_INCREASE
    global LABEL
    OUTPUT_DIR = config["output"]
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    BATTERY_CAPACITY = BASE_BATTERY_CAPACITY * config["battery"] / BASE_BATTERY_CAPACITY
    SOLAR_CAPACITY = BASE_SOLAR_CAPACITY * config["solar"] / BASE_SOLAR_CAPACITY
    MIN_BATTERY_WINTER = config.get("min_battery_winter", MIN_BATTERY_WINTER)
    MIN_BATTERY_SUMMER = config.get("min_battery_summer", MIN_BATTERY_SUMMER)
    ANNUAL_RATE_INCREASE = config.get("annual_rate_increase", ANNUAL_RATE_INCREASE)
    LABEL = config.get(
        "description",
        f"{round(SOLAR_CAPACITY)} kW solar, {round(BATTERY_CAPACITY)} kWh battery",
    )


def set_rates(df: pd.DataFrame, tariff: str, rates: dict[str, Any]) -> pd.DataFrame:
    """Set rate per kWh from period and tariff"""
    for period in rates:
        charges = rates[period]
        df.loc[df["period"] == period, f"{tariff} generation"] = charges.generation
        df.loc[df["period"] == period, f"{tariff} delivery"] = charges.delivery
        df.loc[df["period"] == period, f"{tariff} other"] = charges.other
    df_export = load_export()
    # drop credits and replace with new values
    for key in ["generation credit", "delivery credit", "bonus credit"]:
        if key in df.columns:
            df = df.drop(columns=[key])
    df = pd.merge(df, df_export, left_on="Timestamp", right_on="DateTime", how="left")
    # drop spring DST hour from PG&E sheet
    # drop row where demand is NaN
    df = df[df["demand"].notna()]
    df["generation credit"] = df["generation credit"].fillna(0.0)
    df["delivery credit"] = df["delivery credit"].fillna(0.0)
    # set bonus credit to BONUS_CREDIT through BONUS_END_YEAR
    df["bonus credit"] = np.where(
        df["Timestamp"].dt.year <= BONUS_END_YEAR, BONUS_CREDIT, 0.0
    )
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
    month = date(START_YEAR, month_num, 1)
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
        month_name = datetime(START_YEAR, month, 1).strftime("%B")
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
    cols = ["hour", "electricity"]
    by_month = by_month[cols]
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


def apply_flow(flow: Flow, period_type: str) -> Flow:
    demand = flow.demand
    min_battery = min_battery_level(flow.timestamp.month)
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
            flow.to_battery = min(BATTERY_CAPACITY - flow.start_battery_level, excess)
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
            flow.to_battery = min(excess, BATTERY_CAPACITY - flow.start_battery_level)
    # to grid
    flow.to_grid = flow.solar_generation - flow.from_solar - flow.to_battery
    # update battery level
    flow.end_battery_level = (
        flow.start_battery_level - flow.from_battery + flow.to_battery
    )
    return flow


def apply_arbitrage(flow: Flow, period_type: str, df_arbitrage: pd.DataFrame) -> Flow:
    # might be up to two targets per day
    target = df_arbitrage[df_arbitrage["DateTime"] == pd.Timestamp(flow.timestamp)]
    if target.empty:
        return apply_flow(flow, period_type)
    rows = df_arbitrage[df_arbitrage["DateTime"].dt.date == flow.timestamp.date()]
    target_row = target.iloc[0]
    target_hour = target_row["DateTime"].hour
    # arbitrage passed; back to flow model
    if flow.timestamp.hour > rows["DateTime"].max().hour:
        return apply_flow(flow, period_type)

    # before the arbitrage hour on a day with an arbitrage opportunity
    # send all solar to the battery
    # use the grid for demand
    demand = flow.demand
    # first charge the battery because it will be more valuable later
    flow.to_battery = min(
        BATTERY_CAPACITY - flow.start_battery_level, flow.solar_generation
    )
    # then use solar to meet demand
    available_solar = flow.solar_generation - flow.to_battery
    flow.from_solar = min(demand, available_solar)
    demand -= flow.from_solar
    if demand:
        flow.from_grid = demand
    # if it's the arbitrage hour, dump the battery to grid
    if flow.timestamp.hour == target_hour:
        max_discharge = rows["target discharge"].max()
        battery_level = flow.start_battery_level + flow.to_battery
        discharge = min(battery_level, target_row["target discharge"])
        if target_row["target discharge"] < max_discharge:
            # if this is not the best discharge error
            # keep max_discharge in battery for next hour
            max_discharge = max(battery_level - max_discharge, 0)
            discharge = min(discharge, max_discharge)
        discharge = float(discharge)
        print(
            f"arbitrage: {flow.timestamp}\tbattery\t{battery_level:.1f}"
            f"\tdischarge\t{discharge:.1f}\t{target_row["total credit"]:.1f}"
        )
        flow.to_grid = discharge
        flow.end_battery_level = battery_level - discharge
    else:
        # else just the excess
        flow.to_grid = flow.solar_generation - flow.from_solar - flow.to_battery
        # update battery level
        flow.end_battery_level = (
            flow.start_battery_level - flow.from_battery + flow.to_battery
        )
    return flow


def calculate_v3(
    df: pd.DataFrame,
    hourly_output: dict[str, float],
    df_arbitrage: pd.DataFrame,
    suffix: str = "",
) -> pd.DataFrame:
    rows = []
    # initial battery level for the model
    battery_level = BATTERY_CAPACITY / 2
    arbitrage_dates = (
        set(df_arbitrage["DateTime"].dt.date.unique())
        if not df_arbitrage.empty
        else set()
    )
    for _, row in df.iterrows():
        flow = Flow(
            timestamp=row["Timestamp"].to_pydatetime(),
            # need this much
            demand=row["demand"],
            # available solar generation
            solar_generation=hourly_output.get(
                row["Timestamp"].strftime("%m-%d %H:%M"), 0.0
            )
            * row["solar_efficiency"],
            start_battery_level=battery_level,
        )
        arbitrage_day = flow.timestamp.date() in arbitrage_dates
        if arbitrage_day:
            flow = apply_arbitrage(flow, row["period_type"], df_arbitrage)
        else:
            flow = apply_flow(flow, row["period_type"])
        battery_level = flow.end_battery_level
        # print(flow)
        rows.append(flow.model_dump(mode="json"))
    df_flows = pd.DataFrame(rows)
    # add pd.Timestamp column to df_flows from timestamp
    df_flows["Timestamp"] = pd.to_datetime(df_flows["timestamp"])
    df = df.merge(df_flows, on="Timestamp", how="left")
    # both df and df_flows have demand column; drop one
    df["demand"] = df["demand_x"]
    filename = f"{get_output_dir()}/flows_v3{suffix}.csv"
    cols = [
        "Timestamp",
        "season",
        "period",
        "period_type",
        "solar_efficiency",
        "demand",
        "start_battery_level",
        "end_battery_level",
        "solar_generation",
        "from_solar",
        "from_battery",
        "from_grid",
        "to_battery",
        "to_grid",
    ] + date_helpers()
    df = df[cols]
    df = df.sort_values(by="Timestamp")
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
        lambda x: datetime(START_YEAR, x, 1).strftime("%b")
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
    'ELEC', 'generation credit', 'delivery credit', 'bonus credit', 'pge cost ELEC', 'pge credit', 'grid cost ELEC', 'net cost ELEC', 'savings ELEC',
    'EV2-A', 'pge cost EV2-A', 'grid cost EV2-A', 'net cost EV2-A', 'savings EV2-A']
    """
    cols = [
        "Timestamp",
        "season",
        "period",
        "period_type",
        "demand",
        "start_battery_level",
        "end_battery_level",
        "solar_generation",
        "from_solar",
        "from_battery",
        "from_grid",
        "to_battery",
        "to_grid",
        "generation credit",
        "delivery credit",
        "bonus credit",
        "ELEC generation cost",
        "ELEC generation credit",
        "ELEC delivery cost",
        "ELEC delivery credit",
        # includes service charge
        "ELEC other cost",
        "ELEC grid cost",
    ]
    """
        "EV2-A",
        "EV2-A generation cost",
        "EV2-A generation credit",
        "EV2-A delivery cost",
        "EV2-A delivery credit",
        "EV2-A grid cost",

    """
    # convert Timestamp to string
    df["Timestamp"] = df["Timestamp"].astype(str)
    # keep only cols
    df = df[cols]
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
    df: pd.DataFrame,
    tariffs: dict[str, dict[str, float]],
    suffix: str = "",
) -> pd.DataFrame:
    print(f"adding costs {suffix}")
    for tariff in tariffs:
        df = set_rates(df, tariff, tariffs[tariff])
        df[f"{tariff} grid cost"] = 0.0
        for key in ["generation", "delivery"]:
            df[f"{tariff} {key} cost"] = df["from_grid"] * df[f"{tariff} {key}"]
            df[f"{tariff} {key} credit"] = df["to_grid"] * df[f"{key} credit"]
            df[f"{tariff} grid cost"] += df["demand"] * df[f"{tariff} {key}"]
        df[f"{tariff} bonus credit"] = df["to_grid"] * df["bonus credit"]
        df[f"{tariff} other cost"] = df["from_grid"] * df[f"{tariff} other"]
        df[f"{tariff} grid cost"] += df["demand"] * df[f"{tariff} other"]
        # apply credits monthly; cost does not go negative but credits rollover
    cols = [
        "Timestamp",
        "season",
        "period",
        "period_type",
        "solar_efficiency",
        "demand",
        "start_battery_level",
        "end_battery_level",
        "solar_generation",
        "from_solar",
        "from_battery",
        "from_grid",
        "to_battery",
        "to_grid",
        "ELEC generation",
        "ELEC delivery",
        "ELEC other",
        "generation credit",
        "delivery credit",
        "bonus credit",
        "ELEC generation cost",
        "ELEC generation credit",
        "ELEC delivery cost",
        "ELEC delivery credit",
        "ELEC bonus credit",
        "ELEC other cost",
        "ELEC grid cost",
    ] + date_helpers()
    """
        "EV2-A",
        "EV2-A generation cost",
        "EV2-A generation credit",
        "EV2-A delivery cost",
        "EV2-A delivery credit",
        "EV2-A grid cost",
    """
    df = df[cols]
    df.to_csv(f"{get_output_dir()}/costs{suffix}.csv", index=False)
    return df


def date_helpers():
    return ["month", "hour", "day", "yyyymm", "ymd"]


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
    df["demand"] = df["kW"]
    # fill DST hour with 0.0
    return df.fillna(0.0)


def get_output_dir():
    return OUTPUT_DIR


def print_summary(
    df: pd.DataFrame,
    monthly_costs: dict[str, pd.DataFrame],
    config: dict,
    payback_period: float,
    irr: float,
):
    lines: list[str] = []
    lines.append(f"## {label_from_params()}\n")
    lines.append(f"solar capacity: {round(SOLAR_CAPACITY)} kW")
    lines.append(f"battery capacity: {round(BATTERY_CAPACITY)} kWh")
    lines.append(f"model: {config.get('model', 'flow')}")
    lines.append(
        f"rate increase: {config.get('annual_rate_increase', ANNUAL_RATE_INCREASE)}"
    )
    lines.append(
        f"min battery level: summer={round(MIN_BATTERY_SUMMER)} kWh winter={round(MIN_BATTERY_WINTER)} kWh"
    )

    # average daily use by period type
    lines.append("## Average daily use by period\n")
    daily = df.groupby(["ymd", "period_type"]).agg({"demand": "sum"}).reset_index()
    average = daily.groupby("period_type")["demand"].mean()
    for period_type, avg_kw in average.items():
        lines.append(f"{period_type}: {round(avg_kw)} kWh")
    lines.append("\n")
    lines.append(
        "\ntariff\tPG&E\tnet cost\tsavings\t"
        "delivery credit rollover\tgeneration credit rollover\tbonus credit rollover\t"
        "used credit per exported kWh"
    )
    row = {"scenario": f"{config['description']}"}
    # for forecast tariff only
    annual: dict[str, float] = {}
    for tariff in RATE_VALUES.keys():
        df_monthly = monthly_costs[tariff]
        net_cost = df_monthly["net cost"].sum()
        grid_cost = df_monthly["grid cost"].sum()
        # rollover from last row
        last_row = df_monthly.iloc[-1]
        delivery_rollover = last_row["delivery rollover credit"]
        generation_rollover = last_row["generation rollover credit"]
        bonus_rollover = last_row["bonus rollover credit"]
        # credit used; excludes rollover
        credit_per_kwh = (
            df_monthly["delivery credit applied"].sum()
            + df_monthly["generation credit applied"].sum()
            + df_monthly["bonus credit applied"].sum()
        ) / df_monthly["to_grid"].sum()
        savings = grid_cost - net_cost
        # TODO: want actual generation cost
        row.update(
            {
                f"{tariff} grid cost": grid_cost,
                f"{tariff} net cost": net_cost,
                f"{tariff} savings": savings,
                f"{tariff} applied generation credit": df_monthly[
                    "generation credit applied"
                ].sum(),
                f"{tariff} applied delivery credit": df_monthly[
                    "delivery credit applied"
                ].sum(),
                f"{tariff} applied bonus credit": df_monthly[
                    "bonus credit applied"
                ].sum(),
                f"{tariff} delivery rollover credit": delivery_rollover,
                f"{tariff} generation rollover credit": generation_rollover,
                f"{tariff} bonus rollover credit": bonus_rollover,
                f"{tariff} credit per kWh": credit_per_kwh,
            }
        )
        if tariff == FORECAST_TARIFF:
            annual = {
                "demand": df["demand"].sum(),
                "solar": df["solar_generation"].sum(),
                "from_grid": df["from_grid"].sum(),
                "to_grid": df["to_grid"].sum(),
                "grid cost": grid_cost,
                "net cost": net_cost,
                "savings": savings,
                "applied generation credit": df_monthly[
                    "generation credit applied"
                ].sum(),
                "applied delivery credit": df_monthly["delivery credit applied"].sum(),
                "applied bonus credit": df_monthly["bonus credit applied"].sum(),
                "delivery rollover credit": delivery_rollover,
                "generation rollover credit": generation_rollover,
                "bonus rollover credit": bonus_rollover,
                "credit per kWh": credit_per_kwh,
            }
        lines.append(
            f"{tariff}\t${grid_cost:,.0f}\t${net_cost:,.0f}\t${savings:,.0f}\t"
            f"${delivery_rollover:,.0f}\t${generation_rollover:,.0f}\t"
            f"${bonus_rollover:,.0f}\t"
            f"${credit_per_kwh:,.2f}"
        )
    lines.append("\n")

    tariff = FORECAST_TARIFF
    lines.append(f"## Annual values for {tariff}\n")

    lines.append(f"demand: {annual['demand']:,.0f} kWh")
    lines.append(f"solar: {annual['solar']:,.0f} kWh")
    lines.append(f"to grid: {annual['to_grid']:,.0f} kWh")
    lines.append(
        f"applied generation credit: ${annual['applied generation credit']:,.0f}"
    )
    lines.append(
        f"generation rollover credit: ${annual['generation rollover credit']:,.0f}"
    )
    lines.append(f"applied delivery credit: ${annual['applied delivery credit']:,.0f}")
    lines.append(
        f"delivery rollover credit: ${annual['delivery rollover credit']:,.0f}"
    )
    lines.append(f"applied bonus credit: ${annual['applied bonus credit']:,.0f}")
    lines.append(f"bonus rollover credit: ${annual['bonus rollover credit']:,.0f}")
    lines.append(f"credit per kWh: ${annual['credit per kWh']:,.2f}")
    lines.append("\n")
    lines.append(f"payback period\t{payback_period:.1f} years")
    lines.append(f"IRR\t{irr:.2%}")

    row["annual rate increase"] = ANNUAL_RATE_INCREASE
    row["payback period"] = payback_period
    row["IRR"] = irr
    row["tariff"] = FORECAST_TARIFF
    # TODO: merge with Jinja template
    # arbitrage-20/costs_{01-15}.png
    # arbitrage-20/credits_{01-15}.png
    # arbitrage-20/daily_{01-12}.png
    # arbitrage-20/flow_{01-12}.png
    # arbitrage-20/monthly_sources.png
    # arbitrage-20/roi.png
    print("\n".join(lines))
    filename = f"{get_output_dir()}/summary.md"
    with open(filename, "w") as f:
        f.write("\n\n".join(lines))
    print(f"wrote summary to {filename}")
    data = {}
    data.update(annual)
    data.update(row)
    df_summary = pd.DataFrame([data])
    cols = [
        "scenario",
        "payback period",
        "IRR",
        "tariff",
        "grid cost",
        "net cost",
        "savings",
        "applied generation credit",
        "applied delivery credit",
        "applied bonus credit",
        "delivery rollover credit",
        "generation rollover credit",
        "bonus rollover credit",
        "credit per kWh",
        "demand",
        "solar",
        "from_grid",
        "to_grid",
        "annual rate increase",
        "ELEC grid cost",
        "ELEC net cost",
        "ELEC savings",
        "ELEC applied generation credit",
        "ELEC applied delivery credit",
        "ELEC delivery rollover credit",
        "ELEC generation rollover credit",
        "ELEC credit per kWh",
    ]
    """
        "EV2-A grid cost",
        "EV2-A net cost",
        "EV2-A savings",
        "EV2-A applied generation credit",
        "EV2-A applied delivery credit",
        "EV2-A delivery rollover",
        "EV2-A generation rollover",
        "EV2-A credit per kWh",

    """
    df_summary = df_summary[cols]
    filename = f"{get_output_dir()}/summary.csv"
    df_summary.to_csv(filename, index=False)
    print(f"wrote summary to {filename}")


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
        label="battery level",
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
    return LABEL


def cost_chart_setup(ax):
    x = np.arange(1, 13)
    ax.set_xticks(x)
    months = [date(START_YEAR, month, 1).strftime("%b") for month in x]
    ax.set_xticklabels(months)
    ax.legend()
    # add grid lines
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0.5, 12.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${int(x):,}"))


def calculate_monthly_costs(
    df: pd.DataFrame,
    tariff: str,
    generation_credit: float = 0.0,
    delivery_credit: float = 0.0,
    bonus_credit: float = 0.0,
) -> pd.DataFrame:
    # group by month and sum costs by tariff
    rollover = {
        "generation": generation_credit,
        "delivery": delivery_credit,
        "bonus": bonus_credit,
        # no rollover tracking but makes code simpler
        "other": 0.0,
    }
    df = df.sort_values(by="Timestamp")
    year = df["Timestamp"].dt.year.max()
    df_monthly = (
        df.groupby("month")
        .agg(
            {
                f"{tariff} generation cost": "sum",
                f"{tariff} generation credit": "sum",
                f"{tariff} delivery cost": "sum",
                f"{tariff} delivery credit": "sum",
                f"{tariff} bonus credit": "sum",
                f"{tariff} other cost": "sum",
                f"{tariff} grid cost": "sum",
                "to_grid": "sum",
            }
        )
        .reset_index()
        .sort_values(by="month")
    )

    service_charge = RATE_VALUES_MONTHLY.get(tariff, {}).get("service charge", 0.0)
    df_monthly["bonus credit applied"] = 0.0
    for idx, row in df_monthly.iterrows():
        total = 0.0
        grid_cost = 0.0
        # get number of days in this month
        days = calendar.monthrange(year, int(row["month"]))[1]
        # can't offset this
        delivery_minimum = (
            RATE_VALUES_DAILY.get(tariff, {}).get("delivery minimum", 0.0) * days
        )
        credit_used: dict[str, float] = {
            "generation": 0.0,
            "delivery": 0.0,
            "other": 0.0,
            "bonus": 0.0,
        }
        for key in ["generation", "delivery", "other"]:
            cost_key = f"{tariff} {key} cost"
            cost = row[cost_key]
            if key != "other":
                credit_key = f"{tariff} {key} credit"
                credit_available = row[credit_key] + rollover[key]
            grid_cost += cost
            net = max(cost - credit_available, 0.0)
            if key == "delivery" and net < delivery_minimum:
                net = delivery_minimum
            credit_used[key] = max(cost - net, 0)
            rollover[key] = credit_available - credit_used[key]
            # use bonus credit for generation or delivery (electricity cost)
            credit_used["bonus"] = 0.0
            credit_available = float(row[f"{tariff} bonus credit"] + rollover["bonus"])
            if credit_available and net > 0.0:
                if key == "generation":
                    credit_used["bonus"] = min(credit_available, net)
                if key == "delivery" and net > delivery_minimum:
                    credit_used["bonus"] = min(credit_available, net - delivery_minimum)
            net -= credit_used["bonus"]
            rollover["bonus"] = credit_available - credit_used["bonus"]
            total += net
            df_monthly.at[idx, f"{key} net cost"] = net
            df_monthly.at[idx, f"{key} credit applied"] = credit_used[key]
            df_monthly.at[idx, f"{key} rollover credit"] = rollover[key]
            df_monthly.at[idx, "bonus credit applied"] += credit_used["bonus"]
            df_monthly.at[idx, "bonus rollover credit"] = rollover["bonus"]
        df_monthly.at[idx, "net cost"] = total + service_charge
        df_monthly.at[idx, f"{tariff} other cost"] += service_charge
    df_monthly["grid cost"] = df_monthly[f"{tariff} grid cost"]
    return df_monthly


def chart_credits(df_monthly: pd.DataFrame, tariff: str, year: int = 1):
    # credits
    fig, ax = plt.subplots(figsize=(12, 6))
    gen_used = df_monthly["generation credit applied"]
    gen_created = df_monthly[f"{tariff} generation credit"]
    bonus_used = df_monthly["bonus credit applied"]
    bonus_created = df_monthly[f"{tariff} bonus credit"]
    delivery_used = df_monthly["delivery credit applied"]
    delivery_created = df_monthly[f"{tariff} delivery credit"]
    df_monthly["credits used"] = gen_used + delivery_used + bonus_used
    df_monthly["credits created"] = gen_created + delivery_created + bonus_created
    # credit applied: below the axis since it's a bill credit
    ax.bar(
        df_monthly["month"],
        -delivery_used,
        color=COLORS["delivery"],
        label="delivery credit applied",
    )
    ax.bar(
        df_monthly["month"],
        -gen_used,
        color=COLORS["generation"],
        bottom=-delivery_used,
        label="generation credit applied",
    )
    ax.bar(
        df_monthly["month"],
        -bonus_used,
        color=COLORS["bonus"],
        bottom=-(delivery_used + gen_used),
        label="bonus credit applied",
    )
    # credit generation: positive
    ax.bar(
        df_monthly["month"],
        delivery_created,
        color=COLORS["delivery"],
        label="delivery export credit",
        alpha=0.6,
    )
    ax.bar(
        df_monthly["month"],
        gen_created,
        bottom=delivery_created,
        color=COLORS["generation"],
        label="generation export credit",
        alpha=0.6,
    )
    ax.bar(
        df_monthly["month"],
        bonus_created,
        color=COLORS["bonus"],
        bottom=(delivery_created + gen_created),
        label="bonus export credit",
        alpha=0.6,
    )
    df_monthly["credit per kWh"] = df_monthly["credits created"] / df_monthly["to_grid"]
    df_monthly["credit per kWh"] = df_monthly["credit per kWh"].fillna(0.0)
    ax.set_title(f"{label_from_params()} export credit: year {year}")
    last_row = df_monthly.iloc[-1]
    gen_credit_used = round(gen_used.sum())
    gen_credit_created = round(gen_created.sum())
    gen_credit_usage = 0.0
    if gen_credit_created:
        gen_credit_usage = round(gen_credit_used / gen_credit_created * 100)
    gen_credit = round(last_row["generation rollover credit"])
    delivery_credit = round(last_row["delivery rollover credit"])
    bonus_credit = round(last_row["bonus rollover credit"])
    text = (
        f"{gen_credit_usage}% generation credits used\n"
        f"${gen_credit:,.0f} generation rollover credit\n"
        f"${delivery_credit:,.0f} delivery rollover credit\n"
        f"${bonus_credit:,.0f} bonus rollover credit\n"
    )
    plt.text(
        0.15,  # x-coordinate
        0.9,  # y-coordinate
        text,
        ha="center",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=12,
        color="black",
    )
    ax.set_ylabel("credit $")
    y_min = -df_monthly["credits used"].max()
    y_max = df_monthly["credits created"].max()
    ax.set_ylim(y_min * 1.1, y_max * 1.2)
    cost_chart_setup(ax)
    ax.legend()
    # add value of df_monthly["to_grid"] for each bar
    for _, row in df_monthly.iterrows():
        text = (
            f"{row['to_grid']:.0f} kWh\n"  # export kWh
            f"{row['credit per kWh']:.2f}/kWh"  # credit per kWh
        )
        plt.text(
            row["month"],
            row["credits created"],
            text,
            fontsize=9,
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    filename = f"{get_output_dir()}/credits_{year:02d}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"wrote credits to {filename}")
    plt.close(fig)


def chart_costs(df_monthly: pd.DataFrame, tariff: str, year: int = 1):
    """Create chart with side by side bars for pge cost ELEC and net cost ELEC

    Add total savings for the year as a text box.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    other_cost = df_monthly[f"{tariff} other cost"]
    cost = round(other_cost.sum())
    ax.bar(
        df_monthly["month"],
        other_cost,
        label=f"solar + battery other ${cost:,.0f}",
        color=COLORS["other"],
    )
    delivery_cost = df_monthly["delivery net cost"]
    cost = round(delivery_cost.sum())
    ax.bar(
        df_monthly["month"],
        delivery_cost,
        bottom=other_cost,
        label=f"solar + battery delivery ${cost:,.0f}",
        color=COLORS["delivery"],
    )
    generation_cost = df_monthly["generation net cost"]
    cost = round(generation_cost.sum())
    ax.bar(
        df_monthly["month"],
        generation_cost,
        bottom=delivery_cost + other_cost,
        label=f"solar + battery generation ${cost:,.0f}",
        color=COLORS["generation"],
    )
    credit = (
        df_monthly["delivery credit applied"]
        + df_monthly["generation credit applied"]
        + df_monthly["bonus credit applied"]
    )
    credit_applied = round(credit.sum())
    ax.plot(
        df_monthly["month"],
        credit,
        label=f"export credit applied ${credit_applied:,.0f}",
        color=COLORS["from_grid"],
        # dotted line
        linestyle="--",
    )
    cost_chart_setup(ax)
    ax.set_ylabel("cost $")
    grid_cost = round(df_monthly[f"{tariff} grid cost"].sum())
    solar_cost = round(df_monthly["net cost"].sum())
    delivery_credit = round(df_monthly.iloc[11]["delivery rollover credit"])
    generation_credit = round(df_monthly.iloc[11]["generation rollover credit"])
    bonus_credit = round(df_monthly.iloc[11]["bonus rollover credit"])
    savings_pct = round(solar_cost / grid_cost * 100)
    generational_total = df_monthly[f"{tariff} generation cost"].sum()
    # group by month, sum net cost {tariff}
    monthly_bill = solar_cost / 12
    ax.set_title(f"{tariff} cost: {label_from_params()} year {year}")

    text = (
        # f"${total_savings:,.0f} annual savings\n"
        f"${monthly_bill:,.0f} average monthly bill\n"
        f"${generational_total:,.0f} generation cost\n"
        f"${generation_credit:,.0f} generation rollover credit\n"
        f"{savings_pct}% of PG&E cost\n"
    )
    if delivery_credit:
        text += f"${delivery_credit:,.0f} delivery rollover credit\n"
    if bonus_credit:
        text += f"${bonus_credit:,.0f} bonus rollover credit\n"
    plt.text(
        0.85,
        0.95,
        text,
        ha="center",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=12,
        color="black",
    )
    plt.tight_layout()
    filename = f"{get_output_dir()}/costs_{year:02d}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"wrote costs to {filename}")
    plt.close(fig)


def chart_roi(df: pd.DataFrame, payback_period: float, irr: float):
    """Chart cumulative savings line"""
    df["year"] = pd.to_datetime(df["year_month"]).dt.year
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        df["year_month"],
        df["cumulative savings"],
        color=COLORS["savings"],
        label="savings",
        linestyle="--",
    )
    ax.plot(
        df["year_month"], df["cumulative pge"], color=COLORS["from_grid"], label="PG&E"
    )
    ax.plot(
        df["year_month"],
        df["cumulative net"],
        color=COLORS["system"],
        label="solar + battery",
    )
    ax.set_title(f"Cumulative: {label_from_params()}")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(df["year_month"].min(), df["year_month"].max())
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


def calculate_roi(df: pd.DataFrame, df_arbitrage: pd.DataFrame, config: dict):
    """
    Timestamp,kW,hour,yyyymm,month,day,ymd,season,period,period_type
    """
    current_year = START_YEAR
    years = 15
    hourly_output = load_hourly_output()
    df_years = pd.DataFrame()
    rates = {}
    rates.update(RATE_VALUES)
    solar_degradation_factor = SOLAR_DEGRADATION_FACTOR
    solar_efficiency = 1.0
    tariff = FORECAST_TARIFF
    rollover_generation = 0.0
    rollover_delivery = 0.0
    rollover_bonus = 0.0
    for year in range(years):
        if year > 0:
            # reduce the solar generation by the degradation factor
            solar_efficiency = (1 - solar_degradation_factor) ** year
            # increase the per kwH cost by 4%
            for tariff in rates:
                for period in rates[tariff]:
                    charges = rates[tariff][period]
                    new_charges = Charges(
                        generation=charges.generation
                        + (charges.generation * ANNUAL_RATE_INCREASE),
                        delivery=charges.delivery
                        + (charges.delivery * ANNUAL_RATE_INCREASE),
                        other=charges.other + (charges.other * ANNUAL_RATE_INCREASE),
                    )
                    rates[tariff][period] = new_charges
        print(
            f"year {year} solar efficiency: {solar_efficiency} off peak rate: {rates[FORECAST_TARIFF]['winter off peak']}"
        )
        if year == 3:
            print("check on year 3 credits")
        df_year = df.copy()
        df_year["Timestamp"] = df["Timestamp"] + pd.Timedelta(days=365 * year)
        df_year["solar_efficiency"] = solar_efficiency
        suffix = f"_year_{year:02d}"
        df_year = calculate_v3(
            df_year,
            hourly_output,
            df_arbitrage,
            suffix=suffix,
        )
        df_year = add_costs(df_year, rates, suffix)
        df_year = calculate_monthly_costs(
            df_year,
            tariff,
            rollover_generation,
            rollover_delivery,
            rollover_bonus,
        )
        chart_costs(df_year, FORECAST_TARIFF, year + 1)
        chart_credits(df_year, FORECAST_TARIFF, year + 1)
        row = df_year.iloc[-1]
        rollover_generation = row["generation rollover credit"]
        rollover_delivery = row["delivery rollover credit"]
        rollover_bonus = row["bonus rollover credit"]
        print(
            f"{year}: generation rollover credit ${rollover_generation:,.0f}\tdelivery rollover credit ${rollover_delivery:,.0f}"
        )
        df_year["year_month"] = df_year["month"].apply(
            lambda x: date(current_year, x, 1)
        )

        df_years = pd.concat([df_years, df_year])
        current_year += 1

    df_years["savings"] = df_years["grid cost"] - df_years["net cost"]
    df_years["cumulative savings"] = (
        df_years["savings"].cumsum() - config["system_cost"]
    )
    df_years["cumulative pge"] = df_years[f"{tariff} grid cost"].cumsum()
    df_years["cumulative net"] = df_years["net cost"].cumsum()

    # payback period: first month where cumulative savings is positive
    payback_dt = df_years[df_years["cumulative savings"] > 0]["year_month"].iloc[0]
    min_timestamp = df["Timestamp"].min().date()
    years = (payback_dt - min_timestamp).days / 365
    print(f"payback period: {payback_dt} ({years:.1f} years)")
    irr = numpy_financial.irr([-config["system_cost"]] + df_years["savings"].to_list())
    print(f"IRR: {irr:.2%}")
    cols = [
        "month",
        "year_month",
        "to_grid",
        "generation net cost",
        "generation credit applied",
        "generation rollover credit",
        "delivery net cost",
        "delivery credit applied",
        "delivery rollover credit",
        "bonus credit applied",
        "bonus rollover credit",
        "grid cost",
        "net cost",
        "savings",
        "cumulative savings",
        "cumulative pge",
        "cumulative net",
    ]
    df_years = df_years[cols]
    df_years.to_csv(f"{get_output_dir()}/roi.csv", index=False)
    return df_years, years, irr


def load_arbitrage_targets(arbitrage_discharge: float) -> pd.DataFrame:
    max_discharge = min(BATTERY_CAPACITY * BATTERY_DISCHARGE_RATIO, arbitrage_discharge)
    remainder_discharge = min(0, arbitrage_discharge - max_discharge)
    df = pd.read_csv("data/arbitrage_targets.csv", parse_dates=["DateTime"])
    # sort by total_credit descending then by DateTime ascending
    df = df.sort_values(by=["total credit", "DateTime"], ascending=[False, True])
    df["target discharge"] = remainder_discharge
    # group by date and iterate over each group
    for dt, group in df.groupby(df["DateTime"].dt.date):
        row = group.iloc[0]
        df.loc[df["DateTime"] == row["DateTime"], "target discharge"] = max_discharge
    return df


def calculate_tariff_monthly_costs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    monthly_costs: dict[str, pd.DataFrame] = {}
    for tariff in RATE_VALUES.keys():
        df_monthly = calculate_monthly_costs(df, tariff)
        cols = [
            "month",
            f"{tariff} generation cost",
            f"{tariff} generation credit",
            f"{tariff} delivery cost",
            f"{tariff} delivery credit",
            f"{tariff} bonus credit",
            f"{tariff} other cost",
            f"{tariff} grid cost",
            "to_grid",
            "generation net cost",
            "generation credit applied",
            "generation rollover credit",
            "delivery net cost",
            "delivery credit applied",
            "delivery rollover credit",
            "bonus credit applied",
            "bonus rollover credit",
            "net cost",
            "grid cost",
        ]
        df_monthly = df_monthly[cols]
        df_monthly.to_csv(f"{get_output_dir()}/monthly_costs_{tariff}.csv", index=False)
        monthly_costs[tariff] = df_monthly
    return monthly_costs


def main(config: dict):
    df_initial = initial_setup(config.get("usage", DEFAULT_USAGE))
    df_initial = label_periods(df_initial)
    df_initial["solar_efficiency"] = 1.0
    cols = [
        "Timestamp",
        "demand",
        "season",
        "period",
        "period_type",
        "solar_efficiency",
    ] + date_helpers()
    df_initial = df_initial[cols]
    df_initial.to_csv(f"{get_output_dir()}/initial.csv", index=False)
    hourly_output = load_hourly_output()

    if config.get("model", "flow") == "arbitrage":
        df_arbitrage = load_arbitrage_targets(
            config.get("arbitrage_discharge", BATTERY_CAPACITY)
        )
    else:
        df_arbitrage = pd.DataFrame()

    # year 1 flows
    df = calculate_v3(df_initial, hourly_output, df_arbitrage)
    df = add_costs(df, RATE_VALUES)
    # calculate monthly costs for all tariffs
    monthly_costs = calculate_tariff_monthly_costs(df)
    df_monthly = monthly_costs[FORECAST_TARIFF]
    chart_costs(df_monthly, FORECAST_TARIFF)
    chart_credits(df_monthly, FORECAST_TARIFF)

    df_roi, payback_period, irr = calculate_roi(df_initial, df_arbitrage, config)
    chart_roi(df_roi, payback_period, irr)
    print_summary(df, monthly_costs, config, payback_period, irr)

    chart_monthly_sources(df)

    max_y = df[["demand", "solar_generation"]].max().max() * 1.1
    # chart for each month
    for month, group in df.groupby(df["month"]):
        chart_flows(group, date(START_YEAR, month, 1), max_y=max_y)
        chart_daily(group, date(START_YEAR, month, 1))

    # chart_solar_hourly()
    write_to_sheet(df, config["output"])


def load_config(filename: str):
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, default="solar-10-battery-30.yml", nargs="?"
    )
    args = parser.parse_args()
    _config = load_config(args.config)
    set_params(_config)
    main(_config)
