import argparse
import csv
from datetime import date, datetime
import os
from typing import Any, Optional
import calendar
from dataclasses import dataclass

import gspread
from gspread import Spreadsheet
from gspread.utils import ValueInputOption
import jinja2
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

r"""
outputs

costs_{01-15}.png - costs by month, 15 years
credits_{01-15}.png - credit by month, 15 years
daily_{01-12}.png - sources by day per month
flow_{01-12}.png - sources by hour per month, with battery
monthly_sources.png - sources by month
roi.png - cumulative savings by month, 15 years
"""

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
Note: In states such as California, regulations prohibit batteries from discharging
to the grid if the grid was used to charge the batteries.

Credits only apply to generation:
April bill: 435.52
163.65 generation (38%) 271.87 T&D

generation fraction = 0.33 from recent bills
"""

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

START_YEAR = 2026
ROI_YEARS = 15

# 0.4% degredation per year
SOLAR_DEGRADATION_FACTOR = 0.004

# capacity value used to generate hourly output
# from https://www.renewables.ninja
MODELED_SOLAR_CAPACITY = 10.0

# usage during peak hours (kWh)
PEAK_USAGE = 10.0
DAILY_USE = 43.0
FORECAST_TARIFF = "ELEC"

SHEET_ID = os.getenv("SHEET_ID")

FIRST_PEAK_HOUR = 15
# max 10 kW per 15 kWh battery
BATTERY_DISCHARGE_RATIO = 10 / 15


@dataclass
class Config:
    """Set default values, override with yml config file."""

    # solar capacity for this run
    solar_capacity: float = 10
    battery_capacity: float = 30
    # use battery in off peak if it's at least this level
    # might need all daily solar production to offset peak usage
    min_battery_winter: float = 10
    # in summer, let battery go lower since it will likely refill
    min_battery_summer: float = 2.5
    annual_rate_increase: float = 0.04
    usage_filename: str = "data/usage.csv"
    output_dir: str = "output"
    model: str = "flow"
    arbitrage_discharge: Optional[float] = None
    description: str = "default"
    system_cost: float = 1

    @property
    def label(self) -> str:
        return f"{round(self.solar_capacity)} kW solar, {round(self.battery_capacity)} kWh battery"


config = Config()


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
    return month.strftime("%b")


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


def chart_solar_hourly_by_month(df: pd.DataFrame, max_y: float):
    """Group by day. Plot one line per day with hour on x-axis and electricity on y-axis."""
    # get min, max, and median
    per_day = df.groupby("day").agg({"electricity": "sum"})
    min_val = round(per_day["electricity"].min())
    max_val = round(per_day["electricity"].max())
    median_val = round(per_day["electricity"].median())
    days_over_use = per_day[per_day["electricity"] >= DAILY_USE].shape[0]
    over_pct = round(days_over_use / len(per_day) * 100)

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
    plt.xticks(range(0, 24, 1))
    plt.title("Average hourly solar output")
    plt.xlabel("Hour")
    plt.ylabel("kW")
    highlight_peak(plt.gca())
    row = df.iloc[0]
    month_name = row["month_label"]
    month_num = row["month"]
    month = date(START_YEAR, month_num, 1)

    # add month name as title
    plt.title(f"Hourly solar output {month_name}:  {round(config.solar_capacity)} kW")
    filename = f"{config.output_dir}/solar_hourly_{month_label(month)}.png"
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
    # scale modeled solar capacity to config.solar_capacity
    df["electricity"] = (
        df["electricity"] * config.solar_capacity / MODELED_SOLAR_CAPACITY
    )
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
    plt.title(f"Average hourly solar output: {round(config.solar_capacity)} kW system")
    plt.xlabel("Hour")
    plt.ylabel("kW")
    plt.legend(labelcolor=label_colors)
    # Set legend text color to red for under_months
    filename = f"{config.output_dir}/solar_all_months.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {filename}")
    # write data for use by other charts
    by_month.to_csv(f"{config.output_dir}/solar_hourly_by_month.csv", index=False)
    cols = ["hour", "electricity"]
    by_month = by_month[cols]
    # create grid of all months
    os.system(
        f"cd {config.output_dir}; "
        "montage solar_hourly_*.png -tile 3x4 -geometry +2+2 solar_monthly.png; "
        "cd .."
    )


def load_export():
    return pd.read_csv("data/pge-export-pt.csv", parse_dates=["DateTime"])


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
        # scale modeled solar capacity to config.solar_capacity
        output[hour] = (
            float(row["electricity"]) * config.solar_capacity / MODELED_SOLAR_CAPACITY
        )
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
        df.loc[row.name, "battery percent"] = (
            current_battery_level / config.battery_capacity
        )
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
        return config.min_battery_summer
    return config.min_battery_winter


def apply_flow(flow: Flow, period_type: str) -> Flow:
    """Set values in flow model for this period type."""
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
            flow.to_battery = min(
                config.battery_capacity - flow.start_battery_level, excess
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
                excess, config.battery_capacity - flow.start_battery_level
            )
    # to grid
    flow.to_grid = flow.solar_generation - flow.from_solar - flow.to_battery
    # update battery level
    flow.end_battery_level = (
        flow.start_battery_level - flow.from_battery + flow.to_battery
    )
    return flow


def apply_arbitrage(flow: Flow, period_type: str, df_arbitrage: pd.DataFrame) -> Flow:
    """Set values in flow model for arbitrage."""
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
        config.battery_capacity - flow.start_battery_level, flow.solar_generation
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
            # if this is not the best discharge hour
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
    battery_level = config.battery_capacity / 2
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
        rows.append(flow.model_dump(mode="json"))
    df_flows = pd.DataFrame(rows)
    # add pd.Timestamp column to df_flows from timestamp
    df_flows["Timestamp"] = pd.to_datetime(df_flows["timestamp"])
    df = df.merge(df_flows, on="Timestamp", how="left")
    # both df and df_flows have demand column; drop one
    df["demand"] = df["demand_x"]
    filename = f"{config.output_dir}/flows_v3{suffix}.csv"
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
        f"Sources by month:  {round(config.solar_capacity)} kW solar, "
        f"{round(config.battery_capacity)} kWh battery"
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    plt.tight_layout()
    fig.savefig(
        f"{config.output_dir}/monthly_sources.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def highlight_peak(ax: plt.Axes):
    """Add shading for peak and part peak hours."""
    ax.set_xlim(-0.5, 23.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    # part peak: with light yellow color for hour=15, hour=21-23
    ax.axvspan(15, 24, color=COLORS["part peak"], alpha=1.0)
    # peak: full height bar with light red color for hour=16-20
    ax.axvspan(16, 20, color=COLORS["peak"], alpha=0.5)


def chart_flows(df: pd.DataFrame, month: date, max_y: float):
    """Chart flows for a single month.

    positive: from_solar + from_battery + from_grid = demand
    negative: to_battery + to_grid = excess
    solar generation, demand
    battery level
    """
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
    axs[1].set_ylim(0, config.battery_capacity * 1.05)
    plt.tight_layout()
    plt.suptitle(f"{month.strftime('%B')} hourly: {config.label}", y=1.00)
    fig.savefig(
        f"{config.output_dir}/flow_{month_label(month)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def open_spreadsheet() -> Spreadsheet:
    gc = gspread.auth.service_account(filename="google.json")
    return gc.open_by_key(SHEET_ID)


def write_to_sheet(df: pd.DataFrame, sheet_name: str):
    """Use gspread to write to Google Sheet."""
    if not SHEET_ID:
        print("SHEET_ID not set, skipping Google Sheet write")
        return
    print("writing to Google Sheet")
    """
    ['Timestamp', 'kW', 'month', 'hour', 'yyyymm', 'ymd', 'season', 'period', 'period_type',
    'timestamp', 'demand', 'start_battery_level', 'end_battery_level',
    'solar_generation', 'from_solar',
    'from_battery', 'from_grid', 'to_battery', 'to_grid',
    'ELEC', 'generation credit', 'delivery credit', 'bonus credit', 'pge cost ELEC',
    'pge credit', 'grid cost ELEC', 'net cost ELEC', 'savings ELEC',
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
    df.to_csv(f"{config.output_dir}/flows-model.csv", index=False)
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
    df.to_csv(f"{config.output_dir}/costs{suffix}.csv", index=False)
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


def generate_summary(
    df_years: pd.DataFrame,
    demand: float,
    payback_period: float,
    irr: float,
):
    context = {
        "description": config.description,
        "solar": f"{round(config.solar_capacity):,.0f}",
        "battery": f"{round(config.battery_capacity)}",
        "rate_increase": f"{round(config.annual_rate_increase*100)}",
        "min_battery_summer": f"{round(config.min_battery_summer)}",
        "min_battery_winter": f"{round(config.min_battery_winter)}",
        "system_cost": f"{config.system_cost:,.0f}",
        "model": f"{config.model}",
        "arbitrage_discharge": f"{config.arbitrage_discharge}",
        "payback_period": f"{payback_period:.1f}",
        "irr": f"{irr:.2%}",
        "path": config.output_dir,
        "tariff": FORECAST_TARIFF,
        "years": int(len(df_years.index) / 12),
        "demand": f"{demand:,.0f}",
    }
    """ df_years = one row per month
    ['month', 'year_month', 'to_grid', 'generation net cost',
            'generation credit applied', 'generation rollover credit',
            'delivery net cost', 'delivery credit applied',
            'delivery rollover credit', 'bonus credit applied',
            'bonus rollover credit', 'grid cost', 'net cost', 'savings',
            'cumulative savings', 'cumulative pge', 'cumulative net', 'year']
    """
    # year 2 results to apply summer credits
    df_year2 = df_years.iloc[13:24]
    # rollover from last row of df_year2
    last_row = df_year2.iloc[-1]
    delivery_credit = df_year2["delivery credit applied"].sum()
    total_delivery_credit = last_row["delivery rollover credit"] + delivery_credit
    generation_credit = df_year2["generation credit applied"].sum()
    total_generation_credit = last_row["generation rollover credit"] + generation_credit
    to_grid = df_year2["to_grid"].sum()
    from_grid = df_year2["from_grid"].sum()
    credit_per_kwh = (total_delivery_credit + total_generation_credit) / to_grid
    # excludes rollover (unused)
    effective_credit_per_kwh = (delivery_credit + generation_credit) / to_grid
    cost_per_kwh = df_year2["net cost"].sum() / demand
    context.update(
        {
            "grid_cost": f"{df_year2["grid cost"].sum():,.0f}",
            "net_cost": f"{df_year2["net cost"].sum():,.0f}",
            "savings": f"{df_year2["savings"].sum():,.0f}",
            "credit_per_kwh": f"{credit_per_kwh:,.2f}",
            "cost_per_kwh": f"{cost_per_kwh:,.4f}",
            "effective_credit_per_kwh": f"{effective_credit_per_kwh:,.2f}",
            "to_grid": f"{to_grid:,.0f}",
            "from_grid": f"{from_grid:,.0f}",
            "from_grid_pct": f"{round(from_grid / demand * 100):.0f}",
            "generation_credit_applied": f"{generation_credit:,.0f}",
            "generation_credit_rollover": f"{last_row['generation rollover credit']:,.0f}",
            "delivery_credit_applied": f"{delivery_credit:,.0f}",
            "delivery_credit_rollover": f"{last_row['delivery rollover credit']:,.0f}",
            "bonus_credit_applied": f"{df_year2['bonus credit applied'].sum():,.0f}",
            "bonus_credit_rollover": f"{last_row['bonus rollover credit']:,.0f}",
        }
    )
    # load summary.jinja2
    with open("summary.jinja2", "r") as f:
        template = f.read()
    # render a template with Jinja2
    rendered = jinja2.Template(template).render(context)
    # write to summary.md
    with open(f"{config.output_dir}/summary.md", "w") as f:
        f.write(rendered)
    print(f"wrote summary to {config.output_dir}/summary.md")


def write_summary_csv(
    df: pd.DataFrame,
    monthly_costs: dict[str, pd.DataFrame],
    payback_period: float,
    irr: float,
):
    row = {
        "scenario": config.description,
        "annual rate increase": config.annual_rate_increase,
        "payback period": payback_period,
        "IRR": irr,
        "tariff": FORECAST_TARIFF,
    }
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
        if tariff != FORECAST_TARIFF:
            continue
        row.update(
            {
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
        )

    df_summary = pd.DataFrame([row])
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
    filename = f"{config.output_dir}/summary.csv"
    df_summary.to_csv(filename, index=False)
    print(f"wrote summary to {filename}")


def source_label(df: pd.DataFrame, col: str) -> int:
    percent = round(df[col].sum() / df["demand"].sum() * 100)
    return f"({percent}%)"


def chart_daily(df: pd.DataFrame, month: date):
    print(f"writing daily chart for {month.strftime('%B')}")
    df = df.sort_values(by="Timestamp")
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
    ax.set_title(f"{month.strftime('%B')}: {config.label}")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    plt.tight_layout()
    # plt.show()
    ax.legend()
    fig.savefig(
        f"{config.output_dir}/daily_{month_label(month)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    """Group by month and sum costs by tariff."""
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
                "from_grid": "sum",
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
    ax.set_title(f"{config.label} export credit: year {year}")
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
    filename = f"{config.output_dir}/credits_{year:02d}.png"
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
    # credits
    bonus_credit = df_monthly["bonus credit applied"]
    credit_value = round(bonus_credit.sum())
    ax.bar(
        df_monthly["month"],
        bonus_credit,
        bottom=delivery_cost + other_cost + generation_cost,
        label=f"bonus credit applied ${credit_value:,.0f}",
        color=COLORS["bonus"],
        alpha=0.2,
        hatch="//",
        edgecolor="gray",
    )
    delivery_credit = df_monthly["delivery credit applied"]
    credit_value = round(delivery_credit.sum())
    ax.bar(
        df_monthly["month"],
        delivery_credit,
        bottom=delivery_cost + other_cost + generation_cost + bonus_credit,
        label=f"delivery credit applied ${credit_value:,.0f}",
        color=COLORS["delivery"],
        alpha=0.2,
        hatch="//",
        edgecolor="gray",
    )
    generation_credit = df_monthly["generation credit applied"]
    credit_value = round(generation_credit.sum())
    ax.bar(
        df_monthly["month"],
        generation_credit,
        bottom=delivery_cost
        + other_cost
        + generation_cost
        + bonus_credit
        + delivery_credit,
        label=f"generation credit applied ${credit_value:,.0f}",
        color=COLORS["generation"],
        alpha=0.2,
        hatch="//",
        edgecolor="gray",
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
    arbitrage = (
        f" {round(config.arbitrage_discharge)} kWh arbitrage"
        if config.arbitrage_discharge
        else ""
    )
    title = f"{tariff} cost: {config.label}{arbitrage} year {year}"
    print(f"writing {title}")
    ax.set_title(title)

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
    filename = f"{config.output_dir}/costs_{year:02d}.png"
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
    ax.set_title(f"Cumulative: {config.label}")
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
    filename = f"{config.output_dir}/roi.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"wrote {filename}")
    plt.close(fig)


def calculate_roi(
    df: pd.DataFrame, df_arbitrage: pd.DataFrame
) -> tuple[pd.DataFrame, float, float]:
    """Run model for ROI_YEARS years and calculate ROI.

    Apply annual rate increase and solar degradation factor;
    rollover credits from previous year.

    Timestamp,kW,hour,yyyymm,month,day,ymd,season,period,period_type
    """
    current_year = START_YEAR
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
    for year in range(ROI_YEARS):
        if year > 0:
            # reduce the solar generation by the degradation factor
            solar_efficiency = (1 - solar_degradation_factor) ** year
            # increase the per kwH cost by annual rate increase
            for tariff in rates:
                for period in rates[tariff]:
                    charges = rates[tariff][period]
                    new_charges = Charges(
                        generation=charges.generation
                        + (charges.generation * config.annual_rate_increase),
                        delivery=charges.delivery
                        + (charges.delivery * config.annual_rate_increase),
                        other=charges.other
                        + (charges.other * config.annual_rate_increase),
                    )
                    rates[tariff][period] = new_charges
        print(
            f"year {year} solar efficiency: {solar_efficiency} "
            f"off peak rate: {rates[FORECAST_TARIFF]['winter off peak']}"
        )
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
            f"{year}: generation rollover credit ${rollover_generation:,.0f}\t"
            f"delivery rollover credit ${rollover_delivery:,.0f}"
        )
        df_year["year_month"] = df_year["month"].apply(
            lambda x: date(current_year, x, 1)
        )

        df_years = pd.concat([df_years, df_year])
        current_year += 1

    df_years["savings"] = df_years["grid cost"] - df_years["net cost"]
    df_years["cumulative savings"] = df_years["savings"].cumsum() - config.system_cost
    df_years["cumulative pge"] = df_years[f"{tariff} grid cost"].cumsum()
    df_years["cumulative net"] = df_years["net cost"].cumsum()

    # payback period: first month where cumulative savings is positive
    payback_dt = df_years[df_years["cumulative savings"] > 0]["year_month"].iloc[0]
    min_timestamp = df["Timestamp"].min().date()
    years = (payback_dt - min_timestamp).days / 365
    print(f"payback period: {payback_dt} ({years:.1f} years)")
    irr = numpy_financial.irr([-config.system_cost] + df_years["savings"].to_list())
    print(f"IRR: {irr:.2%}")
    cols = [
        "month",
        "year_month",
        "to_grid",
        "from_grid",
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
    df_years.to_csv(f"{config.output_dir}/roi.csv", index=False)
    return df_years, years, irr


def load_arbitrage_targets(arbitrage_discharge: float) -> pd.DataFrame:
    """Load arbitrage targets and set target discharge."""
    max_discharge = min(
        config.battery_capacity * BATTERY_DISCHARGE_RATIO, arbitrage_discharge
    )
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
        df_monthly.to_csv(
            f"{config.output_dir}/monthly_costs_{tariff}.csv", index=False
        )
        monthly_costs[tariff] = df_monthly
    return monthly_costs


def main():
    df_initial = initial_setup(config.usage_filename)
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
    df_initial.to_csv(f"{config.output_dir}/initial.csv", index=False)
    hourly_output = load_hourly_output()

    if config.model == "arbitrage":
        df_arbitrage = load_arbitrage_targets(config.arbitrage_discharge)
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

    df_roi, payback_period, irr = calculate_roi(df_initial, df_arbitrage)
    chart_roi(df_roi, payback_period, irr)
    write_summary_csv(df, monthly_costs, payback_period, irr)
    generate_summary(df_roi, df_initial["demand"].sum(), payback_period, irr)

    chart_monthly_sources(df)

    max_y = df[["demand", "solar_generation"]].max().max() * 1.1
    # chart for each month
    for month, group in df.groupby(df["month"]):
        chart_flows(group, date(START_YEAR, month, 1), max_y=max_y)
        chart_daily(group, date(START_YEAR, month, 1))

    chart_solar_hourly()
    write_to_sheet(df, config.output_dir)
    print(f"\nwrote outputs to {config.output_dir}")


def load_config(filename: str):
    with open(filename, "r") as f:
        config_dict = yaml.safe_load(f)
    config.output_dir = config_dict.get("output", config.output_dir)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    config.annual_rate_increase = config_dict.get(
        "annual_rate_increase", config.annual_rate_increase
    )
    config.arbitrage_discharge = config_dict.get(
        "arbitrage_discharge", config.arbitrage_discharge
    )
    config.battery_capacity = config_dict.get("battery", config.battery_capacity)
    config.description = config_dict.get("description", config.description)
    config.min_battery_winter = config_dict.get(
        "min_battery_winter", config.min_battery_winter
    )
    config.min_battery_summer = config_dict.get(
        "min_battery_summer", config.min_battery_summer
    )
    config.model = config_dict.get("model", config.model)
    config.output_dir = config_dict.get("output", config.output_dir)
    config.solar_capacity = config_dict.get("solar", config.solar_capacity)
    config.usage_filename = config_dict.get("usage_filename", config.usage_filename)
    config.system_cost = config_dict.get("system_cost", config.system_cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, default="solar-10-battery-30.yml", nargs="?"
    )
    args = parser.parse_args()
    load_config(args.config)
    main()
