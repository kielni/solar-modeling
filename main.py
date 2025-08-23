import argparse
import os
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import csv
import pandas as pd
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field
import gspread
from gspread import Spreadsheet
from gspread.utils import ValueInputOption


# https://www.pge.com/assets/pge/docs/account/rate-plans/residential-electric-rate-plan-pricing.pdf

"""
solar panels
https://connect.soligent.net/site/Item%20Documents/7739_110-3006%20Data%20Sheet.pdf
- 1.3–1.8 kWh/day
First year degradation: 1%
· Linear warranty after initial year:
 with 0.4%p annual degradation,
 87.4% is guaranteed up to 30 years
"""
"""
https://www.franklinwh.com/products/apower2-home-battery-backup/
15 year warranty
2x  15kWh batteries = 27,300
$910 per kWh
"""

# capacity value used to generate hourly output
# from https://www.renewables.ninja
BASE_SOLAR_CAPACITY = 10
# scale by command line args
SOLAR_CAPACITY = BASE_SOLAR_CAPACITY
BASE_BATTERY_CAPACITY = 30
# scale by command line args
BATTERY_CAPACITY = BASE_BATTERY_CAPACITY

# use battery in off peak if it's at least this level
OFF_PEAK_BATTERY_LEVEL = 10
# average is 10 but add a little margin
OFF_PEAK_USAGE = 11
DAILY_USE = 42

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


def set_capacity(solar_kw: float, battery_kwh: float):
    global BATTERY_CAPACITY
    global SOLAR_CAPACITY
    BATTERY_CAPACITY = BASE_BATTERY_CAPACITY * battery_kwh / BASE_BATTERY_CAPACITY
    SOLAR_CAPACITY = BASE_SOLAR_CAPACITY * solar_kw / BASE_SOLAR_CAPACITY


def set_rates(df: pd.DataFrame, tariff: str) -> pd.DataFrame:
    """Set rate per kWh from period and tariff"""
    for period, rate in RATE_VALUES[tariff].items():
        df.loc[df["period"] == period, tariff] = rate
    # credit of 0.03 for excess output
    df["credit per kWh"] = -0.03
    df["service charge"] = 0.0
    # ELEC has a service charge of $15 per billing period
    # approximate hourly to make the calculations easier
    df["service charge"] = (15 * 12) / (365 * 24)
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
    df["kW ev"] = df["kW"]
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


def load_output() -> dict[str, float]:
    """Load hourly output estimate from output.csv

    day,kW
    01-01,49.18
    from https://www.renewables.ninja
    """
    output: dict[str, float] = {}
    with open("output.csv") as f:
        for line in f.readlines():
            if not line:
                continue
            md, kw = line.strip().split(",")
            try:
                output[md] = float(kw) * SOLAR_CAPACITY / BASE_SOLAR_CAPACITY
            except ValueError:
                continue
    return output


def chart_hourly_output_month(df: pd.DataFrame, max_y: float):
    """Group by day. Plot one line per day with hour on x-axis and electricity on y-axis."""
    plt.figure(figsize=(10, 6))  # create figure once
    # get min, max, and median
    per_day = df.groupby("day").agg({"electricity": "sum"})
    min_val = round(per_day["electricity"].min())
    max_val = round(per_day["electricity"].max())
    median_val = round(per_day["electricity"].median())
    days_over_use = per_day[per_day["electricity"] >= DAILY_USE].shape[0]
    over_pct = round(days_over_use / len(per_day) * 100)
    stats_text = (
        f"{min_val} - {max_val} kWh\n"
        f"median  {median_val} kWh\n"
        f"{days_over_use} days ({over_pct})% ≥ {DAILY_USE} kWh"
    )
    plt.text(
        0.99,
        0.95,
        stats_text,
        ha="right",
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
    row = df.iloc[0]
    month_name = row["month_label"]
    month_num = row["month"]
    # add month name as title
    plt.title(f"Hourly solar output {month_name}:  {SOLAR_CAPACITY} kW")
    filename = f"{get_output_dir()}/solar_hourly_{month_num:02d}_{month_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {filename}")


def chart_hourly_output():
    """Load hourly output estimate from output.csv

    time,local_time,electricity,day,month
    2019-01-01 0:00,2018-12-31 16:00,1.34,01-01,01 January,,,,,,,,,,,
    from https://www.renewables.ninja
    """
    df = pd.read_csv(
        "output_hourly.csv", parse_dates=["local_time"], dtype={"electricity": float}
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
    # color palette for up to 12 months
    colors = [
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
    by_month = pd.DataFrame()
    label_colors = []
    # chart: average by hour, one line per month
    for i, (month, group) in enumerate(sorted(df.groupby("month"))):
        chart_hourly_output_month(group, max_y)
        days = len(group["day"].unique())
        daily = round(group["electricity"].sum() / days)
        label_colors.append("black" if daily >= DAILY_USE else "red")
        group = group.groupby("hour").agg({"electricity": "mean"}).reset_index()
        by_month = pd.concat([by_month, group], ignore_index=True)
        group = group.sort_values(by="hour")
        month_name = datetime(2025, month, 1).strftime("%B")
        color = colors[i % len(colors)]
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
    legend = plt.legend(labelcolor=label_colors)
    # Set legend text color to red for under_months
    filename = f"{get_output_dir()}/solar_all_months.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {filename}")
    # write data for use by other charts
    by_month.to_csv(f"{get_output_dir()}/solar_hourly_by_month.csv", index=False)
    # montage *.png -tile 3x4 -geometry +2+2 grid.png

    # one chart per month: average by hour, online line per day
    for i, (month, group) in enumerate(sorted(df.groupby("month"))):
        plt.figure(figsize=(10, 6))


def load_hourly_output() -> dict[str, float]:
    """Load hourly output estimate from output_hourly.csv

    time,local_time,electricity,day,month
    2019-01-01 0:00,2018-12-31 16:00,1.34,01-01,01 January,,,,,,,,,,,
    from https://www.renewables.ninja
    """
    output: dict[str, float] = {}
    reader = csv.DictReader(open("output_hourly.csv"))
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


def calculate_v2(df: pd.DataFrame):
    hourly_output = load_hourly_output()
    rows = []
    battery_level = BATTERY_CAPACITY / 2
    # for each row in df
    for _, row in df.iterrows():
        flow = Flow(
            timestamp=row["Timestamp"].to_pydatetime(),
            # need this much
            demand=row["kW ev"],
            # available solar generation
            solar_generation=hourly_output.get(
                row["Timestamp"].strftime("%m-%d %H:%M"), 0.0
            ),
            start_battery_level=battery_level,
        )
        period_type = row["period_type"]
        demand = flow.demand
        if period_type in ["peak", "part peak"]:
            # from solar first
            flow.from_solar = min(demand, flow.solar_generation)
            demand -= flow.from_solar
            # then from battery
            flow.from_battery = min(demand, flow.start_battery_level)
            demand -= flow.from_battery
            # then from grid
            flow.from_grid = demand
            # to grid
            excess = flow.solar_generation - flow.from_solar
            flow.to_battery = min(BATTERY_CAPACITY - flow.start_battery_level, excess)
            flow.to_grid = excess - flow.to_battery
        else:
            # solar to battery first
            flow.to_battery = min(
                flow.solar_generation, BATTERY_CAPACITY - flow.start_battery_level
            )
            battery_level = flow.start_battery_level + flow.to_battery
            # solar to demand next
            flow.from_solar = min(demand, flow.solar_generation - flow.to_battery)
            demand -= flow.from_solar
            # then from battery if battery is high enough
            if battery_level > OFF_PEAK_BATTERY_LEVEL:
                flow.from_battery = min(demand, battery_level)
                demand -= flow.from_battery
            # then from grid
            flow.from_grid = demand
            # to grid
            flow.to_grid = flow.solar_generation - flow.from_solar - flow.to_battery
        # update battery level
        flow.end_battery_level = (
            flow.start_battery_level - flow.from_battery + flow.to_battery
        )
        battery_level = flow.end_battery_level
        print(flow)
        rows.append(flow.model_dump(mode="json"))
    df_flows = pd.DataFrame(rows)
    # add pd.Timestamp column to df_flows from timestamp
    df_flows["Timestamp"] = pd.to_datetime(df_flows["timestamp"])
    df = df.merge(df_flows, on="Timestamp", how="left")
    df.to_csv(f"{get_output_dir()}/flows.csv", index=False)
    return df


def calculate_v3(df: pd.DataFrame) -> pd.DataFrame:
    hourly_output = load_hourly_output()
    rows = []
    battery_level = BATTERY_CAPACITY / 2
    for _, row in df.iterrows():
        flow = Flow(
            timestamp=row["Timestamp"].to_pydatetime(),
            # need this much
            demand=row["kW ev"],
            # available solar generation
            solar_generation=hourly_output.get(
                row["Timestamp"].strftime("%m-%d %H:%M"), 0.0
            ),
            start_battery_level=battery_level,
        )
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
        else:
            # from battery if battery is high enough
            if demand and battery_level > OFF_PEAK_BATTERY_LEVEL:
                flow.from_battery = min(demand, battery_level)
                demand -= flow.from_battery
            if demand:
                flow.from_grid = demand
            else:
                # to battery
                excess = flow.solar_generation - flow.from_solar
                flow.to_battery = min(
                    excess, BATTERY_CAPACITY - flow.start_battery_level
                )
        # to grid
        flow.to_grid = flow.solar_generation - flow.from_solar - flow.to_battery
        # update battery level
        flow.end_battery_level = (
            flow.start_battery_level - flow.from_battery + flow.to_battery
        )
        battery_level = flow.end_battery_level
        print(flow)
        rows.append(flow.model_dump(mode="json"))
    df_flows = pd.DataFrame(rows)
    # add pd.Timestamp column to df_flows from timestamp
    df_flows["Timestamp"] = pd.to_datetime(df_flows["timestamp"])
    df = df.merge(df_flows, on="Timestamp", how="left")
    df.to_csv(f"{get_output_dir()}/flows_v3.csv", index=False)
    return df


def chart_flows(df: pd.DataFrame, month: str):
    # positive: from_solar + from_battery + from_grid = demand
    # negative: to_battery + to_grid = excess
    # solar generation, demand
    # battery level
    # get rows where yyyymm = 202507
    df_month = df[df["yyyymm"] == month].sort_values(by="Timestamp")

    cols = [
        "from_solar",
        "from_battery",
        "from_grid",
        "demand",
        "solar_generation",
        "to_battery",
        "to_grid",
    ]
    # average columns by hour_label
    df_month = df_month.groupby("hour").agg({col: "mean" for col in cols})
    title = f"{month[:4]}-{month[4:6]} flows"
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    # chart 1
    # draw full height bar  with light yellow color for hour=15, hour=21-23
    # draw full height bar with light red color for hour=16-20
    # stack bars for from_solar, from_battery, from_grid
    max_y = df_month[cols].max().max() * 1.1
    axs[0].set_ylim(0, max_y)
    df_month["hour_label"] = df_month.index.astype(str)
    """
    for hour in [15, 21, 22, 23]:
        axs[0].bar(
            [hour],
            [max_y],
            color="lightyellow",
        )
    for hour in [16, 17, 18, 19, 20]:
        axs[0].bar(
            [hour],
            [max_y],
            color="lightcoral",
        )
    """
    for ax in axs:
        ax.axvspan(15, 24, color="lightyellow", alpha=1.0)
        ax.axvspan(16, 20, color="lightcoral", alpha=0.5)
    axs[0].bar(
        df_month["hour_label"],
        df_month["from_solar"],
        label="from_solar",
        color="yellow",
    )
    axs[0].bar(
        df_month["hour_label"],
        df_month["from_battery"],
        label="from_battery",
        bottom=df_month["from_solar"],
        color="green",
    )
    axs[0].bar(
        df_month["hour_label"],
        df_month["from_grid"],
        label="from_grid",
        bottom=df_month["from_solar"] + df_month["from_battery"],
        color="blue",
    )
    axs[0].plot(df_month["hour_label"], df_month["demand"], label="demand")
    axs[0].plot(
        df_month["hour_label"], df_month["solar_generation"], label="solar_generation"
    )
    # set colors: solar = yellow, battery = green, grid = blue, demand = red, solar_generation = orange
    axs[0].legend()
    # chart 2
    # chart 2: to_battery bar, to_grid bar, battery level line
    axs[1].bar(
        df_month["hour_label"],
        df_month["to_battery"],
        label="to_battery",
        color="lightgreen",
    )
    axs[1].bar(
        df_month["hour_label"],
        df_month["to_grid"],
        label="to_grid",
        bottom=df_month["to_battery"],
        color="lightblue",
    )
    # axs[1].plot(df_month["hour_label"], df_month["battery_level"], label="battery_level")
    axs[1].legend()
    plt.tight_layout()
    plt.title(title)
    # plt.show()
    # save to month.png
    fig.savefig(f"{get_output_dir()}/{month}.png", dpi=300, bbox_inches="tight")


def with_solar(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate solar output and battery use.

    Return dataframe with daily usage and battery level.
    """
    # load solar output model
    output = load_output()
    # start with half-full batter
    battery_level = BATTERY_CAPACITY / 2
    df[
        [
            "kW grid use",
            "kW battery use peak",
            "kW battery use part peak",
            "kW battery use off peak",
            "battery percent",
        ]
    ] = 0.0
    rows = []
    print("ymd\tuse\toutput\tbattery percent\texcess")
    # for each day
    for ymd, group in df.groupby("ymd"):
        day_use = group["kW ev"].sum()
        # apply battery to use for peak usage first
        battery_peak = set_use(group, "peak", battery_level)
        battery_level -= battery_peak
        # then part peak (one hour of part peak before peak but treat it as all after peak)
        battery_part_peak = set_use(group, "part peak", battery_level)
        battery_level -= battery_part_peak
        # then off peak
        battery_off_peak = set_use(group, "off peak", battery_level)
        battery_level -= battery_off_peak

        # copy values from group to df
        for col in [
            "kW battery use peak",
            "kW battery use part peak",
            "kW battery use off peak",
            "kW grid use",
            "battery percent",
        ]:
            df.loc[group.index, col] = group[col].values

        day_grid_use = group["kW grid use"].sum()

        # ymd = 2025-01-01, output = 01-01
        output_key = ymd[5:]  # get mm-dd from ymd
        output_kwh = output.get(output_key, 0.0)
        excess = max(0, battery_level + output_kwh - BATTERY_CAPACITY)
        # excess is at the daily level; assign to noon for the day
        df.loc[group.index, "excess"] = 0
        ts = pd.Timestamp(ymd) + pd.Timedelta(hours=12)
        df.loc[df["Timestamp"] == ts, "excess"] = excess
        battery_level = min(battery_level + output_kwh, BATTERY_CAPACITY)
        rows.append(
            {
                "ymd": ymd,
                "use": day_use,
                "battery peak": battery_peak,
                "battery part peak": battery_part_peak,
                "battery off peak": battery_off_peak,
                "grid use": day_grid_use,
                "battery percent": battery_level,
                "excess": excess,
            }
        )
        print(
            f"{ymd}\t{round(day_use)}\t{round(output_kwh)}\t {round(battery_level)}\t{round(excess)}"
        )
    days = pd.DataFrame(rows)
    days.to_csv(f"{get_output_dir()}/days.csv", index=False)
    return df


def create_cost_chart(df: pd.DataFrame):
    """
    Timestamp,kW,kW ev,month,hour,yyyymm,ymd,season,period,period_type,
    kW grid use,kW battery use peak,kW battery use part peak,kW battery use off peak,battery level,excess,
    ELEC,credit per kWh,service charge,pge cost ELEC,net cost ELEC,savings,EV2-A,pge cost EV2-A,net cost EV2-A
    """
    df["kW total battery use"] = (
        df["kW battery use peak"]
        + df["kW battery use part peak"]
        + df["kW battery use off peak"]
    )
    # Average monthly utility bill
    # group by yyyymm
    rows = []
    for yyyymm, group in df.groupby("yyyymm"):
        # total use = sum of kW ev
        total_use = group["kW ev"].sum()
        pge_cost = group["pge cost EV2-A"].sum()
        net_cost = group["net cost EV2-A"].sum()
        month = int(yyyymm[5:7])
        dt = date(int(yyyymm[:4]), month, 1)
        rows.append(
            {
                "month_name": dt.strftime("%b"),
                "month": month,
                # "total use": total_use,
                "PG&E cost": pge_cost,
                "net cost": net_cost,
            }
        )
    df_cost = pd.DataFrame(rows)
    # sort by month
    df_cost = df_cost.sort_values(by="month")
    # create a side-by-side bar chart: months -> PG&E cost, net cost
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the width of each bar and positions for side-by-side bars
    bar_width = 0.35
    x = range(len(df_cost))

    # Create side-by-side bars
    ax.bar(
        [i - bar_width / 2 for i in x],
        df_cost["PG&E cost"],
        bar_width,
        label="PG&E Cost",
        alpha=0.8,
    )
    ax.bar(
        [i + bar_width / 2 for i in x],
        df_cost["net cost"],
        bar_width,
        label="Net Cost",
        alpha=0.8,
    )

    # Set x-axis labels and formatting
    ax.set_xlabel("Month")
    ax.set_ylabel("Cost ($)")
    ax.set_title("Monthly Utility Bills")
    ax.set_xticks(x)
    ax.set_xticklabels(df_cost["month_name"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
    # write chart to cost.png
    fig.savefig(f"{get_output_dir()}/cost.png", dpi=300, bbox_inches="tight")
    df_cost.to_csv(f"{get_output_dir()}/monthly_cost.csv", index=False)
    print("wrote cost.png")


def open_spreadsheet() -> Spreadsheet:
    gc = gspread.auth.service_account(filename="google.json")
    return gc.open_by_key(SHEET_ID)


def write_to_sheet(df: pd.DataFrame):
    # use gspread to write to Google Sheet
    print("writing to Google Sheet")
    print("all columns=", df.columns.values.tolist())
    """
    ['Timestamp', 'kW', 'kW ev', 'month', 'hour', 'yyyymm', 'ymd', 'season', 'period', 'period_type', 'timestamp', 
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
    ss = open_spreadsheet()
    worksheet = ss.worksheet("model")
    worksheet.update(
        [df.columns.values.tolist()] + df.values.tolist(),
        value_input_option=ValueInputOption.user_entered,
    )


def add_costs(df: pd.DataFrame) -> pd.DataFrame:
    tariffs = list(RATE_VALUES.keys())
    for tariff in tariffs:
        df = set_rates(df, tariff)
        # cost without solar: demand * rate
        col = f"pge cost {tariff}"
        df[col] = df["demand"] * df[tariff] + df["service charge"]
        # cost with solar: from_grid * rate - to_grid * credit per kWh
        credit_col = "pge credit"
        df[credit_col] = -(df["to_grid"] * df["credit per kWh"])
        grid_col = f"grid cost {tariff}"
        df[grid_col] = df["from_grid"] * df[tariff]
        if tariff == "ELEC":
            df[grid_col] += df["service charge"]
        net_col = f"net cost {tariff}"
        # credit is negative
        df[net_col] = df[credit_col] + df[grid_col]
        df[f"savings {tariff}"] = df[col] - df[net_col]
    df.to_csv(f"{get_output_dir()}/costs.csv", index=False)
    return df


def label_periods(df: pd.DataFrame) -> pd.DataFrame:
    # date helpers
    df["month"] = pd.to_datetime(df["Timestamp"]).dt.month
    df["hour"] = pd.to_datetime(df["Timestamp"]).dt.hour
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


def _initial_setup():
    """
    Timestamp,kW
    8/13/2025 11:00 PM,0.92
    """
    df = pd.read_csv("2024-2025.csv", parse_dates=["Timestamp"], dtype={"kW": float})
    df.to_csv("actual.csv", index=False)
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

    df.to_csv("charging.csv", index=False)


def initial_setup():
    return pd.read_csv("charging.csv", parse_dates=["Timestamp"], dtype={"kW": float})


def get_output_dir():
    return f"solar-{round(SOLAR_CAPACITY)}-battery-{round(BATTERY_CAPACITY)}"


def main():
    df = initial_setup()
    df = label_periods(df)

    # average daily use by period type
    print("average daily use by period")
    daily = df.groupby(["ymd", "period_type"]).agg({"kW ev": "sum"}).reset_index()
    average = daily.groupby("period_type")["kW ev"].mean()
    for period_type, avg_kw in average.items():
        print(f"{period_type}\t{round(avg_kw)} kWh")
    print("\n")

    # add battery and grid use
    # df = with_solar(df)
    # df = calculate_v2(df)
    df = calculate_v3(df)
    df = add_costs(df)

    """
    Timestamp,kW,kW ev,month,hour,yyyymm,ymd,season,period,period_type,
    kW grid use,kW battery use peak,kW battery use part peak,kW battery use off peak,excess,
    ELEC,credit per kWh,service charge,pge cost ELEC,net cost ELEC,savings,EV2-A,pge cost EV2-A,net cost EV2-A
    """
    """
    df.to_csv("cost.csv", index=False)
    df_daily = df.groupby("ymd").agg({col: "sum" for col in cost_cols}).reset_index()
    df_daily.to_csv("cost_daily.csv", index=False)


    # group by monthly
    daily_sums = [
        "kW ev",
        "kW grid use",
        "kW battery use peak",
        "kW battery use part peak",
        "kW battery use off peak",
        "excess",
        "credit per kWh",
        "service charge",
    ]
    df_monthly = (
        df.groupby("yyyymm").agg({col: "sum" for col in cost_cols}).reset_index()
    )

    df_monthly = df_monthly.sort_values(by="yyyymm", ascending=False)
    print(df_monthly, "\n")
    for key in cost_cols:
        total = df_monthly[key].sum()
        print(f"{key}\t${total:,.0f}")x
    """
    # create_cost_chart(df)
    write_to_sheet(df)
    for month in df["yyyymm"].unique():
        chart_flows(df, month)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solar", type=int, default=SOLAR_CAPACITY)
    parser.add_argument("--battery", type=int, default=BATTERY_CAPACITY)
    args = parser.parse_args()
    set_capacity(args.solar, args.battery)
    output_dir = get_output_dir()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # main()
    chart_hourly_output()
