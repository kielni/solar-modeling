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

BATTERY_CAPACITY = 30
# use battery in off peak if it's at least this level
OFF_PEAK_BATTERY_LEVEL = 10
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
                output[md] = float(kw)
            except ValueError:
                continue
    return output


def load_hourly_output() -> dict[str, float]:
    """Load hourly output estimate from output.csv

    time,local_time,electricity,day,month
    2019-01-01 0:00,2018-12-31 16:00,1.34,01-01,01 January,,,,,,,,,,,
    from https://www.renewables.ninja
    """
    output: dict[str, float] = {}
    reader = csv.DictReader(open("output_hourly.csv"))
    for row in reader:
        # 2018-12-31 16:00
        hour = row["local_time"][5:]
        output[hour] = float(row["electricity"])
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


def calculate(df: pd.DataFrame) -> pd.DataFrame:
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
    df.to_csv("flows.csv", index=False)
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
    # stack bars for from_solar, from_battery, from_grid
    df_month["hour_label"] = df_month.index.astype(str)
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
    fig.savefig(f"{month}.png", dpi=300, bbox_inches="tight")


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
    days.to_csv("days.csv", index=False)
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
    fig.savefig("cost.png", dpi=300, bbox_inches="tight")
    df_cost.to_csv("monthly_cost.csv", index=False)
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
    df.to_csv("costs.csv", index=False)
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


def main():
    """
    Timestamp,kW
    8/13/2025 11:00 PM,0.92
    """
    # df = pd.read_csv("2024-2025.csv", parse_dates=["Timestamp"], dtype={"kW": float})
    # df.to_csv("actual.csv", index=False)

    # copy ev charging from 2025-04+ backwards
    # df = add_ev(df)
    # df.to_csv("charging.csv", index=False)
    # load use with charging already applied
    df = pd.read_csv("charging.csv", parse_dates=["Timestamp"], dtype={"kW": float})
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
    df = calculate(df)
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


    # group by monthy
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
        print(f"{key}\t${total:,.0f}")
    """
    # create_cost_chart(df)
    write_to_sheet(df)
    for month in df["yyyymm"].unique():
        chart_flows(df, month)


if __name__ == "__main__":
    main()
