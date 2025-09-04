import argparse
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

from util import COLORS


def chart_cumulative(scenarios: dict[str, dict], suffix: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
    ]
    for i, key in enumerate(scenarios):
        scenario = scenarios[key]
        filename = f"{scenario['output']}/roi.csv"
        if not os.path.exists(filename):
            print(f"skipping {filename}: not found")
            continue
        df = pd.read_csv(filename, parse_dates=["year_month"])
        """
        month,year_month,to_grid,
        generation net cost,generation credited,generation rollover credit,
        delivery net cost,delivery credited,delivery rollover credit,
        grid cost,net cost,savings,
        cumulative savings,cumulative pge,cumulative net
        """
        df["date"] = df["year_month"]
        label = scenario["description"]
        ax.plot(
            df["date"],
            df["cumulative savings"],
            color=colors[i],
            label=label,
            linewidth=2,
        )
        # plot generation rollover credit as a dashed line
        if df["generation rollover credit"].max() > 500:
            ax.plot(
                df["date"],
                df["generation rollover credit"],
                color=colors[i],
                label=f"{label} generation rollover credit",
                linestyle="--",
            )
        # plot delivery rollover credit as a dotted line
        if df["delivery rollover credit"].max() > 500:
            ax.plot(
                df["date"],
                df["delivery rollover credit"],
                color=colors[i],
                label=f"{label} delivery rollover credit",
                linestyle=":",
            )

        ax.set_xlim(df["date"].min(), df["date"].max())

    ax.set_title("Cumulative savings")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${int(x):,}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    plt.tight_layout()
    filename = f"output/roi{suffix}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_periods(path: str):
    """Summarize use and cost by period.

    Timestamp,season,period,period_type,solar_efficiency,demand,
    start_battery_level,end_battery_level,solar_generation,
    from_solar,from_battery,from_grid,to_battery,to_grid,
    ELEC,service charge,generation demand,generation credit,
    delivery demand,delivery credit,
    ELEC generation cost,ELEC generation credit,ELEC delivery cost,
    ELEC delivery credit,ELEC service charge,ELEC grid cost,
    EV2-A,EV2-A generation cost,EV2-A generation credit,EV2-A delivery cost,
    EV2-A delivery credit,EV2-A service charge,EV2-A grid cost,
    month,hour,day,yyyymm,ymd
    """
    df = pd.read_csv(f"{path}/costs.csv")
    cols = ["period", "demand", "EV2-A grid cost"]
    df_periods = df[cols]
    # group by period and sum cost and demand
    df_periods = df_periods.groupby("period").sum()
    # percent of demand by period, percent of cost by period
    # TODO: same keys?
    df_periods["demand percent"] = df_periods["demand"] / df_periods["demand"].sum()
    df_periods["cost percent"] = (
        df_periods["EV2-A grid cost"] / df_periods["EV2-A grid cost"].sum()
    )
    # sort by period
    df_periods = df_periods.sort_values(by="period")
    print("\nUse and cost by period")
    print("period\tuse %\tcost %\tcost $")
    for period, row in df_periods.iterrows():
        print(
            f"{period}\t{row['demand percent']:.0%}\t{row['cost percent']:.0%}\t"
            f"{row['EV2-A grid cost']:,.0f}"
        )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_periods.index, df_periods["cost percent"], color=COLORS["system"])
    ax.set_title("Cost by period")
    ax.set_xlabel("Period")
    ax.set_ylabel("Cost")
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    # format y axis as percentage
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"{x:.0%}"))
    plt.tight_layout()
    filename = "output/periods.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {filename}")


def chart_daily_actual():
    df = pd.read_csv("data/actual.csv", parse_dates=["Timestamp"])
    df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
    df = df.groupby("date")["kW"].sum().reset_index().sort_values(by="date")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["date"], df["kW"], color="#FE6033")
    ax.set_title("Daily demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(df["date"].min(), df["date"].max())
    plt.tight_layout()
    filename = "output/daily_demand.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {filename}")


def main(scenario_set: str):
    # read all .yml files in the current directory
    df = pd.DataFrame()
    sets = {
        "base": [
            "solar-8-battery-30.yml",
            "solar-10-battery-30.yml",
            "solar-10-battery-45.yml",
            "solar-8-battery-30-rate-8.yml",
            "arbitrage-limit-8.yml",
        ],
        "arbitrage": [
            "arbitrage-max.yml",
            "arbitrage-limit-10.yml",
            "arbitrage-limit-20.yml",
            "arbitrage-limit-8.yml",
        ],
    }
    suffix = "-arbitrage" if scenario_set == "arbitrage" else ""
    included = sets[scenario_set]
    scenarios = {}

    for file in included:
        # parse the yml file
        print(f"reading {file}")
        with open(file, "r") as f:
            config = yaml.safe_load(f)
            scenarios[config["output"]] = config
        filename = f"{config['output']}/summary.csv"
        if not os.path.exists(filename):
            print(f"skipping {filename}: not found")
            continue
        summary = pd.read_csv(filename)
        df = pd.concat([df, summary])
    df = df.sort_values(by="IRR")
    df.to_csv(f"output/summary{suffix}.csv", index=False)
    print("scenario\ttariff\tyear 1 savings\tpayback period\tIRR\tsolar %")
    for _, row in df.iterrows():
        solar_pct = 1 - (row["from_grid"] / row["demand"])
        print(
            f"{row['scenario']}\t{row['tariff']}\t{row['savings']:,.0f}\t"
            f"{row['payback period']:.1f}\t{row['IRR']:.2%}\t{solar_pct:.0%}"
        )
    chart_cumulative(scenarios, suffix)
    # summarize_periods(list(paths.keys())[0])
    # chart_daily_actual()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("set", type=str, default="base", nargs="?")
    args = parser.parse_args()
    main(args.set)
