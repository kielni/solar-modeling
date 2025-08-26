import glob
import os
import yaml
import pandas as pd
import numpy_financial
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates


def chart_cumulative(paths: dict[str, str]):
    fig, ax = plt.subplots(figsize=(12, 6))
    chart_pge = True
    colors = [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
    ]
    df = pd.read_csv("summary.csv")
    labels = {}
    for index, row in df.iterrows():
        labels[row["scenario"]] = (
            f"{row['scenario']} {row['payback period']:.1f} years {row['IRR']:.2%}"
        )

    for i, path in enumerate(paths):
        filename = f"{path}/roi.csv"
        if not os.path.exists(filename):
            print(f"skipping {filename}: not found")
            continue
        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["yyyymm"])
        scenario = paths[path]
        label = labels[scenario]
        ax.plot(
            df["date"],
            df["cumulative savings"],
            color=colors[i],
            label=label,
            linewidth=2,
        )
        ax.set_xlim(df["date"].min(), df["date"].max())

    ax.set_title(f"Cumulative savings")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${int(x):,}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    df = pd.read_csv("summary.csv")
    df = df.sort_values(by="IRR", ascending=False)
    plt.tight_layout()
    filename = "roi.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_periods(path: str):
    """

    Timestamp,kW,hour,yyyymm,kW ev,month,day,ymd,season,period,period_type,
    solar_efficiency,timestamp,demand,start_battery_level,end_battery_level,
    solar_generation,from_solar,from_battery,from_grid,to_battery,to_grid,ELEC,
    credit per kWh,service charge,pge cost ELEC,pge credit,grid cost ELEC,
    net cost ELEC,savings ELEC,EV2-A,pge cost EV2-A,grid cost EV2-A,net cost EV2-A,
    savings EV2-A"
    """
    df = pd.read_csv(f"{path}/costs.csv")
    cols = ["period", "demand", "pge cost EV2-A"]
    df = df[cols]
    # group by period and sum cost and demand
    df = df.groupby("period").sum()
    # percent of demand by period, percent of cost by period
    df["demand percent"] = df["demand"] / df["demand"].sum()
    df["cost percent"] = df["pge cost EV2-A"] / df["pge cost EV2-A"].sum()
    # sort by period
    df = df.sort_values(by="period")
    print("\nUse and cost by period")
    print("period\tuse %\tcost %\tcost $")
    for period, row in df.iterrows():
        print(
            f"{period}\t{row['demand percent']:.0%}\t{row['cost percent']:.0%}\t"
            f"{row['pge cost EV2-A']:,.0f}"
        )


def chart_daily_actual():
    df = pd.read_csv("actual.csv", parse_dates=["Timestamp"])
    df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
    df = df.groupby("date")["kW"].sum().reset_index().sort_values(by="date")
    # chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["date"], df["kW"], color="#FE6033")
    ax.set_title("Daily demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.grid(axis="y", color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    filename = "output/daily_demand.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {filename}")


def main():
    # read all .yml files in the current directory
    df = pd.DataFrame()
    paths = {}
    for file in glob.glob("solar*.yml"):
        # parse the yml file
        print(f"reading {file}")
        with open(file, "r") as f:
            config = yaml.safe_load(f)
        filename = f"{config['output']}/summary.csv"
        if not os.path.exists(filename):
            print(f"skipping {filename}: not found")
            continue
        summary = pd.read_csv(filename)
        # get scenario from first line of summary.csv
        scenario = summary.iloc[0]["scenario"]
        paths[config["output"]] = scenario
        df = pd.concat([df, summary])
    # sort by savings EV2-A
    df = df.sort_values(by="savings EV2-A")
    df.to_csv("summary.csv", index=False)

    for index, row in df.iterrows():
        print(
            f"{row['scenario']}\t\t{row['solar']:,.0f}\t{row['to grid']:,.0f}\t{row['savings EV2-A']:,.0f}"
        )
    chart_cumulative(paths)
    summarize_periods(list(paths.keys())[0])
    chart_daily_actual()


if __name__ == "__main__":
    main()
