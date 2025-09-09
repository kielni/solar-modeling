# Input data files

## Usage

### usage.csv

Hourly electricity usage, in kW.

```
Timestamp,kW
2024-08-01 00:00:00,1.1211
```

## Solar output

### output_hourly.csv

Hourly solar output with local time, from https://www.renewables.ninja.

```
time,local_time,electricity
2019-01-01 0:00,2018-12-31 16:00,1.34
```

## Export credits

### pge-export.csv

PG&E export credits, downloaded from !!! TODO: 
Renamed from `PG&E NBT EEC Values 2025 Vintage.csv` (& in filenames is a nuisance).
DateTime converted to PST/PDT.
The PG&E file contains one row per credit type (delivery or generation);
write one row per hour with generation credit and delivery credit fields
to `pge-export-pt.csv`.

```
RIN,RateName,DateStart,TimeStart,DateEnd,TimeEnd,DayStart,DayEnd,ValueName,Value,Unit,RateType,Sector
USCA-XXPG-NB24-0000,NBT24,1/1/2024,8:00:00,1/1/2024,8:59:59,8,8,Jan Weekend HS0,0.05372,Export $/kWh,TOU,All
```

### pge-export-pt.csv

PG&E export credits for 2023-2043. One row per Pacific time hour,
with generation credit and delivery credit values.
```
DateTime,generation credit,delivery credit
2024-01-01 00:00:00,0.05372,0.00313
```

### arbitrage_targets.csv

Filtered and re-formatted `pge-export-pt.csv`: 2 highest value credits per day
where total credit is above off peak rate for the season.

```
DateTime,generation credit,delivery credit,date,total credit,month,hour,season,target
2024-08-01 19:00:00,0.97728,0.22477,2024-08-01,1.20205,8,19,summer,0.6128600000000001
```
