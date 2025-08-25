# Solar sizing

## setup

Get 1 year of hourly interval data from PG&E.

Added EV charging in March 2025.

<img src="presentation/load_curve.png">

Copy 12-3am charging data to

  - May -> Jan, Mar, Jun,Oct, Nov, Dec (winter)
  - Apr -> Feb, Nov (winter)
  - Jun -> Sep (summer)
  - Jul -> Aug (summer)

Then swap 12-3am / 12-3pm to do EV charging when solar is available.

Get solar outout estimates from https://www.renewables.ninja

  - 10 kW system size
  - San Jose, CA
  - 16,962 annual kWh
  - higher than solar company estimate: 16,680 kWh for 10.44 kW system

### daily usage pattern

- 33 kWh  off peak
- 4 kWh part peak
- 6 kWh peak

10 kWh part peak + peak

77% usage is off peak

TODO: EV charging estimate

### costs

| season | off peak | part peak | peak | to grid |
|--------|----------:|---:|---:|---:|
| winter | 0.31 | 0.51 | 0.62 | -0.03 |
| summer | 0.31 | 0.48 | 0.50 | -0.03 |

## methdology

Label TOU periods according to tariff rules:

  - 3-4pm 9pm-12am part peak
  - 4-9pm peak
  - 12am-3pm off peak

Load hourly solar output model.

Create flow model for each period

  - timestamp
  - demand
  - start_battery_level
  - end_battery_level
  - solar_generation
  - from_solar
  - from_battery
  - from_grid
  - to_battery
  - to_grid

In winter, use battery in off peak if it's at least this level;
might need all daily solar production to offset peak usage
`MIN_BATTERY_WINTER = EAK_USAGE`

In summer, let battery go lower since it will likely refill.
`MIN_BATTERY_SUMMER = PEAK_USAGE / 4`

### all periods

Use all available solar generation for demand (from_solar).

### peak, part peak

Then use all available battery (from_battery).

Then draw from grid (from_grid).

If excess (solar_generation > from_solar), charge battery to capacity (to_battery).

### off peak

Then draw from battery, maintaining minimum battery level (from_battery).

Then draw from grid (from_grid).

If fill off peak and last off peak hour, fill battery from grid (to_battery, from_grid).

### all periods

Export remainder (to_grid).

### calculate costs

rate = per-kWh cost from period and tariff

credit = -0.03

- pge cost = demand * rate
- pge credit = to_grid * credit
- grid cost = from_grid * rate
- net cost = grid cost + pge credit
- savings = pge cost - net cost

## observations

- chart_flows (1/mo)
- chart_daily (1/mo)

### hourly averages are misleading

Quote: "most typical day of the selected month"

<img src="presentation/quote_apr.png">

<img src="presentation/quote_jul.png">


### solar production is variable, except in summer

<img src="solar-10-battery-30/solar_monthly.png">

- Dec, Jan, Feb don't have enough generation available to cover demand: shorter day, more cloudy days
- Jul, Aug, Sep have ideal curves: no clouds
- average is not useful outside of summer

In January, only 7 days have enough solar to match demand
<img src="solar-10-battery-30/solar_hourly_01_January.png">

April usually has more solar than demand, and sometimes to 2/3 fill the battery
<img src="solar-10-battery-30/solar_hourly_04_April.png">

In July, every day can 2/3 fill the battery
<img src="solar-10-battery-30/solar_hourly_07_July.png">


Quote shows hourly
but there is 
  - EV charging is not every day; need total not average
  - solar is low for days at a time. This matters because the battery may drop to 0

Usually in winter (Dec-Feb), there's not enough solar available to cover demand.

But even in Jan, a 30 kWh battery sometimes fills.

## scenarios

### both from grid and export 

<img src="solar-10-battery-30/monthly_sources.png">

- sum from_solar, from_battery, from_grid, to_grid for all periods in a month
- Jul, Aug, Sep and almost Jun are solar+battery only
- some export in all but Dec. Runs of sun overfill the battery but runs of cloud require grid.

<img src="solar-10-battery-30/flow_01_January.png">
<img src="solar-10-battery-30/flow_04_April.png">
<img src="solar-10-battery-30/flow_07_January.png">

## more battery

Any solar generated when battery is full is a waste:
grid export credit is 5-10% of grid price. 
With more storage capacity, we could use battery instead of grid for 0.31-0.62. But batteries are expensive: $14k for 15 kWh capacity ($910/kWh).

<img src="solar-10-battery-45/monthly_sources.png">

Much reduced export in winter months, but still signifcant exports in summer.
Is this worth the battery cost?

## fewer panels

Reduce generation to reduce grid export waste.

<img src="solar-8-battery-30/monthly_sources.png">

This looks better: reduced exports compared to more battery. Fewer
panels cost less. Some grid use and also export in summer months.
How can these match better? 

## ROI

## questions

- what is the cost of the panels? $34,548.80 from total - storage - installation
- what is the 9 - 11.99 kW Group -$1,372.80 line item?
- what is the source for the Yr. 1 Production Estimate: 16,680 kWh with Solar System Size: 11.44
- what is Post-solar Year 1 estimate Fixed costs $13.55
- how is Post-solar Year 1 estimate Grid use $0.63 calculated
- explain methodology: show solar, battery charge, battery charge, grid export, demand for Jan, Jul, and May
- which PG&E tariff used for cost? E-TOU-C Residential - Time of Use Baseline Region X
  - 37-50c off peak, 40-63c peak
  - we are on EV2-A and that is still the best rate with solar
  - subtracting costs on higher E-TOU-C costs from actual EV2-A costs is not a valid methodology
- is Escalation 4.00% the expected change in YOY blended rates?

