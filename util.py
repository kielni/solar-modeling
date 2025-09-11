from pydantic import BaseModel

COLORS = {
    "from_solar": "#FDC451",
    "from_battery": "#167590",
    "from_grid": "#FE6033",
    "to_grid": "#8E0B59",
    "to_battery": "#25A9D0",
    # test these
    "solar_generation": "yellow",
    "battery_level": "darkblue",
    "system": "#1a9641",
    "savings": "#018571",
    "part peak": "lightyellow",
    "peak": "lightcoral",
    "delivery": "#1f78b4",
    "delivery-light": "#a6cee3",
    "generation": "#33a02c",
    "generation-light": "#b2df8a",
    "bonus": "#6157d1",
    "other": "#fb9a99",
    "vs": "lightgray",
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
https://www.pge.com/tariffs/assets/pdf/tariffbook/ELEC_SCHEDS_EV2%20(Sch).pdf

Winter Season: October-May
0.31 Off-peak hours are 12 midnight to 3 p.m.
0.62 Peak hours (4-9 p.m.): electricity is more expensive
0.51 Partial-peak (3-4 p.m. and 9 p.m. - 12 midnight)

Summer Season: June-September
0.31 Off-peak hours are 12 midnight to 3 p.m.
0.50 Peak hours (4-9 p.m.): electricity is more expensive
0.48 Partial-peak (3-4 p.m. and 9 p.m. - 12 midnight)
"""


class Charges(BaseModel):
    generation: float
    delivery: float
    other: float

    def total(self) -> float:
        return self.generation + self.delivery + self.other


"""
https://www.pge.com/tariffs/assets/pdf/tariffbook/ELEC_SCHEDS_E-ELEC.pdf
Residential customers billed on the Net Billing Tariff must be served under this schedule and
are not required to have any of the eligible technologies listed above.

PEAK PART-PEAK OFF-PEAK
Generation:
Summer Usage $0.31659 $0.21748 $0.17238
Winter Usage $0.15446 $0.13449 $0.12114
Distribution**:
Summer Usage $0.23100 (R) $0.16823 (R) $0.15665 (R)
Winter Usage $0.16162 (R) $0.15950 (R) $0.15899 (R)

https://www.pge.com/tariffs/assets/pdf/tariffbook/ELEC_SCHEDS_EV2%20(Sch).pdf

Delivery Minimum Bill Amount ($ per meter per day) $0.40317

Generation:
Summer Usage $0.23985 $0.19514 $0.15400
Winter Usage $0.18298 $0.17049 $0.14701
Distribution**:
Summer Usage $0.31630 (R) $0.25052 (R) $0.08965 (R)
Winter Usage $0.24606 (R) $0.24185 (R) $0.09664 (R)

"""
OTHER = 0.05671
RATE_VALUES = {
    "ELEC": {
        # $0.60430
        "summer peak": Charges(
            # 52%
            generation=0.31659,
            # 38%
            delivery=0.23100,
            # 9%
            other=OTHER,
        ),
        # $0.44242
        "summer part peak": Charges(
            # 49%
            generation=0.21748,
            # 38%
            delivery=0.16823,
            # 13%
            other=OTHER,
        ),
        # $0.38574
        "summer off peak": Charges(
            # 45%
            generation=0.17238,
            # 41%
            delivery=0.15665,
            # 15%
            other=OTHER,
        ),
        # $0.37279
        "winter peak": Charges(
            # 41%
            generation=0.15446,
            # 43%
            delivery=0.16162,
            # 15%
            other=OTHER,
        ),
        # $0.35070
        "winter part peak": Charges(
            # 38%
            generation=0.13449,
            # 46%
            delivery=0.15950,
            # 16%
            other=OTHER,
        ),
        # $0.33684
        "winter off peak": Charges(
            # 36%
            generation=0.12114,
            # 47%
            delivery=0.15899,
            # 17%
            other=OTHER,
        ),
    },
}
"""
"EV2-A": {
        "summer peak": Charges(generation=0.23985, delivery=0.31630, other=OTHER),
        "summer part peak": Charges(generation=0.19514, delivery=0.25052, other=OTHER),
        "summer off peak": Charges(generation=0.15400, delivery=0.08965, other=OTHER),
        "winter peak": Charges(generation=0.18298, delivery=0.24606, other=OTHER),
        "winter part peak": Charges(generation=0.17049, delivery=0.24185, other=OTHER),
        "winter off peak": Charges(generation=0.14701, delivery=0.09664, other=OTHER),
    },
"""
RATE_VALUES_DAILY = {
    # Delivery Minimum Bill Amount
    "EV2-A": {"delivery minimum": 0.40317},
    """
    Customers on this
    schedule are not subject to the delivery minimum bill amount applied to the delivery portion of
    the bill (i.e. to all rate components other than the generation rate)
    """
    "ELEC": {"delivery minimum": 0.0},
}

# TODO: not in tariff document?
RATE_VALUES_MONTHLY = {"ELEC": {"service charge": 15.0}}

"""
https://www.pge.com/assets/pge/docs/save-energy-and-money/energy-savings-programs/solar-billing-plan-marketing-toolkit-en.pdf.coredownload.pdf

Customers who enroll before 2027 will get an additional
credit for the excess energy they send back to the grid.
The value of this bonus credit will decrease by 20% each
year and end (expire) in 2027.

The assigned additional credit value will be locked in for customers and continue
for nine years.
"""
BONUS_CREDIT = 0.013
BONUS_END_YEAR = 2028
