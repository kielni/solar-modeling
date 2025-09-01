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


OTHER = 0.05671

"""
Generation:
Summer Usage $0.23985 $0.19514 $0.15400
Winter Usage $0.18298 $0.17049 $0.14701
Distribution**:
Summer Usage $0.31630 (R) $0.25052 (R) $0.08965 (R)
Winter Usage $0.24606 (R) $0.24185 (R) $0.09664 (R)
"""
RATE_VALUES = {
    "ELEC": {
        "summer peak": Charges(generation=0.23985, delivery=0.31630, other=OTHER),
        "summer part peak": Charges(generation=0.19514, delivery=0.25052, other=OTHER),
        "summer off peak": Charges(generation=0.15400, delivery=0.08965, other=OTHER),
        "winter peak": Charges(generation=0.18298, delivery=0.24606, other=OTHER),
        "winter part peak": Charges(generation=0.17049, delivery=0.24185, other=OTHER),
        "winter off peak": Charges(generation=0.14701, delivery=0.09664, other=OTHER),
    },
}
"""
    "EV2-A": {
        "winter off peak": 0.31,
        "winter peak": 0.48,
        "winter part peak": 0.50,
        "summer off peak": 0.31,
        "summer peak": 0.62,
        "summer part peak": 0.51,
    },
"""
