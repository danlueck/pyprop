import numpy as np
from numpy.typing import ArrayLike
from valladopy.astro.celestial.moon import position as moon_pos
from valladopy.astro.celestial.sun import position as sun_pos
from src.constants import MU_MOON, MU_SUN
from datetime import datetime
from valladopy.mathtime.julian_date import invjday, jday


def get_acc_3rd_body(state: ArrayLike, date: datetime, bodies: dict):
    acc_3rd = np.zeros(3)
    if(bodies["Sun"]):
        acc_3rd += get_acc_n_body(state=state, date=date, pos_fun=sun_pos, mu_body=MU_SUN)
    if(bodies["Moon"]):
        acc_3rd += get_acc_n_body(state=state, date=date, pos_fun=moon_pos, mu_body=MU_MOON)
    return acc_3rd

def get_acc_n_body(state: ArrayLike, date: datetime, pos_fun, mu_body: float) -> np.ndarray:
    """Estimates the Accelaration due to Gravity at a given moment

    Args:
        state (ArrayLike): State Vector at given time

    Returns:
        np.ndarray: Accelaration vector
    """
    julian_day, julian_fraction = jday(date.year, date.month, date.day, date.hour, date.minute, date.second)
    julian_date = julian_day + julian_fraction 
    pos_earth_3rd, _, _ = pos_fun(julian_date)
    pos_sat_3rd = pos_earth_3rd - state[0:3]

    return mu_body*(pos_sat_3rd*(1.0/np.linalg.norm(pos_sat_3rd)**3)-pos_earth_3rd*(1.0/np.linalg.norm(pos_earth_3rd)**3))
