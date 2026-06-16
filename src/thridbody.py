import numpy as np
from valladopy.astro.celestial.moon import position as moon_pos
from valladopy.astro.celestial.sun import position as sun_pos
from src.constants import MU_MOON, MU_SUN
from datetime import datetime
from valladopy.mathtime.julian_date import jday


def get_acc_3rd_body(state: np.ndarray, date: datetime, bodies: dict) -> np.ndarray:
    """_summary_

    Args:
        state (np.ndarray): Cartesian state vector in ECI frame
        date (datetime): Date to evaluate
        bodies (dict): Dictionary of 3rd bodies to consider

    Returns:
        np.ndarray: Accelaration vector due to 3rd bodies gravity
    """
    acc_3rd = np.zeros(3)
    if(bodies["Sun"]):
        acc_3rd += get_acc_n_body(state=state, date=date, pos_fun=sun_pos, mu_body=MU_SUN)
    if(bodies["Moon"]):
        acc_3rd += get_acc_n_body(state=state, date=date, pos_fun=moon_pos, mu_body=MU_MOON)
    return acc_3rd

def get_acc_n_body(state: np.ndarray, date: datetime, pos_fun, mu_body: float) -> np.ndarray:
    """Get the accelaration due to 3rd body gravities

    Args:
        state (np.ndarray): Cartesian state vector in ECI frame
        date (datetime): Date to evaluate
        pos_fun (_type_): Function to get postition of 3rd body relative to earth
        mu_body (float): Gravity constant of 3rd body

    Returns:
        np.ndarray: Accelaration vector due to 3rd body gravity
    """
    julian_day, julian_fraction = jday(date.year, date.month, date.day, date.hour, date.minute, date.second)
    julian_date = julian_day + julian_fraction 
    pos_earth_3rd, _, _ = pos_fun(julian_date)
    pos_sat_3rd = pos_earth_3rd - state[0:3]

    return mu_body*(pos_sat_3rd*(1.0/np.linalg.norm(pos_sat_3rd)**3)-pos_earth_3rd*(1.0/np.linalg.norm(pos_earth_3rd)**3))
