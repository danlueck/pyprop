import numpy as np
from valladopy.astro.celestial.sun import position as sun_pos
from valladopy.astro.celestial.sun import in_light
from datetime import datetime
from valladopy.mathtime.julian_date import jday

def get_acc_srp(state: np.ndarray, date: datetime, satellite: dict) -> np.ndarray:
    """Get Acceleration due to solar radiation pressure

    Args:
        state (np.ndarray): Cartesian state vector in ECI frame
        date (datetime): Date to evaluate
        satellite (dict): Dictionary of physical satellite properties

    Returns:
        np.ndarray: Accelerations due to solar radiation pressure
    """
    julian_day, julian_fraction = jday(date.year, date.month, date.day, date.hour, date.minute, date.second)
    julian_date = julian_day + julian_fraction 
    if(not in_light(state[0:3], julian_date)):
        return np.zeros(3)
    # Get Sun position
    pos_earth_3rd, _, _ = sun_pos(julian_date)
    # Get vector from Sun to Satellite
    pos_sat_3rd = pos_earth_3rd - state[0:3]
    srp_effect = -4.57e-9*((satellite["c_r"]*satellite["area"])/satellite["mass"])
    return srp_effect*(pos_sat_3rd*(1/np.linalg.norm(pos_sat_3rd)))