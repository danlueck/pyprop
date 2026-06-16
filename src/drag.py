import math
import numpy as np
from numpy.typing import ArrayLike
from src.constants import R_EARTH
from src.constants import SEC_SID_DAY
from valladopy.astro.time.frame_conversions import ecef2teme, teme2ecef


def get_acc_drag(state: np.ndarray, sat: dict) -> np.ndarray:
    """Estimates the Accelaration due to Gravity at a given moment

    Args:
        state (np.ndarray): State Vector at given time
        sat (dict): Dictionary with physical properties of satellite
    Returns:
        np.ndarray: Accelaration vector
    """
    altitude = np.linalg.norm(state[0:3]) - R_EARTH
    density = get_density(altitude=altitude)

    B_inv = (sat["area"]*sat["c_d"]*1e-6)/sat["mass"]
    vel_atm = get_vel_atm(state=state[0:3])
    rel_vel = state[3:6] - vel_atm

    acc_drag = -0.5 * B_inv * density * np.linalg.norm(rel_vel)**2*(rel_vel/(np.linalg.norm(rel_vel)))
    return acc_drag


def get_density(altitude: float) -> float:
    """Retrieve the density of the atmosphere at a given altitude using an exponential model

    Args:
        altitude (float): Altitude for which to evaluate density

    Raises:
        Exception: Throws exception if objects altitude is negative

    Returns:
        float: Density of the atmosphere at the given altitude
    """
    expAltitudes = np.array([25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,100.0, 
                             110.0,120.0,130.0,140.0,150.0,180.0,200.0,250.0,300.0, 
                             350.0,400.0,450.0,500.0,600.0,700.0,800.0,900.0,1000.0])
    exph0 = np.array([0.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,100.0, 
                     110.0,120.0,130.0,140.0,150.0,180.0,200.0,250.0,300.0, 
                     350.0,400.0,450.0,500.0,600.0,700.0,800.0,900.0])
    expDensity = np.array([1.225,3.899e-2,1.774e-2,3.972e-3,1.057e-3,3.206e-4,8.770e-5, 
                     1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 
                     2.070e-9, 5.464e-10,2.789e-10,7.248e-11,2.418e-11,9.518e-12,3.725e-12,
                     1.585e-12,6.967e-13,1.454e-13,3.614e-14,1.170e-14,5.245e-15])
    expHeight = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 
                        5.877, 7.263, 9.473,12.636,16.149,22.523,29.740,37.105,45.546,
                        53.628,53.298,58.515,60.828,63.822,71.835,88.667,124.64,181.05])
    if(altitude<0.0):
        raise Exception("Object Impacted EARTH")
    if(altitude>1000.0): return 0.0

    for i in range(0, len(expAltitudes)):
        if(altitude<expAltitudes[i]):
            idx = i
            return 1e9* expDensity[idx]*math.exp(-(altitude - exph0[idx])/expHeight[idx])
        
    return 0.0
   
def get_vel_atm(state: np.ndarray) -> np.ndarray:
    """Estimate the velocity of the rotating atmosphere in ECI frame

    Args:
        state (np.ndarray): State Vector at given time
    Returns:
        np.ndarray: Velocity of Atmosphere
    """
    r_loc = math.sqrt(state[0]**2 + state[1]**2)
    c_loc = 2*math.pi*r_loc
    vel_atm_loc = c_loc/SEC_SID_DAY
    vel_vec_atm = state[0:3]
    rot_mat = np.array([[0.0, -1.0, 0.0],
                         [1.0,  0.0, 0.0],
                         [0.0,  0.0, 0.0]])
    vel_vec_atm = rot_mat.dot(vel_vec_atm)
    vel_vec_atm = vel_vec_atm * (vel_atm_loc/np.linalg.norm(vel_vec_atm))    
    return vel_vec_atm
   
def get_vel_atm_slow(state, date, frame):
    r_ecef, _, _ = teme2ecef(state[0:3], np.zeros(3), np.zeros(3), 0.0, date, 0.0, 0.0, 0.0, frame)
    _, vel_atm, _ = ecef2teme(r_ecef, np.zeros(3), np.zeros(3), 0.0, date, 0.0, 0.0, 0.0, frame)    
    return vel_atm
