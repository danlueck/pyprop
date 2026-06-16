import numpy as np
from numpy.typing import ArrayLike
from src.constants import MU, J2, J3, J4, R_EARTH



def get_acc_grav(state: ArrayLike, geopotenital_order: int=0) -> np.ndarray:
    """Estimates the Accelaration due to Gravity at a given moment

    Args:
        state (ArrayLike): State Vector at given time
        geopotenital_order (int): Geopotential degree order

    Returns:
        np.ndarray: Accelaration vector
    """
    acc_grav = -(MU/(np.linalg.norm(state[0:3]))**3)*state[0:3]
    if (geopotenital_order >= 2):
        acc_grav += get_acc_J2(state=state)
    if (geopotenital_order >= 3):
        acc_grav += get_acc_J3(state=state)
    if (geopotenital_order >= 4):
        acc_grav += get_acc_J4(state=state)
        
    return acc_grav


def get_acc_J2(state: ArrayLike) -> np.ndarray:
    """Estimates the Accelaration due to J2 Term

    Args:
        state (ArrayLike): State Vector at given time

    Returns:
        np.ndarray: Accelaration vector
    """
    acc_J2 = np.zeros(3)
    pre_term = -(3.0/2.0) * MU*J2 * ((R_EARTH**2)/(np.linalg.norm(state[0:3])**5))
    acc_J2[0] = pre_term * state[0] * (1.0 - 5.0*((state[2]**2)/(np.linalg.norm(state[0:3])**2)))
    acc_J2[1] = pre_term * state[1] * (1.0 - 5.0*((state[2]**2)/(np.linalg.norm(state[0:3])**2)))
    acc_J2[2] = pre_term * state[2] * (3.0 - 5.0*((state[2]**2)/(np.linalg.norm(state[0:3])**2)))
    return acc_J2


def get_acc_J3(state: ArrayLike) -> np.ndarray:
    """Estimates the Accelaration due to J3 term

    Args:
        state (ArrayLike): State Vector at given time

    Returns:
        np.ndarray: Accelaration vector
    """
    acc_J3 = np.zeros(3)
    pre_term = -(5.0/2.0) * MU*J3 * ((R_EARTH**3)/(np.linalg.norm(state[0:3])**7))
    acc_J3[0] = pre_term * state[0] * (3.0 * state[2] - 7.0*((state[2]**3)/(np.linalg.norm(state[0:3])**2)))
    acc_J3[1] = pre_term * state[1] * (3.0 * state[2] - 7.0*((state[2]**3)/(np.linalg.norm(state[0:3])**2)))
    acc_J3[2] = pre_term * (6.0 * state[2]**2 - 7.0*((state[2]**4)/(np.linalg.norm(state[0:3])**2)) - (3.0/5.0)*(np.linalg.norm(state[0:3])**2))
    return acc_J3


def get_acc_J4(state: ArrayLike) -> np.ndarray:
    """Estimates the Accelaration due to J4 term

    Args:
        state (ArrayLike): State Vector at given time

    Returns:
        np.ndarray: Accelaration vector
    """
    acc_J4 = np.zeros(3)
    pre_term = -(15.0/8.0) * MU*J4 * ((R_EARTH**4)/(np.linalg.norm(state[0:3])**7))
    acc_J4[0] = pre_term * state[0] * ( 1.0
                                        - 14.0*((state[2]**2)/(np.linalg.norm(state[0:3])**2))
                                        + 21.0*((state[2]**4)/(np.linalg.norm(state[0:3])**4)))
    acc_J4[1] = pre_term * state[1] * ( 1.0
                                        - 14.0*((state[2]**2)/(np.linalg.norm(state[0:3])**2))
                                        + 21.0*((state[2]**4)/(np.linalg.norm(state[0:3])**4)))
    acc_J4[2] = pre_term * state[2] * ( 5.0
                                        - (70.0/3.0)*((state[2]**2)/(np.linalg.norm(state[0:3])**2))
                                        + 21.0*((state[2]**4)/(np.linalg.norm(state[0:3])**4)))
    return acc_J4