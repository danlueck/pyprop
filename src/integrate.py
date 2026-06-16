
import numpy as np
from numpy.typing import ArrayLike

from src.gravity import get_acc_grav
from src.drag import get_acc_drag
from src.thridbody import get_acc_3rd_body
from src.srp import get_acc_srp
from datetime import datetime, timedelta

def deriv(state: ArrayLike, params: dict, date: datetime) -> np.ndarray:
    """_summary_

    Args:
        state (ArrayLike): _description_
        params (dict): _description_

    Returns:
        np.ndarray: _description_
    """
    satellite = params["satellite_properties"]["physical_properties"]
    perturbations = params["integration"]["perturbations"]

    acc = get_all_acc(state=state, perturbations=perturbations, satellite=satellite, date=date)
    vel = state[3:6]
    deriv = np.zeros(6)
    deriv[0:3] = vel
    deriv[3:6] = acc

    return deriv

def get_all_acc(state: ArrayLike, perturbations: dict, satellite: dict, date: datetime) -> np.ndarray:
    """_summary_

    Args:
        state (ArrayLike): _description_
        perturbations (dict): _description_

    Returns:
        np.ndarray: _description_
    """
    all_acc = get_acc_grav(state=state, geopotenital_order=perturbations["Geopotential"])
    if(perturbations["drag"]):
        all_acc += get_acc_drag(state=state, sat=satellite)
    if(perturbations["Moon"] or perturbations["Sun"]):
        all_acc += get_acc_3rd_body(state=state, date=date, bodies=perturbations)
    if(perturbations["srp"]):
        all_acc += get_acc_srp(state=state, date=date, satellite=satellite)

    return all_acc

def integrate_step_rk4(state: ArrayLike, step: float, params: dict, date:datetime, fun_deriv=deriv) -> np.ndarray:
    k1 = fun_deriv(state=state, params=params, date=date)
    state_1 = state + k1*(step/2)

    k2 = fun_deriv(state=state_1, params=params, date=(date+timedelta(0,0.5*step)))
    state_2 = state + k2*(step/2)

    k3 = fun_deriv(state=state_2, params=params, date=(date+timedelta(0,0.5*step)))
    state_3 = state + k3*(step)

    k4 = fun_deriv(state=state_3, params=params, date=(date+timedelta(0,step)))

    return (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)*step
