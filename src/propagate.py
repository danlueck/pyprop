import numpy as np
from src.integrate import deriv, integrate_step_rk4
from src.io import initialize_dataframes, append_dataframe
from src.manoeuvre import apply_dv
from datetime import datetime, timedelta

def propagate(state: np.ndarray, params: dict, date_0: datetime, date_f: datetime) -> tuple[datetime, np.ndarray, dict, dict]:
    """Propagate the state from the starttime to the stop time

    Args:
        state (np.ndarray): Cartesian state vector in ECI frame
        params (dict): Dictionary of input paramets
        date_0 (datetime): Initial state date
        date_f (datetime): Propagation end date

    Returns:
        tuple[datetime, np.ndarray, dict, dict]:    Propagation end date,
                                                    Final state,
                                                    Dictionary of Cartesian states,
                                                    Dictionary of Keplerian states
    """
    current_date = date_0
    current_state = state

    df_kep, df_cart = initialize_dataframes(params=params)

    # Set up parameters for considering Manoeuvres
    manoeuvres = False
    if("instantaneous_manoeuvres" in params["satellite_properties"].keys()):
        manoeuvres = True
        man_time = date_0 = datetime(  params["satellite_properties"]["instantaneous_manoeuvres"]["epoch"]["year"],
                    params["satellite_properties"]["instantaneous_manoeuvres"]["epoch"]["month"],
                    params["satellite_properties"]["instantaneous_manoeuvres"]["epoch"]["day"],
                    params["satellite_properties"]["instantaneous_manoeuvres"]["epoch"]["hour"],
                    params["satellite_properties"]["instantaneous_manoeuvres"]["epoch"]["minute"],
                    int(params["satellite_properties"]["instantaneous_manoeuvres"]["epoch"]["second"]))
        manv = np.array([params["satellite_properties"]["instantaneous_manoeuvres"]["N"],
                        params["satellite_properties"]["instantaneous_manoeuvres"]["T"],
                        params["satellite_properties"]["instantaneous_manoeuvres"]["W"]])

    i = 0
    # Main Propagation Loop
    while(current_date<date_f):

        if((i%params["outputs"]["out_step"])==0):
            df_kep, df_cart = append_dataframe(params=params, state=current_state, date=current_date, df_kep=df_kep, df_cart=df_cart)        
        
        try:
            current_state = current_state + integrate_step_rk4(current_state, params["integration"]["step"], params, current_date, deriv)
            if(manoeuvres and current_date < man_time <= current_date + timedelta(0, params["integration"]["step"])):
                current_state[3:6] = current_state[3:6] + apply_dv(current_state, manv)
            current_date = current_date + timedelta(0, params["integration"]["step"]) 
            
        except Exception as e:
            print(e)
            break
        i += 1
    df_kep, df_cart = append_dataframe(params=params, state=current_state, date=current_date, df_kep=df_kep, df_cart=df_cart)     
    return current_date, current_state, df_cart, df_kep