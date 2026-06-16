from numpy.typing import ArrayLike
from src.integrate import deriv, integrate_step_rk4
from src.constants import SEC_IN_DAY
from src.io import initialize_dataframes, append_dataframe
from datetime import datetime, timedelta
def propagate(state: ArrayLike, params: dict, date_0:datetime, date_f:datetime):
    current_date = date_0
    current_state = state

    df_kep, df_cart = initialize_dataframes(params=params)
    i = 0
    while(current_date<date_f):
        if((i%params["outputs"]["out_step"])==0):
            df_kep, df_cart = append_dataframe(params=params, state=current_state, date=current_date, df_kep=df_kep, df_cart=df_cart)        
        try:
            current_state = current_state + integrate_step_rk4(current_state, params["integration"]["step"], params, current_date, deriv)
            current_date = current_date + timedelta(0, params["integration"]["step"]) 
        except Exception as e:
            print(e)
            break
        i += 1
    df_kep, df_cart = append_dataframe(params=params, state=current_state, date=current_date, df_kep=df_kep, df_cart=df_cart)     
    return current_date, current_state, df_cart, df_kep