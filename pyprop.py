import argparse
from src.io import process_input, write_outputs, plot_results
from src.propagate import propagate
import time

def pyprop(input, output):
    params, state, date_0, date_f =  process_input(input)
    current_date, current_state, df_cart, df_kep = propagate(state=state, params=params, date_0=date_0, date_f=date_f)
    print(f"Final State at  {current_date} is {current_state}")
    write_outputs(df_kep, df_cart, output)
    plot_results(df_kep, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('propagator parser')
    parser.add_argument('-i', '--input', type=str, default="work/input/input.json", help='Location of Input json file')
    parser.add_argument('-o', '--output', type=str, default="work/output", help='Location of Output direcotry')
    args = parser.parse_args()
    tic = time.perf_counter()
    pyprop(input=args.input, output=args.output)

    toc = time.perf_counter()
    print(f"Propagation finished after {toc - tic:0.4f} seconds")