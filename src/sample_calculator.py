import math
import os
import random

import pandas as pd

this_dir = os.path.dirname(os.path.abspath(__file__))
text_column = 'review'
res_column = 'category'
output_path = os.path.normpath(os.path.join(this_dir, '..', 'data', 'output'))

s_min = 0
s_max = 2**32 - 1

confidence_levels = [90, 95, 99]
z_scores = [1.645, 1.96, 2.576]
st_dev = 0.5


def calculate_sample(df, conf_level, conf_interval):
    # 3. Check for and remove duplicates
    initial_len = len(df)

    # Remove duplicates based on row IDs
    df.drop_duplicates(subset=['id'], inplace=True)

    final_len = len(df)
    if initial_len != final_len:
        print(f"Warning: Removed {initial_len - final_len} duplicated rows from the original dataset.")

    print(f"[CONFIDENCE LEVEL]: {conf_level}%")
    print(f"[CONFIDENCE INTERVAL]: {str(conf_interval)[:3]}%")

    dec_digits = str(conf_interval)[::-1].find('.')

    if dec_digits > 1:
        print("\nMore than one decimal digit detected.\nThe given interval will be sliced.")

    conf_interval = float(str(conf_interval)[:str(conf_interval).find('.') + 2]) / 100

    population = len(df)

    # 4. Calculate sample size
    ## Algorithm -> (n = N * Z^2 * p * (1 - p)) / (e^2 * (N - 1) + Z^2 * p * (1 - p))
    ### N = Population, Z = Z-Score, p = Standard Deviation, e = Confidence Interval
    z_sc = z_scores[confidence_levels.index(conf_level)]
    n_instances = math.ceil((population * pow(z_sc, 2) * st_dev * (1 - st_dev)) /
                            (pow(conf_interval, 2) * (population - 1) + pow(z_sc, 2) * st_dev * (1 - st_dev)))

    # 5. Sample generation and saving
    sample_df = df.sample(n=n_instances, random_state=random.randint(s_min, s_max))[['id', text_column, res_column]]
    sample_df['id'] = pd.factorize(sample_df['id'])[0] + 1
    return sample_df