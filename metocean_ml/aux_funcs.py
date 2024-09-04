from __future__ import annotations

import pandas as pd
import numpy as np
import time


def convert_spec_to_dataframe(ds , spec_name = 'efth', dir_name='direction', freq_name='frequency'):
    
    # Reshape the data so that each combination of direction and frequency has its own column
    df_spec = (
        ds[spec_name].stack(combination=[dir_name, freq_name])  # Flatten direction and frequency dimensions
        .to_pandas()  # Convert to a pandas DataFrame
    )
    # Rename columns to reflect direction and frequency combinations
    df_spec.columns = [f"{spec_name}_{dir_name}{dir_idx}_{freq_name}{freq_idx}" 
                         for dir_idx, freq_idx in zip(*df_spec.columns.codes)]

    
    return df_spec

def create_zeros_dataframe_like(df):
    empty_df = 0*df #pd.DataFrame(columns=df.columns)
    
    return empty_df
