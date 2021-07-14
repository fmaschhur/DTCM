from typing import List, Dict
import argparse

import pandas as pd


def get_cli_args(default_data_path: str):
    parser = argparse.ArgumentParser(description='Evaluate model on test data.')
    parser.add_argument('--input', default=default_data_path,
                        help='path of the test data file')
    return parser.parse_args()


def save_predictions(predictions: List[Dict], file_path: str):
    df_predictions = pd.DataFrame.from_dict(predictions)
    df_predictions.to_csv(file_path, index=False)

