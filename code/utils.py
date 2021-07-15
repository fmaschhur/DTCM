from typing import List, Dict
import argparse

import pandas as pd


def get_cli_args(default_data_path: str) -> argspace.Namespace:
    """Parses the given client arguments.
    
    Reacts to the argument --input, which specifies the path
    to a CSV dataset.

    Parameters
    ----------
    default_data_path : str
        the default location of the expected CSV dataset.
        It is used, if the --input client argument is
        not specified
    """

    parser = argparse.ArgumentParser(description='Evaluate model on test data.')
    parser.add_argument('--input', default=default_data_path,
                        help='path of the test data file')
    return parser.parse_args()


def save_predictions(predictions: List[Dict], file_path: str):
    """Saves a list of predictions as a CSV file in the specified path.

    Parameters
    ----------
    predictions : List[Dict]
        a list of complexity score predictions, where
        each prediction consists of the according
        sentence id and the predicted MOS
    file_path : str
        a path specifying the location of the result CSV file   
    """

    df_predictions = pd.DataFrame.from_dict(predictions)
    df_predictions.to_csv(file_path, index=False)

