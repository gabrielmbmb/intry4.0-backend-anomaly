from typing import Union


def validate_data(columns, data, model_columns) -> Union[dict, None]:
    """
    Validates the train and predict data.

    Args:
        columns (list of str): provided columns names in payload.
        data (list of list of float): rows in payload.
        model_columns (list of str): columns in models.

    Returns:
        dict or None: errors found or None if not found.
    """
    errors = {"columns": "", "data": ""}

    # Check that columns provided are the same as the ones in the model
    if not all(column in model_columns for column in columns):
        errors["columns"] += (
            "The provided columns are not the same as those previously created in the "
            "Blackbox model."
        )

    # Check that data list is not empty
    if not data:
        errors["data"] += "Cannot be empty. "

    # Check if every row in training data has the same length
    rows_lengths = set(list(map(len, data)))
    if len(rows_lengths) != 1:
        errors["data"] += "Rows have to be the same length. "

    # Check if the rows length is equal to the columns length
    columns_length = len(columns)
    rows_length = rows_lengths.pop()
    if rows_length != columns_length:
        errors["data"] += (
            f"Rows have {rows_length} elements but {columns_length} columns were "
            "provided"
        )

    # If no errors...
    if all(error == "" for error in errors.values()):
        return {}

    return errors
