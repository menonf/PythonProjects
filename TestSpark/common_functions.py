from typing import Any, List
import numpy as np
from pyspark.sql import Column
from pyspark.sql.functions import col, rank
from pyspark.sql.window import Window


def get_max(list_items: List[Any]) -> Any:
    """
    | Return the maximum value from a given list.

    :param list_items: list of numbers or strings
    :return: number or string
    """
    try:
        return max(list_items)
    except Exception:
        raise Exception(f"Invalid list_type: {list_items}")


def get_min(list_items: List[Any]) -> Any:
    """
    | Return the minimum value from a given list.

    :param list_items: list of numbers or strings
    :return: number or string
    """
    try:
        return min(list_items)
    except Exception:
        raise Exception(f"Invalid list_type: {list_items}")


def crisp_rank(
    partition_by: List[Any], fields_order: List[Any], sort_direction: List[str]
) -> Column:
    """
    | Return back the list, ranked according to the specified order.

    | Mimics the sql RANK() window function that assigns a rank to each row within a partition of a result set.
    :partition_by: list of column names, can be numbers or strings
    :fields_order: list of column names, can be numbers or strings
    :sort_direction: list of sort order "Asc" or "Desc".
                    The number of items in sort_direction should match the number of columns in fields_order.
    :return: pyspark column
    """
    try:
        col_sort_order = []
        if len(fields_order) == len(sort_direction):
            for i, j in zip(fields_order, sort_direction):
                if j.lower() == "asc":
                    col_sort_order.append(col(i).asc())
                elif j.lower() == "desc":
                    col_sort_order.append(col(i).desc())
                else:
                    raise Exception(
                        f"Invalid Sort Order {j}. Only 'asc' or 'desc' is accepted"
                    )
            windowSpec = Window.partitionBy(partition_by).orderBy(col_sort_order)  # type: ignore
            return rank().over(windowSpec)
        else:
            raise Exception
    except Exception:
        if len(fields_order) != len(sort_direction):
            raise Exception(
                f"The number of items in sort_direction parameter \
                    does not match the number of columns in fields_order parameter"
            )
        else:
            raise Exception(f"Invalid list_type: {partition_by} or {fields_order}")


def weighted_average(weights: List[float], values: List[float]) -> np.float32:
    """
    | Return calculated weighted average.

    :param weights: list of floating point numbers
    :param values: list of floating point numbers
    :return: numpy floating point number
    """
    try:
        return np.float32(np.sum(np.multiply(weights, values)) / np.sum(weights))
    except Exception:
        raise Exception(f"Invalid list_type: {weights} or {values}")
