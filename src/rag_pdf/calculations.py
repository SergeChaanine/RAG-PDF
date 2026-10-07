"""Deterministic table arithmetic using validated source cell references."""

import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation


def parse_numeric_value(raw):
    """Parse plain numeric values, including Arabic-Indic digits and separators."""

    normalized = "".join(
        str(unicodedata.digit(char)) if char.isdigit() else char for char in str(raw)
    )
    normalized = normalized.translate(str.maketrans({"٫": ".", "٬": "", "−": "-", "٪": "%"}))
    normalized = normalized.replace(",", "").replace("$", "").replace("€", "")
    normalized = normalized.replace("£", "").replace("¥", "").replace("﷼", "")
    normalized = normalized.replace("%", "")
    normalized = re.sub(r"\s+", "", normalized).strip()
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = "-" + normalized[1:-1]
    try:
        number = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError(f"Not a plain numeric value: {raw}") from exc
    if not number.is_finite():
        raise ValueError("Non-finite numbers are not supported.")
    return number


def calculate_table(sources, source_number, operation, cells):
    if not isinstance(source_number, int) or not 1 <= source_number <= len(sources):
        raise ValueError("Invalid source number.")
    table = sources[source_number - 1].table
    if table is None:
        raise ValueError("This source does not contain a complete table.")
    if not cells or len(cells) > 1000:
        raise ValueError("Select between 1 and 1000 numeric cells.")
    values = []
    labels = []
    for cell in cells:
        row, column = cell["row"], cell["column"]
        if not (
            isinstance(row, int)
            and isinstance(column, int)
            and 1 <= row <= len(table.rows)
            and 1 <= column <= len(table.headers)
        ):
            raise ValueError("Cell coordinates must be valid one-based row/column numbers.")
        raw = table.rows[row - 1][column - 1].strip()
        try:
            number = parse_numeric_value(raw)
        except ValueError as exc:
            raise ValueError(f"Cell ({row}, {column}) is not numeric: {raw}") from exc
        values.append(number)
        labels.append(f"row {row}, {table.headers[column - 1]} = {raw}")
    if operation == "sum":
        value = sum(values)
    elif operation == "mean":
        value = sum(values) / len(values)
    elif operation == "min":
        value = min(values)
    elif operation == "max":
        value = max(values)
    elif operation in {"difference", "percent_change"}:
        if len(values) != 2:
            raise ValueError("Select exactly two cells, old value first and new value second.")
        value = values[1] - values[0]
        if operation == "percent_change":
            if values[0] == 0:
                raise ValueError("Percentage change from zero is undefined.")
            value = value / values[0] * 100
    else:
        raise ValueError("Unsupported operation.")
    return {
        "result": str(value),
        "operation": operation,
        "operands": labels,
        "source": source_number,
        "units": "percent" if operation == "percent_change" else "same units as the selected cells",
    }


def plot_table(sources, source_number, x_column, y_columns, chart_type, title=""):
    """Build chart data only from validated numeric cells in a complete table."""

    if not isinstance(source_number, int) or not 1 <= source_number <= len(sources):
        raise ValueError("Invalid source number.")
    table = sources[source_number - 1].table
    if table is None:
        raise ValueError("This source does not contain a complete table.")
    if chart_type not in {"line", "bar", "scatter"}:
        raise ValueError("Chart type must be line, bar, or scatter.")
    if not isinstance(x_column, int) or not 1 <= x_column <= len(table.headers):
        raise ValueError("X column must be a valid one-based table column.")
    if (
        not isinstance(y_columns, list)
        or not 1 <= len(y_columns) <= 5
        or any(
            not isinstance(column, int)
            or not 1 <= column <= len(table.headers)
            or column == x_column
            for column in y_columns
        )
        or len(set(y_columns)) != len(y_columns)
    ):
        raise ValueError("Choose one to five distinct numeric Y columns, excluding X.")
    if len(table.rows) > 300:
        raise ValueError("Charts are limited to 300 table rows.")

    labels, series = [], {}
    series_labels = {column: f"{column}. {table.headers[column - 1]}" for column in y_columns}
    for label in series_labels.values():
        series[label] = []
    for row_number, row in enumerate(table.rows, 1):
        label = row[x_column - 1].strip()
        if not label:
            continue
        if chart_type == "scatter":
            try:
                label = float(parse_numeric_value(label))
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_number}, X column {x_column} is not numeric for a scatter chart."
                ) from exc
            if not math.isfinite(label):
                raise ValueError("Scatter X values must be finite numbers.")
        parsed = {}
        for column in y_columns:
            raw = row[column - 1].strip()
            if not raw:
                continue
            try:
                parsed[column] = float(parse_numeric_value(raw))
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_number}, column {column} is not numeric; no chart was created."
                ) from exc
            if not math.isfinite(parsed[column]):
                raise ValueError(f"Row {row_number}, column {column} exceeds chart limits.")
        if not parsed:
            continue
        labels.append(label)
        for column in y_columns:
            series[series_labels[column]].append(parsed.get(column))
    if len(labels) < 2:
        raise ValueError("At least two rows with numeric data are needed for a chart.")

    return {
        "title": (title or table.caption or table.location)[:160],
        "chart_type": chart_type,
        "x_label": table.headers[x_column - 1],
        "y_labels": [series_labels[column] for column in y_columns],
        "labels": labels,
        "series": series,
        "source_number": source_number,
        "source_location": table.location,
    }


CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate_table",
        "description": "Calculate from complete source tables. Coordinates are one-based. "
        "For difference/percent_change select old then new. Include all "
        "relevant rows for totals; do not mix units or double-count totals.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_number": {"type": "integer"},
                "operation": {
                    "type": "string",
                    "enum": ["sum", "mean", "min", "max", "difference", "percent_change"],
                },
                "cells": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"row": {"type": "integer"}, "column": {"type": "integer"}},
                        "required": ["row", "column"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["source_number", "operation", "cells"],
            "additionalProperties": False,
        },
    },
}


PLOT_TABLE_TOOL = {
    "type": "function",
    "function": {
        "name": "plot_table",
        "description": "Create a chart only when a complete retrieved source table contains "
        "a useful statistical relationship or trend for the user's question. Use line for "
        "time trends, bar for category comparisons, scatter for relationships between two "
        "numeric columns. X column is a category/date or the first numeric variable; Y columns "
        "must contain numeric data. All column numbers are one-based from the numbered headers.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_number": {"type": "integer"},
                "x_column": {"type": "integer"},
                "y_columns": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "chart_type": {"type": "string", "enum": ["line", "bar", "scatter"]},
                "title": {"type": "string"},
            },
            "required": ["source_number", "x_column", "y_columns", "chart_type", "title"],
            "additionalProperties": False,
        },
    },
}
