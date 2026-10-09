"""Deterministic table arithmetic using validated source cell references."""

import math
import re
import statistics
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


def _selected_rows(table, row_numbers):
    if row_numbers is not None:
        if (
            not isinstance(row_numbers, list) or not row_numbers
            or any(type(i) is not int or not 1 <= i <= len(table.rows) for i in row_numbers)
            or len(set(row_numbers)) != len(row_numbers)
        ):
            raise ValueError("Select distinct, valid one-based row numbers.")
        return [(i, table.rows[i - 1]) for i in row_numbers]
    return list(enumerate(table.rows, 1))


def _distribution_plot(table, source_number, y_columns, chart_type, title, row_numbers):
    if (
        not isinstance(y_columns, list) or len(y_columns) != 1
        or type(y_columns[0]) is not int or not 1 <= y_columns[0] <= len(table.headers)
    ):
        raise ValueError("Choose one numeric column for a distribution or summary.")
    column = y_columns[0]
    values, included, skipped = [], [], []
    for number, row in _selected_rows(table, row_numbers):
        label = unicodedata.normalize("NFKD", " ".join(row[:2])).casefold()
        label = "".join(c for c in label if unicodedata.category(c) != "Mn")
        if re.search(r"\b(?:total|subtotal|grand total|الاجمالي|اجمالي|المجموع|مجموع)\b", label):
            skipped.append(number)
            continue
        raw = row[column - 1].strip()
        if not raw:
            skipped.append(number)
            continue
        try:
            value = float(parse_numeric_value(raw))
        except ValueError as exc:
            raise ValueError(f"Row {number}, column {column} is not numeric.") from exc
        if not math.isfinite(value):
            raise ValueError("Values exceed figure limits.")
        values.append(value)
        included.append(number)
    if len(values) < 2:
        raise ValueError("At least two numeric observations are needed for a statistical figure.")
    summary = {
        "count": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "min": min(values), "max": max(values),
        "stdev": statistics.stdev(values),
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise ValueError("Summary statistics exceed figure limits.")
    return {
        "title": (title or table.caption or table.location)[:160],
        "chart_type": chart_type, "value_label": table.headers[column - 1],
        "values": values, "summary": summary, "row_numbers": included,
        "skipped_rows": skipped, "source_number": source_number,
        "source_location": table.location,
    }


def plot_table(
    sources, source_number, x_column=None, y_columns=None, chart_type="bar", title="",
    row_numbers=None,
):
    """Build chart data only from validated numeric cells in a complete table."""

    if not isinstance(source_number, int) or not 1 <= source_number <= len(sources):
        raise ValueError("Invalid source number.")
    table = sources[source_number - 1].table
    if table is None:
        raise ValueError("This source does not contain a complete table.")
    if len(table.rows) > 300:
        raise ValueError("Figures are limited to 300 table rows.")
    if chart_type in {"histogram", "box", "summary"}:
        return _distribution_plot(table, source_number, y_columns, chart_type, title, row_numbers)
    if chart_type not in {"line", "bar", "scatter"}:
        raise ValueError("Unsupported statistical figure type.")
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
    labels, series = [], {}
    series_labels = {column: f"{column}. {table.headers[column - 1]}" for column in y_columns}
    for label in series_labels.values():
        series[label] = []
    included = []
    for row_number, row in _selected_rows(table, row_numbers):
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
        included.append(row_number)
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
        "row_numbers": included,
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
        "must contain numeric data. Also use histogram for frequency distributions, box for "
        "median/quartiles/outliers, or summary for statistical metric cards. These three "
        "formats require exactly one numeric Y column and x_column=null. Use observations "
        "with consistent units and granularity, not a mixture of detail and summary rows. "
        "Total rows and blank values are excluded from distributions and summaries. "
        "Optional row_numbers selects the relevant period or population; coordinates are "
        "one-based from the numbered source table. Never invent data.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_number": {"type": "integer"},
                "x_column": {"type": ["integer", "null"]},
                "y_columns": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "scatter", "histogram", "box", "summary"],
                },
                "title": {"type": "string"},
                "row_numbers": {
                    "type": "array", "items": {"type": "integer"},
                    "minItems": 2, "maxItems": 300,
                },
            },
            "required": ["source_number", "x_column", "y_columns", "chart_type", "title"],
            "additionalProperties": False,
        },
    },
}
