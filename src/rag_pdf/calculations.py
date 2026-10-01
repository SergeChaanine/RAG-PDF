"""Deterministic table arithmetic using validated source cell references."""

from decimal import Decimal, InvalidOperation


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
        normalized = raw.replace(",", "").replace("$", "").replace("%", "").strip()
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = "-" + normalized[1:-1]
        try:
            number = Decimal(normalized)
        except InvalidOperation as exc:
            raise ValueError(f"Cell ({row}, {column}) is not a plain numeric value: {raw}") from exc
        if not number.is_finite():
            raise ValueError("Non-finite numbers are not supported.")
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
