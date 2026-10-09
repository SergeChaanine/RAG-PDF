import json
from dataclasses import replace
from types import SimpleNamespace

import altair as alt
import pytest
from test_llm import model_with_client, response

from rag_pdf.calculations import plot_table
from rag_pdf.models import DocumentTable, SearchResult
from rag_pdf.visuals import distribution_spec, summary_metrics


def source():
    table = DocumentTable(
        "monthly", ("Month", "Sales USD"),
        (("Jan", "10"), ("Feb", "20"), ("Mar", "30"), ("2025 TOTAL", "60")),
        "Worksheet Monthly Summary, rows 1-5",
    )
    return SearchResult("a", table.render(), 0, 0, 1, table=table)


@pytest.mark.parametrize("kind", ["histogram", "box", "summary"])
def test_statistical_figures_use_real_observations_and_exclude_totals(kind):
    chart = plot_table([source()], 1, y_columns=[2], chart_type=kind)
    assert chart["values"] == [10, 20, 30]
    assert chart["row_numbers"] == [1, 2, 3]
    assert chart["skipped_rows"] == [4]
    assert chart["summary"] == {
        "count": 3, "mean": 20, "median": 20, "min": 10, "max": 30, "stdev": 10,
    }
    # Saved conversations round-trip the figure without losing provenance.
    assert json.loads(json.dumps(chart)) == chart
    assert chart["source_location"] == source().table.location
    if kind != "summary":
        spec = distribution_spec(chart)
        alt.Chart.from_dict(spec).to_dict(validate=True)
        assert spec["data"]["values"] == [{"value": 10}, {"value": 20}, {"value": 30}]


def test_single_numeric_column_and_arabic_values_are_supported():
    table = DocumentTable("x", ("الربح",), (("١٠",), ("٢٠",), ("٣٠",)), "ورقة الأرباح")
    chart = plot_table([replace(source(), table=table)], 1, y_columns=[1], chart_type="summary")
    chart["language"] = "Arabic"
    assert chart["summary"]["mean"] == 20
    assert summary_metrics(chart)[1] == ("المتوسط", "20")


def test_selected_period_and_arabic_total_do_not_pollute_distribution():
    table = replace(source().table, rows=(
        ("يناير", "10"), ("فبراير", ""), ("مارس", "30"), ("الإجمالي", "40"),
    ))
    chart = plot_table(
        [replace(source(), table=table)], 1, y_columns=[2], chart_type="box",
        row_numbers=[1, 2, 3, 4],
    )
    assert chart["values"] == [10, 30]
    assert chart["skipped_rows"] == [2, 4]
    subset = plot_table([source()], 1, y_columns=[2], chart_type="summary", row_numbers=[2, 3])
    assert subset["summary"]["mean"] == 25


@pytest.mark.parametrize("rows", [[0, 1], [1, 99], [1, 1], [], [True, 2]])
def test_invalid_row_selection_is_rejected(rows):
    with pytest.raises(ValueError, match="row numbers"):
        plot_table([source()], 1, y_columns=[2], chart_type="box", row_numbers=rows)


def test_incomplete_or_non_numeric_data_cannot_generate_a_figure():
    with pytest.raises(ValueError, match="complete table"):
        plot_table([replace(source(), table=None)], 1, y_columns=[2], chart_type="histogram")
    with pytest.raises(ValueError, match="not numeric"):
        plot_table([source()], 1, y_columns=[1], chart_type="histogram")
    with pytest.raises(ValueError, match="At least two"):
        plot_table([source()], 1, y_columns=[2], chart_type="histogram", row_numbers=[1, 4])


def tool_response(kind):
    arguments = {"source_number": 1, "x_column": None, "y_columns": [2], "chart_type": kind}
    call = SimpleNamespace(id="plot", function=SimpleNamespace(
        name="plot_table", arguments=json.dumps(arguments),
    ))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None, tool_calls=[call],
        model_dump=lambda **kwargs: {"role": "assistant", "tool_calls": []},
    ))], usage=None)


@pytest.mark.parametrize("kind", ["histogram", "box", "summary"])
def test_model_tool_call_becomes_a_renderable_figure(kind):
    model = model_with_client()
    model._client.chat.completions.create.side_effect = [
        tool_response(kind), response("The mean is 20 USD. [Source 1]"),
    ]
    model.answer("Describe the sales statistics.", [source()], visualization=kind)
    assert len(model.last_charts) == 1
    chart = model.last_charts[0]
    assert chart["chart_type"] == kind and chart["values"] == [10, 20, 30]
    assert chart["language"] == "English"


def test_tables_only_keeps_calculator_but_disables_figures():
    model = model_with_client()
    model._client.chat.completions.create.return_value = response("Sales rose. [Source 1]")
    model.answer("How did sales change over the months?", [source()], visualization="off")
    request = model._client.chat.completions.create.call_args.kwargs
    assert [tool["function"]["name"] for tool in request["tools"]] == ["calculate_table"]
    assert request["tool_choice"] == "auto"
    assert not model.last_charts


def test_different_figure_type_is_not_silently_used_for_selected_preference():
    model = model_with_client()
    model._client.chat.completions.create.side_effect = [
        tool_response("histogram"), response("Sales data is shown in the table. [Source 1]"),
    ]
    model.answer("Describe the sales statistics.", [source()], visualization="box")
    assert not model.last_charts
    messages = model._client.chat.completions.create.call_args.kwargs["messages"]
    assert "Use the selected figure type: box" in messages[-1]["content"]
