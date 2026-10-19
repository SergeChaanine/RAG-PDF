"""Presentation of validated statistical figures; no model-generated chart code."""

VISUALIZATION_MODES = {
    "auto": "Automatic",
    "histogram": "Histogram",
    "box": "Box plot",
    "summary": "Summary cards",
    "scatter": "Scatter plot",
    "line": "Line chart",
    "bar": "Bar chart",
    "off": "Tables only",
}


def distribution_spec(chart):
    field = {"field": "value", "type": "quantitative", "title": chart["value_label"]}
    spec = {"data": {"values": [{"value": v} for v in chart["values"]]}, "height": 280}
    if chart["chart_type"] == "histogram":
        spec.update(
            mark={"type": "bar", "tooltip": True},
            encoding={
                "x": {**field, "bin": {"maxbins": 20}},
                "y": {
                    "aggregate": "count", "type": "quantitative",
                    "title": "التكرار" if chart.get("language") == "Arabic" else "Frequency",
                },
            },
        )
    elif chart["chart_type"] == "box":
        spec.update(mark={"type": "boxplot", "extent": 1.5}, encoding={"y": field})
    else:
        raise ValueError("This figure is not a distribution plot.")
    return spec


def summary_metrics(chart):
    labels = (
        ("عدد المشاهدات", "المتوسط", "الوسيط", "الأدنى", "الأعلى", "الانحراف المعياري للعينة")
        if chart.get("language") == "Arabic" else
        ("Observations", "Mean", "Median", "Minimum", "Maximum", "Sample standard deviation")
    )
    keys = ("count", "mean", "median", "min", "max", "stdev")
    return [
        (label, f"{chart['summary'][key]:,.6g}")
        for label, key in zip(labels, keys, strict=True)
    ]
