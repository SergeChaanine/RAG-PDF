import pytest
from test_llm import model_with_client, response

from rag_pdf.errors import ProviderError
from rag_pdf.models import SearchResult

CHART_JSON = (
    '{"type":"line","data":{"labels":["2024","2025","2027"],'
    '"datasets":[{"label":"Capacity","data":[310,520,650]}]}}'
)
SOURCE = SearchResult(
    "solar", "Capacity was 310 kW in 2024 and 520 kW in 2025. The 2027 target is 650 kW.",
    12, 0, 1,
)
ARABIC_TABLE = (
    "| السنة | القدرة (كيلوواط) | المصدر |\n"
    "| --- | --- | --- |\n"
    "| 2024 | 310 | [Source 1] |\n"
    "| 2025 | 520 | [Source 1] |\n"
    "| 2027 (مستهدف) | 650 | [Source 1] |"
)


@pytest.mark.parametrize("fenced", [True, False])
def test_arabic_pdf_chart_config_is_repaired_to_a_table_without_code(fenced):
    model = model_with_client()
    create = model._client.chat.completions.create
    leaked = f"```json\n{CHART_JSON}\n```" if fenced else CHART_JSON
    create.side_effect = [response(f"{ARABIC_TABLE}\n\n{leaked}"), response(ARABIC_TABLE)]

    answer = model.answer("ما خطة المشروع لزيادة القدرة بحلول عام 2027؟", [SOURCE])

    assert answer == ARABIC_TABLE
    assert create.call_count == 2
    request = create.call_args.kwargs
    assert "tools" not in request  # Figures in prose are not a chartable source table.
    assert SOURCE.text in request["messages"][1]["content"]
    repair = request["messages"][-1]["content"]
    assert "Remove all code blocks and chart configuration JSON" in repair
    assert model.last_charts == []


def test_repeated_chart_code_is_rejected_after_one_retry():
    model = model_with_client()
    create = model._client.chat.completions.create
    create.return_value = response(f"Capacity increased. [Source 1]\n```json\n{CHART_JSON}\n```")

    with pytest.raises(ProviderError, match="returned code"):
        model.answer("Show a chart of the expansion plan.", [SOURCE])

    assert create.call_count == 2
    assert model.last_charts == []


@pytest.mark.parametrize(
    "question,answer",
    [
        (
            "Write Python code to plot these capacities.",
            "The capacities are from [Source 1].\n```python\n"
            "import matplotlib.pyplot as plt\nplt.plot([2024,2025,2027],[310,520,650])\n```",
        ),
        (
            "اكتب كود لرسم هذه القيم.",
            "هذه القيم مأخوذة من [Source 1].\n```python\n"
            "import matplotlib.pyplot as plt\nplt.plot([2024,2025,2027],[310,520,650])\n```",
        ),
        (
            "Give me the Chart.js JSON for these capacities.",
            f"The capacities are from [Source 1].\n```json\n{CHART_JSON}\n```",
        ),
    ],
)
def test_explicit_code_request_still_allows_code_in_question_language(question, answer):
    model = model_with_client()
    create = model._client.chat.completions.create
    create.return_value = response(answer)

    assert model.answer(question, [SOURCE]) == answer
    assert create.call_count == 1


def test_without_code_request_does_not_allow_chart_json():
    model = model_with_client()
    create = model._client.chat.completions.create
    clean = "The 2027 target is 650 kW. [Source 1]"
    create.side_effect = [response(f"{clean}\n```json\n{CHART_JSON}\n```"), response(clean)]

    assert model.answer("Show the trend without code.", [SOURCE]) == clean
    assert create.call_count == 2
