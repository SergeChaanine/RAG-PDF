"""Generate a labeled synthetic PDF with native text/tables and 22 reference questions."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from rag_pdf.config import Settings


def create_sample(output: Path | None = None):
    root = output or Settings.from_env().data_dir / "samples"
    root.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    for name in ("Title", "Heading1", "Heading2"):
        styles[name].textColor = colors.black
    styles["BodyText"].fontSize = 11
    styles["BodyText"].leading = 16
    story = []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 8))

    def table(headers, rows, widths):
        data = [[Paragraph(str(c), styles["BodyText"]) for c in row] for row in [headers, *rows]]
        item = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEAF3")),
                    ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#D9D9D9")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(item)
        story.append(Spacer(1, 14))

    from reportlab.platypus import PageBreak

    paragraph("Aurora Training Program Evaluation", "Title")
    paragraph("Synthetic benchmark document", "Heading2")
    paragraph(
        "This fictional report was created to test document retrieval and numerical "
        "question answering. All names, dates, and statistics are invented. "
        "The report covers the Aurora internship, historical participation, "
        "support desk activity, and the interpretation of a participant survey."
    )
    paragraph("Program and eligibility", "Heading1")
    paragraph(
        "Aurora is a supervised internship for second-year engineering students. "
        "The internship lasts eight weeks, with ten hours of supervised work each week. "
        "Maya Haddad is the program coordinator. The 2026 internship begins on 6 July 2026. "
        "Applications close on 15 June 2026. The program has capacity for 60 interns."
    )
    paragraph(
        "Applicants submit a short project proposal and an academic transcript. "
        "Selection considers the clarity of the proposed problem and the applicant's "
        "readiness to work with a supervisor. Prior industry employment is not required. "
        "Students are assigned to small project teams after acceptance, and each team "
        "meets a supervisor weekly. The program office handles placements; students "
        "do not need to find their own host organization."
    )
    paragraph("Assessment and support", "Heading1")
    paragraph(
        "Successful completion requires a final demonstration, a written technical report, "
        "and attendance at seven of the eight weekly supervisor meetings. Attendance is "
        "recorded by the supervisor after each meeting. The final demonstration explains "
        "the problem, method, and observed results. The report must describe limitations "
        "and include enough detail for another student to reproduce the work."
    )
    paragraph(
        "The support desk assists students with equipment bookings and software access. "
        "Its monthly visit counts measure interactions, not unique visitors: the same "
        "person can visit several times. Support desk visits therefore cannot be used "
        "as a count of enrolled interns. The desk is open to students outside Aurora, "
        "which explains why its monthly counts exceed the internship's capacity."
    )
    story.append(PageBreak())
    paragraph("Participation outcomes", "Heading1")
    paragraph(
        "Table 1 shows annual Aurora applications, enrollment, and completion. "
        "Each row represents one cohort. Counts are people, and all cohorts have "
        "finished their assessment period. Completion rate uses enrolled students "
        "as its denominator, not applicants."
    )
    table(
        ["Year", "Applicants", "Enrolled", "Completed"],
        [[2023, 80, 40, 32], [2024, 100, 50, 45], [2025, 120, 60, 54]],
        [65, 125, 125, 125],
    )
    paragraph(
        "Annual totals should be interpreted separately from support desk visits. "
        "Enrollment increased as supervision capacity expanded. These descriptive "
        "figures do not establish that any single teaching method caused a change "
        "in completion. The report contains no random assignment or control group."
    )
    paragraph("Participant survey", "Heading1")
    paragraph(
        "In the 2025 cohort, 48 of the 60 enrolled interns responded to a voluntary "
        "survey. Of those respondents, 36 rated the internship as very useful. "
        "The other respondents selected moderately useful or neutral. These "
        "results describe respondents only. Nonresponse bias may affect the findings "
        "because students who chose not to respond might have different experiences."
    )
    paragraph(
        "The survey was anonymous. The program did not link individual responses "
        "to final assessment scores. Therefore this report cannot answer whether "
        "respondents who liked the internship obtained higher marks. It also does "
        "not report post-graduation salaries, employment outcomes, or student ages."
    )
    story.append(PageBreak())
    paragraph("Monthly support desk activity", "Heading1")
    paragraph(
        "Table 2 reports support desk visits during 2025. The unit is visits. "
        "All twelve months are included, and there is no total row."
    )
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    table(["Month", "Visits"], [[m, 120 + i * 12] for i, m in enumerate(months)], [280, 160])
    paragraph(
        "Counting rule: repeated visits by one student are counted separately. "
        "The values are exact recorded counts, with no seasonal adjustment."
    )
    doc = SimpleDocTemplate(
        str(root / "sample_report.pdf"),
        pagesize=(612, 792),
        rightMargin=72,
        leftMargin=72,
        topMargin=48,
        bottomMargin=48,
        title="Aurora Training Program Evaluation",
    )
    doc.build(story)
    questions = []

    def q(text, expected, patterns, evidence, category="text", **extra):
        questions.append(
            {
                "id": f"q{len(questions) + 1:02d}",
                "question": text,
                "expected": expected,
                "answer_patterns": patterns,
                "evidence": evidence,
                "category": category,
                **extra,
            }
        )

    q(
        "Who is eligible for Aurora?",
        "Second-year engineering students.",
        [r"second.year", "engineering"],
        ["second-year engineering"],
    )
    q(
        "How long does the Aurora internship last?",
        "Eight weeks.",
        [r"(?:eight|8)\s*weeks"],
        ["eight weeks"],
    )
    q(
        "How many supervised hours are required each week?",
        "Ten hours.",
        [r"(?:ten|10)\s*hours"],
        ["ten hours"],
    )
    q("Who coordinates Aurora?", "Maya Haddad.", ["Maya Haddad"], ["Maya Haddad"])
    q(
        "When does the 2026 internship begin?",
        "6 July 2026.",
        [r"(?:6 July|July 6).*2026"],
        ["6 July 2026"],
    )
    q(
        "When do applications close?",
        "15 June 2026.",
        [r"(?:15 June|June 15).*2026"],
        ["15 June 2026"],
    )
    q("What is the internship capacity?", "60 interns.", [r"\b60\b"], ["60 interns"])
    q(
        "How many supervisor meetings must a student attend?",
        "Seven of eight meetings.",
        [r"(?:seven|7)", r"(?:eight|8)"],
        ["seven of the eight"],
    )
    q(
        "How many applicants were there in 2024?",
        "100 applicants.",
        [r"\b100\b"],
        ["2024", "100"],
        "table",
    )
    q(
        "How many interns completed Aurora in 2025?",
        "54 interns.",
        [r"\b54\b"],
        ["2025", "54"],
        "table",
    )
    q(
        "How many support desk visits occurred in March 2025?",
        "144 visits.",
        [r"\b144\b"],
        ["March", "144"],
        "table",
    )
    q(
        "Which month had the most support desk visits, and how many?",
        "December, 252 visits.",
        ["December", r"\b252\b"],
        ["December", "252"],
        "table",
    )
    q(
        "What was the total enrollment across the 2023, 2024, and 2025 cohorts?",
        "150 students (40 + 50 + 60).",
        [r"\b150\b"],
        ["40", "50", "60"],
        "calculation",
    )
    q(
        "How many more applicants were there in 2025 than in 2023?",
        "40 more applicants (120 - 80).",
        [r"\b40\b"],
        ["120", "80"],
        "calculation",
    )
    q(
        "What was the percentage increase in enrollment from 2023 to 2025?",
        "50 percent ((60 - 40) / 40 * 100).",
        [r"50\s*(?:%|percent)"],
        ["40", "60"],
        "calculation",
    )
    q(
        "What was the total number of support desk visits in 2025?",
        "2232 visits, summing all twelve months.",
        [r"2,?232"],
        ["January", "December", "252"],
        "calculation",
    )
    q(
        "What was the average monthly number of support desk visits in 2025?",
        "186 visits (2232 / 12).",
        [r"\b186\b"],
        ["January", "December"],
        "calculation",
    )
    q(
        "What percentage of enrolled interns responded to the 2025 survey?",
        "80 percent (48 / 60 * 100).",
        [r"80\s*(?:%|percent)"],
        ["48 of the 60"],
        "calculation",
    )
    q(
        "Why might the participant survey suffer from bias?",
        "Voluntary participation and nonresponse may make respondents unrepresentative.",
        [r"nonresponse|non.response|voluntary"],
        ["Nonresponse bias"],
        "text",
    )
    q(
        "Do support desk visit counts represent unique students?",
        "No. Repeat visits count separately and non-Aurora students can use the desk.",
        [r"not|no\b", r"repeat|several|multiple"],
        ["same person"],
        "text",
    )
    q(
        "How long is it?",
        "The Aurora internship lasts eight weeks.",
        [r"(?:eight|8)\s*weeks"],
        ["eight weeks"],
        "follow_up",
        standalone_question="How long does the Aurora internship last?",
        history=[
            {"role": "user", "content": "What is Aurora?"},
            {
                "role": "assistant",
                "content": "Aurora is a supervised internship "
                "for second-year engineering students.",
            },
        ],
    )
    q(
        "What is the average salary of Aurora graduates?",
        "The report does not provide this information.",
        [r"could not find|not (?:provide|report|contain)|not available"],
        [],
        "unanswerable",
    )
    (root / "sample_questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")
    return root / "sample_report.pdf", root / "sample_questions.json"
