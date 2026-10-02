# Focused experiment protocol

Use one native PDF or DOCX containing prose and tables. The included synthetic
sample has 22 questions. Generate it with `python scripts/create_sample.py`, or
choose it in the app's Comparison study tab. Output paths are printed by the script.

Run the 11 one-factor configurations in `rag_pdf.benchmark.configurations()`.
The reference overlap is 15%; its only overlap variant is 30%. Keep the questions,
reference evidence, top K, prompts, and software environment fixed. Compare PDF and
DOCX extraction separately rather than uploading duplicate content into the study.

Question JSON is an array. Each item requires `id`, `question`, and `expected`.
Optional fields are `category`, `evidence` (reference substrings), `answer_patterns`
(Python regular expressions), and `history` (role/content turns). For retrieval-only
follow-up tests, `standalone_question` supplies the resolved query without an API call.

Review the source document and reference answers before a new study. Automatic
checks are only sanity checks, not semantic grading. Use the editable review table
to score correctness and citation support. Report failures and limitations alongside
the results, especially the small corpus and single-run timing noise.
