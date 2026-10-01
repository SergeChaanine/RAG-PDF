# Measured comparison snapshot

Open [sample/report.html](sample/report.html) locally to present the study. The chart
is embedded, so the HTML file can be shared on its own. The report includes actual
measurements, settings, interpretation, and cases requiring inspection.

- 11 configurations; 22 prepared questions on the synthetic Aurora PDF.
- 232 successful answers; 10 answer requests failed because of Groq's daily token quota.
- All 11 configurations have retrieval evidence for the full question set.
- The three answer LLMs, three databases, both overlaps (15%/30%), and three chunk
  sizes completed their full answer runs. Qwen embeddings completed 12/22 answers.
- `paired_summary.json` uses the 12 questions successful in every configuration.
- `results.jsonl` preserves answers, evidence, calculator traces, timing, and errors.
- `results.csv` includes separate blank fields for human review; automatic checks
  are not human correctness or citation-support scores.

Embedding/database caches are kept outside this snapshot. The original sample PDF,
indexes, and evidence cache remain under `%LOCALAPPDATA%/RAG-PDF` on the machine used
for the study. See the project README for generation and resume commands. Resuming
a matching study retries failed answers and preserves successful rows.
