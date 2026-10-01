# Focused RAG comparison

## Method

11 configurations change one factor at a time. Overlap is 15% or 30%. All use five retrieved chunks and the BGE-small reference tokenizer; model input limits are checked without truncation. Complete retrieved tables are expanded before answering. The answer-model variants reuse identical saved evidence. Follow-up rewriting uses GPT-OSS 20B for every configuration. Temperature is zero.

Measurements are a single pass on one small corpus. Retrieval includes query embedding. Recorded quota waiting is excluded from answer latency and reported separately. Model loading is recorded separately. Index timings with reused indexes/vectors are not cold-build timings. Local exact Qdrant/LanceDB and Chroma HNSW differ in search mode; these results do not measure server scalability.

## Results

| Configuration | Successful / attempted | Evidence checks | Answer checks | Top-1 anchors | Valid citation IDs | Retrieval mean ms | Answer mean s |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 22/22 | 21/21 | 21/22 | 14/21 | 20/21 | 37.8 | 8.37 |
| database_qdrant | 22/22 | 21/21 | 21/22 | 14/21 | 18/21 | 38.3 | 9.20 |
| database_lancedb | 22/22 | 21/21 | 21/22 | 14/21 | 19/21 | 46.9 | 8.48 |
| chunks_128 | 22/22 | 21/21 | 22/22 | 13/21 | 18/21 | 36.5 | 8.35 |
| chunks_480 | 22/22 | 21/21 | 21/22 | 12/21 | 20/21 | 30.6 | 10.00 |
| overlap_30 | 22/22 | 21/21 | 21/22 | 15/21 | 18/21 | 34.9 | 8.94 |
| llm_gpt_oss_120b | 22/22 | 21/21 | 22/22 | 14/21 | 21/21 | 37.8 | 8.75 |
| llm_qwen | 22/22 | 21/21 | 22/22 | 14/21 | 21/21 | 37.8 | 20.41 |
| embedding_e5 | 22/22 | 21/21 | 21/22 | 18/21 | 19/21 | 44.6 | 7.38 |
| embedding_bge_m3 | 22/22 | 21/21 | 21/22 | 14/21 | 18/21 | 103.1 | 4.38 |
| embedding_qwen | 12/22 | 21/21 | 11/12 | 15/21 | 10/12 | 94.4 | 8.94 |

## Interpretation and review

Evidence checks test reference substrings in the evidence supplied to the answer model. Answer checks test prepared regular expressions; neither is a human correctness score. Top-1 anchors inspect only the first saved raw hit, without another run or table expansion; they are a ranking diagnostic, not a top-1 answer experiment. Valid citation IDs do not prove citation support. Review answers and sources in results.jsonl, then fill correctness (0, 0.5, 1) and citation_support (0, 1) in results.csv or the UI. Failed answers are excluded from answer/citation check rates; successful cached retrieval is still scored independently. Failures remain explicit.

Use this as an exploratory case study, not a universal model/database ranking. The included sample is synthetic. Multi-document source selection and DOCX parsing have separate functional tests; this single-document study does not establish those capabilities.

## Presentation outline

1. Problem and pipeline: PDF/DOCX text and tables to cited conversational answers.
2. Dataset: document, questions, reference evidence, and synthetic-data limitation.
3. Experiment: reference configuration and ten one-factor variations.
4. Results: check rates and response times, followed by reviewed answer quality.
5. Examples: a table calculation, a follow-up question, and a failed answer.
6. Recommendation: quality/runtime trade-off within this tested corpus.

## Observed findings

- The reference has 6 indexed chunks and retrieves 5 per question. Broad context coverage and full-table expansion can conceal differences in retrieval ranking on this small document.
- Top-1 raw evidence anchors: BGE-small 14/21, E5-base 18/21, BGE-M3 14/21. This distinguishes ranking even when supplied-context anchor checks tie. Substring anchors remain an imperfect measure of useful evidence.
- 15% overlap has 21/21 supplied-context checks and 14/21 top-1 checks; 30% has 21/21 and 15/21, respectively. The sample gives limited evidence for increasing overlap.
- With identical retrieval evidence, GPT-OSS 20B/120B and Qwen 3.8 average 8.37/8.75/20.41s. Their valid citation-ID checks are 20/21, 21/21, and 21/21. Citation syntax is separate from whether the cited text supports the answer.
- Spot-review of 8 failed hours checks found correct answers such as '10 supervised hours'; the prepared regex expects '10 hours'. The strict automatic results are retained, rather than tuning reference patterns after seeing model answers. No human correctness score is claimed.
- Calculation spot-review: 40 + 50 + 60 = 150 enrolled interns; all twelve visit counts sum to 2,232, with mean 186; 48/60 = 80% survey participation. Saved traces also reveal wrong column selections corrected by the LLM and malformed citations/tool syntax despite correct final numbers. Numerical checks alone would miss those defects. The current app adds bounded citation/tool-output repair; the original measured answers remain available for inspection.
- For this small English sample, retain Chroma/BGE-small, 256 tokens, and 15% overlap as a practical starting point. E5 deserves consideration for ranking; GPT-OSS 120B improved citation formatting with similar observed answer time. Larger embeddings and Qwen answering did not establish a necessary benefit for this case. This conclusion does not extend to multilingual or larger corpora.
- Some requests failed. Compare the explicit denominators; incomplete configurations cannot support an equal-size quality comparison. Retrieval checks include saved evidence from requests whose answer API failed. paired_summary.json compares only questions successful for every configuration.
