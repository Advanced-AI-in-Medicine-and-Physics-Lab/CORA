# Plaque / stenosis label extraction

Volume-level plaque and stenosis labels used for the plaque-characterization and
stenosis-detection downstream tasks were extracted from free-text CCTA radiology
reports using **two large language models independently — GPT-4o and Claude
Sonnet 4.5** — each prompted to emit structured JSON. Records where the two
models disagreed were flagged for manual adjudication.

## Files

| File | Purpose |
|------|---------|
| `plaque_stenosis_extraction_prompt.txt` | The exact extraction prompt. `{{REPORT_TEXT}}` is replaced with the report body. |
| `plaque_label_schema.json` | JSON Schema the model output is validated against. |

## Output

Per report, one JSON object keyed by vessel (`LM`, `LAD`, `LCX`, `RCA`), each
with `calcified_plaque`, `non_calcified_plaque`, `stenosis_present`,
`stenosis_grade`, and `stenosis_percent`. Stenosis grades follow CAD-RADS
categories (0% none, 1–24% minimal, 25–49% mild, 50–69% moderate, 70–99%
severe, 100% occluded).

## Reproducing

1. Send each report's text in place of `{{REPORT_TEXT}}` to GPT-4o and to Claude
   Sonnet 4.5 with `temperature=0` and a JSON response format.
2. Validate each response against `plaque_label_schema.json`.
3. Keep records where both models agree; adjudicate disagreements.

No protected health information is included in this repository; only the prompt
and schema are released.
