# Domain suggestion data filtering and cleaning plan

Prepare the technology-to-domain examples in `row.json` for downstream training. Keep the raw file unchanged, send structurally valid examples to `filtered`, and write normalized, deduplicated examples to `cleaned`. This plan describes implementation work; no dataset processing has been applied yet.

All paths below are relative to `backend-models`.

## Current dataset

The source is `data/raw/row.json`, a valid JSON array containing 106 records and 96 distinct input values. Each record has `input`, `domain`, `profession`, `reason`, and `skills`. All records currently pass the basic schema checks below, with three or four skills per record and no exact duplicate records.

Docker, Kubernetes, Ansible, Redis, PostgreSQL, and Python each appear more than once. These represent multiple possible mappings and must not be removed simply because their input repeats. The content concerns technology and careers, so German language or CEFR filters do not apply.

## Output files

| File | Purpose |
| --- | --- |
| `data/raw/row.json` | Original source, preserved unchanged |
| `data/filtered/row.json` | Records that pass structural filtering |
| `data/filtered/rejected.json` | Original rejected records, source positions, and rejection reasons |
| `data/filtered/filter_report.json` | Source hash, rule version, counts, and warning totals |
| `data/cleaned/row.json` | Normalized records ready for review and splitting |
| `data/cleaned/change_log.json` | Source positions and exact field changes or duplicate removals |
| `data/cleaned/review_queue.json` | Possible semantic issues requiring human review |
| `data/cleaned/clean_report.json` | Counts, normalization statistics, and validation results |

Use UTF-8 JSON arrays for datasets, rejection records, change logs, and review queues. Use JSON objects for reports. Preserve the five-field dataset schema; put provenance in the sidecar logs. Identify source records by their one-based array position and the source file SHA-256 hash.

## Step 1 Validate the source

1. Read the source using UTF-8, allowing a leading byte order mark.
2. Require one complete JSON array. Reject duplicate object keys during parsing so values cannot be silently overwritten.
3. Stop with a useful error if parsing fails, the root is not an array, or the array is empty. Do not replace existing outputs on failure.
4. Record the source hash and input count for reproducibility.

## Step 2 Filter records

Apply these rules to each source record and collect all applicable rejection reasons:

| Check | Action |
| --- | --- |
| Record is not a JSON object | Reject with `invalid_record_type` |
| Any of the five required fields is missing | Reject with `missing_field` and field names |
| Unexpected fields are present | Reject with `unexpected_field`; retain the original record for inspection |
| `input`, `domain`, `profession`, or `reason` is not a nonempty string after whitespace trimming | Reject with `invalid_text_field` and field names |
| `skills` is not a nonempty array of nonempty strings | Reject with `invalid_skills` |
| A text value contains NUL or a non-whitespace control character | Reject with `invalid_control_character` |

Do not coerce numbers into text or split a string into a skills array automatically. Do not reject short valid names such as C, R, or Go. Do not reject multiple domains for the same input. Do not impose vocabulary or skill-count limits based solely on this small sample.

Write accepted records unchanged and in their original order to `data/filtered/row.json`. The current source should retain all 106 records under the basic schema rules; the implementation must verify the additional control-character checks before confirming its actual acceptance count.

## Step 3 Clean accepted records

Perform only deterministic transformations:

1. Normalize all string values to Unicode NFC.
2. Trim surrounding whitespace and collapse internal whitespace runs to a single space, including tabs, line breaks, and nonbreaking spaces.
3. Preserve meaningful punctuation and capitalization in names such as C++, C#, Node.js, Next.js, scikit-learn, and OAuth 2.0. Preserve the meaning of reasons; do not generate replacements or add skills.
4. Remove repeated skills within a record after whitespace and Unicode normalization, using case-insensitive comparison and preserving the first spelling and original order.
5. Validate the five-field schema again after normalization.
6. Remove exact duplicate normalized records, keeping the first source occurrence and recording every removed occurrence. Compare all scalar fields and the normalized skills list. Do not deduplicate by `input` alone.

Keep records that differ only in skill order or label spelling for review rather than assuming they are equivalent. If a record becomes invalid during cleaning, fail the run before publishing outputs and report its source position; do not silently discard it.

## Step 4 Review meaning and naming

Create a review queue for records sharing an input, domain, and profession but differing in reason or skills; possible aliases; and potentially overlapping domain labels. Queue entries must explain the concern and link to the original source positions. Repeated inputs with distinct domains are valid and are not automatically review failures.

For example, Ansible appears under Configuration Management and Server Automation. Review whether these are intentionally separate categories before considering a merge. Do not automatically expand SRE, rewrite TCP IP, or merge related job titles. Store any approved naming decisions in a small versioned alias mapping before applying them in a later run.

A reviewer should confirm that each reason explains the input-to-domain relationship and that each skill fits the profession. Structural validation cannot establish factual accuracy. Keep unresolved semantic flags in the review queue, and mark the dataset as requiring review rather than claiming it is fully verified for training.

## Step 5 Verify and publish

- Check `source_count = accepted_count + rejected_count`.
- Check `accepted_count = cleaned_count + removed_duplicate_count` after successful cleaning.
- Reparse every output file and validate every retained record against the schema.
- Verify that the source hash is unchanged and every output record has a traceable source position in the sidecar logs.
- Verify that distinct mappings for repeated inputs survive filtering and cleaning.
- Run cleaning a second time on the cleaned dataset and require identical dataset content. Running the full pipeline again with the same source and rule version must produce identical dataset files; timestamps may differ in reports.
- Write temporary files first, validate all of them, and publish the completed output set with a shared run identifier in the reports. If publication is interrupted, consumers must reject a mixed or incomplete run using the report hashes.

## Implementation sequence

1. Implement structural filtering in `scripts/filter_datasets.py` with explicit input and output-directory arguments.
2. Add `scripts/clean_datasets.py` for normalization, exact deduplication, provenance logging, and review queue generation.
3. Implement schema and count validation in `scripts/validate_filtered.py`, allowing validation of both output stages.
4. Add focused tests covering malformed JSON, duplicate object keys, missing fields, invalid skills, control characters, Unicode and whitespace normalization, skill duplicates, exact record duplicates, preservation of C++ and Node.js, preservation of multiple domains per input, and idempotence.
5. Run the pipeline on the source, inspect reports and queued examples, then resolve semantic questions before using the output for training.

Suggested commands after these interfaces are implemented, run from `backend-models`:

```powershell
python scripts/filter_datasets.py --input data/raw/row.json --output-dir data/filtered
python scripts/clean_datasets.py --input data/filtered/row.json --output-dir data/cleaned
python scripts/validate_filtered.py --input data/cleaned/row.json
```

## Later training preparation

Split only after cleaning and review. If evaluation should measure performance on unseen technologies, keep all records for an input and its approved aliases in the same split, using a fixed seed and recording the assignment. With only 106 examples, use the data first to validate the end-to-end pipeline; it does not establish broad domain coverage. Training splits, expansion, and model changes are separate work from this filtering and cleaning plan.
