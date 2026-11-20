# My dataset submission

I built and ran everything in Databricks, in the notebook `notebook.py` in this Git folder.

## Inputs and outputs

I worked from two deliveries of synthetic returns: `training.to6_challenge3_inputs.returns_initial`, which I
rejected, and its replacement `training.to6_challenge3_inputs.returns_corrected`, which I accepted and used for
everything downstream. The lineage view `training.to6_challenge3_inputs.source_lineage` shows that both came from
`training.to6_challenge3_inputs.parsed_source`.

Each row is one return, and `return_id` should appear only once. `taxpayer_id` identifies the filer, and the same
filer can show up in several years, so it isn't a key. I left it out of the outputs, along with the raw `income` and
`deduction`, under the need-to-know rule.

All outputs are in my assigned schema, `training.to6_c3_ae9b7bd04b35`:

* `initial_rejected_records`: the 3 problem rows from the initial batch, with the reason each one failed
* `return_features`: 12 rows, one per accepted return
* `features_train` (2022), `features_validation` (2023) and `features_test` (2024): 4 rows each

The feature and split tables have exactly three columns: `return_id`, `tax_year` and `deduction_to_income`. The ratio
is deduction divided by income. When income is 0, the return stays in the table and the ratio is NULL.

## Checks and decisions

I applied the same three rules to both batches: every required field is filled in, each `return_id` appears once,
and no amount is negative. The notebook won't write any feature output until a batch passes.

The initial batch failed on two returns. `R01` was delivered twice, which breaks uniqueness, and `R02` has no income,
which breaks completeness. The gate stopped it with "3 failing row(s), return_ids ['R01', 'R02']". I kept all three
affected rows exactly as delivered, with their reasons, in `initial_rejected_records` and
`outputs/initial_rejected_records.csv`. I didn't try to fix the batch: dropping the duplicate would hide a delivery
error, and setting the missing income to 0 would invent a number. Instead I moved on to the replacement, which passed
all three rules (12 rows, 12 distinct IDs, no failures).

To check the saved feature table, I read it back from the catalog rather than trusting the code that wrote it:

* For a nonzero case I worked out the answer first. `R01` has deduction 20 and income 100, so the ratio should be
  0.2, and the saved table has 0.2.
* The table has exactly the three expected columns and 12 rows, one per accepted return. Every year and ratio matches
  a recalculation from the source.
* The two zero-income returns, `R03` and `R09`, are still there with a NULL ratio.
* Each split contains only its own year and matches `return_features` for that year, and every return lands in
  exactly one split.
* As a final independent check, I joined `return_features` to the source in a separate SQL query outside the
  notebook. It found no mismatched IDs, years or values.

## Two successful runs

I ran the whole notebook twice, interactively (no Jobs):

| Run | Notebook URL | Execution timestamp (UTC) | Snapshot |
| --- | --- | --- | --- |
| 1 | https://dbc-48a16403-f66f.cloud.databricks.com/editor/notebooks/127024866558565 | 2026-11-20T17:08:45Z | `outputs/runs/run_20261120T170845Z.csv` |
| 2 | https://dbc-48a16403-f66f.cloud.databricks.com/editor/notebooks/127024866558565 | 2026-11-20T17:10:50Z | `outputs/runs/run_20261120T171050Z.csv` |

After each run, the notebook saved a snapshot of every ID and value in the four feature tables.
`outputs/run_comparison.json` compares the two snapshots and reports **MATCH**: the IDs and values are identical in
all four tables, there are no duplicates, and both runs have 12 / 4 / 4 / 4 rows. Every table is written with
overwrite or `CREATE OR REPLACE`, so a rerun replaces the data instead of adding to it. The unchanged row counts in
run 2 confirm this.

## Final files and limitations

Everything in `outputs/` is part of the submission except `connection-practice.json`, which comes from the ungraded
connection practice:

* `return_features.csv`, `features_train.csv`, `features_validation.csv`, `features_test.csv`: exports of the saved
  tables
* `initial_rejected_records.csv` and `initial_rejection.json`: the rejected batch's problem rows, reasons and gate
  message
* `quality_checks.json`: rule results for both batches
* `feature_checks.json`: the nonzero and zero-income readback checks
* `runs/`: each run's snapshot and references
* `run_comparison.json`: the comparison of the two runs

There are two things I couldn't verify. There are no labels or model yet, so I can't say whether the ratio leaks
information or will be available at prediction time; the modelling team will need to assess that. I also relied on
the supplied lineage view and didn't inspect the SQL that built the deliveries from `parsed_source`.
