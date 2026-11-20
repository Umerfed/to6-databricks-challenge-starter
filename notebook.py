# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare the next team's dataset
# MAGIC Follow CHALLENGE.md. These headings are a starting point; implement and verify each stage yourself.

# COMMAND ----------

pwd

# COMMAND ----------

from pathlib import Path
import json

# Copy this Git folder's absolute path from its Workspace menu; include /Workspace/.
repo_root = Path("/Workspace/Users/umer.m.mohammad.irs@catalyte.ai/to6-databricks-challenge-starter")
assert (repo_root / "README.md").is_file(), "Check the Git folder path"
practice_result = spark.sql("SELECT 1 AS connection_check").first().asDict()
assert practice_result == {"connection_check": 1}
(repo_root / "outputs").mkdir(exist_ok=True)
(repo_root / "outputs/connection-practice.json").write_text(
    json.dumps(practice_result) + "\n", encoding="utf-8"
)

# COMMAND ----------

import hashlib

INPUT_SCHEMA = "training.to6_challenge3_inputs"
INITIAL_TABLE = f"{INPUT_SCHEMA}.returns_initial"
CORRECTED_TABLE = f"{INPUT_SCHEMA}.returns_corrected"
LINEAGE_VIEW = f"{INPUT_SCHEMA}.source_lineage"
workspace_user = spark.sql("SELECT current_user() AS username").first()["username"]
OUTPUT_SCHEMA = "training.to6_c3_" + hashlib.sha256(workspace_user.encode("utf-8")).hexdigest()[:12]
print({"input_schema": INPUT_SCHEMA, "output_schema": OUTPUT_SCHEMA})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understand the input
# MAGIC Record Catalog descriptions, sample observations and the supplied lineage source.

# COMMAND ----------

# DBTITLE 1,1a. Source reference: Catalog descriptions and lineage
# 1a. Source reference: what the Catalog and the supplied lineage view say about the inputs.
input_name = INPUT_SCHEMA.split(".")[1]

# Table descriptions (Catalog comments) for every object in the input schema
display(spark.sql(f"""
    SELECT table_name, table_type, comment
    FROM training.information_schema.tables
    WHERE table_schema = '{input_name}'
    ORDER BY table_name
"""))

# Column names, types and descriptions of the two delivery tables
display(spark.sql(f"""
    SELECT table_name, column_name, data_type, is_nullable, comment
    FROM training.information_schema.columns
    WHERE table_schema = '{input_name}'
      AND table_name IN ('returns_initial', 'returns_corrected')
    ORDER BY table_name, ordinal_position
"""))

# Supplied lineage view: which upstream table each delivery came from
display(spark.table(LINEAGE_VIEW))

# COMMAND ----------

# DBTITLE 1,1b. Sample rows from each batch
# 1b. At least three sample rows per batch, to see real values and types (ordered for a stable view).
for table in [INITIAL_TABLE, CORRECTED_TABLE]:
    print(table)
    display(spark.table(table).orderBy("return_id").limit(5))

# COMMAND ----------

# DBTITLE 1,1c. Full-batch profile by year
# 1c. Profile the FULL batches (not just samples): returns by year, missing required values,
# duplicate return IDs, zero income and negative amounts. Exploration only; the gate is in step 2.
REQUIRED = ["return_id", "tax_year", "income", "deduction"]
missing_any = " OR ".join(f"{c} IS NULL" for c in REQUIRED)

def profile_by_year(table):
    return spark.sql(f"""
        WITH b AS (
            SELECT *, COUNT(*) OVER (PARTITION BY return_id) AS id_copies FROM {table}
        )
        SELECT '{table.split('.')[-1]}' AS batch,
               CASE WHEN grouping(tax_year) = 1 THEN 'ALL' ELSE CAST(tax_year AS STRING) END AS year,
               COUNT(*)                                            AS rows,
               COUNT(DISTINCT return_id)                           AS distinct_return_ids,
               SUM(CASE WHEN {missing_any} THEN 1 ELSE 0 END)     AS rows_missing_required,
               SUM(CASE WHEN id_copies > 1 THEN 1 ELSE 0 END)     AS rows_with_duplicate_id,
               SUM(CASE WHEN income = 0 THEN 1 ELSE 0 END)        AS zero_income_rows,
               SUM(CASE WHEN income < 0 OR deduction < 0 THEN 1 ELSE 0 END) AS negative_amount_rows
        FROM b
        GROUP BY GROUPING SETS ((tax_year), ())
        ORDER BY year   -- '2022' < '2023' < '2024' < 'ALL' as strings
    """)

display(profile_by_year(INITIAL_TABLE).unionByName(profile_by_year(CORRECTED_TABLE)))

# Which records are behind the non-zero counts above
display(spark.sql(f"""
    WITH b AS (SELECT *, COUNT(*) OVER (PARTITION BY return_id) AS id_copies FROM {INITIAL_TABLE})
    SELECT * FROM b WHERE {missing_any} OR id_copies > 1 OR income = 0 ORDER BY return_id
"""))
display(spark.sql(f"SELECT * FROM {CORRECTED_TABLE} WHERE income = 0 ORDER BY return_id"))

# COMMAND ----------

# DBTITLE 1,1d. Input findings
# MAGIC %md
# MAGIC ### Source reference and findings
# MAGIC * **Source:** `training.to6_challenge3_inputs.returns_initial` and `returns_corrected`; the supplied lineage view `source_lineage` names `training.to6_challenge3_inputs.parsed_source` as the upstream table of both.
# MAGIC * **Row meaning:** one row is one return. `return_id` is the uniqueness identifier; `taxpayer_id` is the filer and recurs across years (e.g. `F0` files in 2022, 2023 and 2024), so it is **not** a key and is excluded from features. Required fields: `return_id`, `tax_year`, `income`, `deduction` (all INT/STRING and nullable in the Catalog, so nothing in the table enforces them).
# MAGIC * **returns_initial (full batch):** 13 rows but 12 distinct IDs - `R01` (2022) is delivered twice; `R02` (2022) has `income` NULL. Zero income: `R03` (2022), `R09` (2024). No negative amounts.
# MAGIC * **returns_corrected (full batch):** 12 rows, 12 distinct IDs, 4 per year 2022-2024, no missing required values, no duplicates, no negatives; same zero-income returns `R03`, `R09`.
# MAGIC
# MAGIC **Effect on next steps**
# MAGIC * Step 2 needs a completeness rule (required fields not null) and a uniqueness rule (each `return_id` once); both fire on `returns_initial`. A NULL income must not be filled with 0 and the duplicate must not be silently dropped - the batch is rejected instead.
# MAGIC * Step 3: zero income is valid, so the ratio must be NULL for `R03`/`R09` (no division error, no dropped rows).
# MAGIC * Step 4: 4 returns per year means expected split sizes of 4 / 4 / 4.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Decide which batch is usable
# MAGIC Keep the initial rejection evidence and apply the same checks to the replacement.

# COMMAND ----------

# DBTITLE 1,2a. Quality rules, gate and run context
# 2a. Quality rules applied identically to every batch, the gate that blocks feature writes,
# and the run context (run ID, repo paths, export helpers) used by later cells.
import csv, json, math
from datetime import datetime, timezone
from pathlib import Path
from pyspark.sql import functions as F, Window

# --- run context ---
RUN_TS = datetime.now(timezone.utc).replace(microsecond=0)
RUN_TS_UTC = RUN_TS.strftime("%Y-%m-%dT%H:%M:%SZ")
RUN_ID = RUN_TS.strftime("%Y%m%dT%H%M%SZ")
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
REPO_ROOT = Path("/Workspace" + _ctx.notebookPath().get()).parent
assert (REPO_ROOT / "CHALLENGE.md").is_file(), f"Notebook is not in the challenge Git folder: {REPO_ROOT}"
NOTEBOOK_URL = f"https://{_ctx.browserHostName().get()}/editor/notebooks/{_ctx.notebookId().get()}"
OUTPUTS = REPO_ROOT / "outputs"
RUNS_DIR = OUTPUTS / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def export_csv(df, relative_path):
    """Write a small table to outputs/ as CSV, keeping ints as ints and NULL as empty."""
    path = OUTPUTS / relative_path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(df.columns)
        for row in df.collect():
            writer.writerow(["" if v is None else v for v in row])
    return f"outputs/{relative_path}"

def export_json(obj, relative_path):
    (OUTPUTS / relative_path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return f"outputs/{relative_path}"

# --- rules ---
RULES = [
    "completeness: return_id, tax_year, income, deduction not null",
    "uniqueness: each return_id exactly once",
    "validity: amounts nonnegative",
]

class BatchRejected(Exception):
    """Raised when a batch fails a rule; nothing may be written from it."""

def failing_records(table):
    """Every row that breaks at least one rule, with all of its reasons. Nothing is dropped or repaired."""
    df = spark.table(table)
    source_cols = df.columns
    by_id = Window.partitionBy("return_id")
    df = (df.withColumn("id_copies", F.count(F.lit(1)).over(by_id))
            .withColumn("copy_number", F.row_number().over(by_id.orderBy(*source_cols))))
    reasons = [F.when(F.col(c).isNull(), F.lit(f"completeness: required field {c} is null")) for c in REQUIRED]
    reasons.append(F.when(F.col("id_copies") > 1,
                          F.concat(F.lit("uniqueness: return_id appears "), F.col("id_copies").cast("string"), F.lit(" times"))))
    reasons += [F.when(F.col(c) < 0, F.lit(f"validity: {c} is negative")) for c in ["income", "deduction"]]
    return (df.withColumn("rejection_reason", F.concat_ws("; ", *reasons))   # concat_ws skips rules that passed
              .where(F.col("rejection_reason") != "")
              .select(F.lit(table).alias("source_table"), *source_cols, "id_copies", "copy_number", "rejection_reason")
              .orderBy("return_id", "copy_number"))

def assess_batch(table):
    """Apply all rules to the full batch; returns (summary dict, DataFrame of failing records)."""
    bad = failing_records(table)
    bad_rows = bad.collect()
    totals = spark.table(table).agg(F.count(F.lit(1)).alias("rows"), F.countDistinct("return_id").alias("ids")).first()
    summary = {
        "table": table,
        "rows": totals["rows"],
        "distinct_return_ids": totals["ids"],
        "rows_failing": len(bad_rows),
        "failing_return_ids": sorted({r["return_id"] for r in bad_rows}, key=str),
        "rules": RULES,
        "decision": "ACCEPTED" if not bad_rows else "REJECTED",
    }
    return summary, bad

def require_accepted(table):
    """Gate: call before writing any feature output from `table`."""
    summary, _ = assess_batch(table)
    if summary["decision"] != "ACCEPTED":
        raise BatchRejected(f"{table} rejected: {summary['rows_failing']} failing row(s), "
                            f"return_ids {summary['failing_return_ids']}")
    return summary

print({"run_id": RUN_ID, "repo_root": str(REPO_ROOT), "notebook_url": NOTEBOOK_URL})

# COMMAND ----------

# DBTITLE 1,2b. Reject the initial batch and preserve affected records
# 2b. Assess returns_initial. The gate must refuse it; affected records and reasons are preserved
# in the output schema and in outputs/, unchanged (no dedup, no zero-fill).
initial_summary, initial_bad = assess_batch(INITIAL_TABLE)

try:
    require_accepted(INITIAL_TABLE)
    raise AssertionError("returns_initial passed the gate but should have been rejected")
except BatchRejected as rejection:
    initial_summary["gate_message"] = str(rejection)

assert initial_summary["decision"] == "REJECTED"
REJECTED_TABLE = f"{OUTPUT_SCHEMA}.initial_rejected_records"
initial_bad.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(REJECTED_TABLE)

initial_summary.update({
    "affected_records_table": REJECTED_TABLE,
    "affected_records_export": export_csv(spark.table(REJECTED_TABLE).orderBy("return_id", "copy_number"),
                                          "initial_rejected_records.csv"),
    "checked_at_utc": RUN_TS_UTC,
    "notebook_url": NOTEBOOK_URL,
    "next_action": "Do not build features from returns_initial; assess returns_corrected with the same rules.",
})
export_json(initial_summary, "initial_rejection.json")

print(initial_summary["gate_message"])
display(spark.table(REJECTED_TABLE).orderBy("return_id", "copy_number"))

# COMMAND ----------

# DBTITLE 1,2c. Assess the replacement batch with the same rules
# 2c. Same rules, same code, applied to returns_corrected. It must pass before step 3 writes anything.
corrected_summary, corrected_bad = assess_batch(CORRECTED_TABLE)
assert corrected_bad.count() == 0, "returns_corrected has failing records"
require_accepted(CORRECTED_TABLE)

quality_checks = {
    "checked_at_utc": RUN_TS_UTC,
    "notebook_url": NOTEBOOK_URL,
    "batches": [{k: initial_summary[k] for k in corrected_summary}, corrected_summary],
    "batch_used_for_features": CORRECTED_TABLE,
}
export_json(quality_checks, "quality_checks.json")
print(json.dumps(quality_checks["batches"], indent=2))

# COMMAND ----------

# DBTITLE 1,2d. Batch decision
# MAGIC %md
# MAGIC ### Decision
# MAGIC * **`returns_initial` is rejected** by the gate ("... rejected: 3 failing row(s), return_ids ['R01', 'R02']"):
# MAGIC   * uniqueness - `R01` (F0, 2022, income 100, deduction 20) is delivered twice, so the batch cannot provide one row per return;
# MAGIC   * completeness - `R02` (F1, 2022, deduction 33) has `income` NULL, so its ratio is unknown.
# MAGIC * All 3 affected rows are kept unchanged with their reasons in `initial_rejected_records` and `outputs/initial_rejected_records.csv`. The batch is **not** repaired: deduplicating would hide a delivery fault and filling income with 0 would invent a value.
# MAGIC * **Next action:** use the owner's replacement `returns_corrected`. Under the same rules it passes (12 rows, 12 distinct IDs, 0 failing rows) and is the only input for step 3.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Deliver a checked feature table
# MAGIC Implement the task's data contract and verify the saved result by reading it back.

# COMMAND ----------

# DBTITLE 1,3a. Build and save return_features (gated)
# 3a. return_features: exactly return_id, tax_year, deduction_to_income (= deduction / income).
# Zero income stays in the table with a NULL ratio. taxpayer_id, income, deduction are excluded (need-to-know).
FEATURES_TABLE = f"{OUTPUT_SCHEMA}.return_features"
FEATURE_COLUMNS = ["return_id", "tax_year", "deduction_to_income"]

def build_features(table):
    return spark.table(table).select(
        "return_id",
        "tax_year",
        F.when(F.col("income") != 0, F.col("deduction").cast("double") / F.col("income"))
         .otherwise(F.lit(None).cast("double"))
         .alias("deduction_to_income"),
    )

def save_return_features(table):
    require_accepted(table)                      # gate runs BEFORE any write
    (build_features(table).write.mode("overwrite")  # overwrite: a rerun replaces, never appends
        .option("overwriteSchema", "true").saveAsTable(FEATURES_TABLE))

# The rejected batch cannot reach the feature table
try:
    save_return_features(INITIAL_TABLE)
    raise AssertionError("returns_initial was written to return_features")
except BatchRejected as blocked:
    print("Blocked before write:", blocked)

save_return_features(CORRECTED_TABLE)
print("Saved", FEATURES_TABLE, "from", CORRECTED_TABLE)

# COMMAND ----------

# DBTITLE 1,3b. Read back and verify the saved feature table
# 3b. Verify the SAVED table (read back from Unity Catalog), independently of build_features():
# exact columns, one row per accepted return, values recomputed in plain Python from the source rows,
# a nonzero example and the zero-income cases.
saved = spark.table(FEATURES_TABLE)
assert saved.columns == FEATURE_COLUMNS, saved.columns

saved_rows = {r["return_id"]: r for r in saved.collect()}
source_rows = {r["return_id"]: r.asDict() for r in spark.table(CORRECTED_TABLE).collect()}
assert saved.count() == len(saved_rows), "duplicate return_id in return_features"
assert set(saved_rows) == set(source_rows), "saved IDs differ from the accepted batch"

for rid, src in source_rows.items():
    got = saved_rows[rid]
    expected = None if src["income"] == 0 else src["deduction"] / src["income"]
    assert got["tax_year"] == src["tax_year"], rid
    if expected is None:
        assert got["deduction_to_income"] is None, rid
    else:
        assert math.isclose(got["deduction_to_income"], expected, rel_tol=1e-12), rid

# Nonzero example chosen from the data (first return with nonzero income)
example_id = min(k for k, s in source_rows.items() if s["income"])
ex = source_rows[example_id]
expected_ratio = ex["deduction"] / ex["income"]
print(f"{example_id}: deduction {ex['deduction']} / income {ex['income']} = {expected_ratio}; "
      f"saved {saved_rows[example_id]['deduction_to_income']}")

zero_ids = sorted(k for k, s in source_rows.items() if s["income"] == 0)
assert zero_ids, "no zero-income return to check"
print("Zero-income returns kept with NULL ratio:", zero_ids)

feature_checks = {
    "table": FEATURES_TABLE,
    "source": CORRECTED_TABLE,
    "columns": saved.columns,
    "rows": len(saved_rows),
    "distinct_return_ids": len(saved_rows),
    "nonzero_example": {"return_id": example_id, "income": ex["income"], "deduction": ex["deduction"],
                        "expected_ratio": expected_ratio,
                        "saved_ratio": saved_rows[example_id]["deduction_to_income"]},
    "zero_income_cases": [{"return_id": k, "saved_ratio": saved_rows[k]["deduction_to_income"]} for k in zero_ids],
    "checked_at_utc": RUN_TS_UTC,
}
export_json(feature_checks, "feature_checks.json")
display(saved.orderBy("return_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Make the handoff usable
# MAGIC Save the required year tables. Keep references and IDs-and-values evidence for two successful runs.
# MAGIC Write SUBMISSION.md and the small exports inside this Databricks Git folder.

# COMMAND ----------

# DBTITLE 1,4a. Save and verify the year split tables
# 4a. Year splits built FROM THE SAVED return_features table. CREATE OR REPLACE: reruns replace, never append.
SPLITS = {"features_train": 2022, "features_validation": 2023, "features_test": 2024}
cols = ", ".join(FEATURE_COLUMNS)
for name, year in SPLITS.items():
    spark.sql(f"CREATE OR REPLACE TABLE {OUTPUT_SCHEMA}.{name} AS "
              f"SELECT {cols} FROM {FEATURES_TABLE} WHERE tax_year = {year}")

# Read back: each split holds only its year and equals return_features for that year;
# together they hold every accepted return exactly once.
features = spark.table(FEATURES_TABLE)
unassigned = features.where(~F.col("tax_year").isin(list(SPLITS.values()))).count()
assert unassigned == 0, f"{unassigned} accepted return(s) have a year outside the splits"

split_ids, split_counts = [], {}
for name, year in SPLITS.items():
    part = spark.table(f"{OUTPUT_SCHEMA}.{name}")
    assert part.columns == FEATURE_COLUMNS, (name, part.columns)
    assert part.where(F.col("tax_year") != year).count() == 0, f"{name} has rows outside {year}"
    expected = features.where(F.col("tax_year") == year)
    assert part.exceptAll(expected).count() == 0 and expected.exceptAll(part).count() == 0, f"{name} differs"
    ids = [r["return_id"] for r in part.select("return_id").collect()]
    split_ids += ids
    split_counts[name] = len(ids)

feature_ids = [r["return_id"] for r in features.select("return_id").collect()]
assert len(split_ids) == len(set(split_ids)), "a return appears in more than one split"
assert set(split_ids) == set(feature_ids), "splits do not cover every accepted return"
print({"return_features": len(feature_ids), **split_counts})

# COMMAND ----------

# DBTITLE 1,4b. Export tables and this run's snapshot
# 4b. Small exports of the saved tables, plus a per-run snapshot of every ID and value so that
# two runs can be compared on IDs AND values (not just row counts).
OUTPUT_TABLES = ["return_features", *SPLITS]
exports = [export_csv(spark.table(f"{OUTPUT_SCHEMA}.{t}").orderBy("return_id"), f"{t}.csv") for t in OUTPUT_TABLES]

snapshot_df = None
for t in OUTPUT_TABLES:
    part = spark.table(f"{OUTPUT_SCHEMA}.{t}").select(F.lit(t).alias("output_table"), *FEATURE_COLUMNS)
    snapshot_df = part if snapshot_df is None else snapshot_df.unionByName(part)
snapshot_path = export_csv(snapshot_df.orderBy("output_table", "return_id"), f"runs/run_{RUN_ID}.csv")

run_record = {
    "run_id": RUN_ID,
    "run_ts_utc": RUN_TS_UTC,
    "notebook_url": NOTEBOOK_URL,
    "input": CORRECTED_TABLE,
    "snapshot": snapshot_path,
    "row_counts": {t: spark.table(f"{OUTPUT_SCHEMA}.{t}").count() for t in OUTPUT_TABLES},
    # per-run copy of the initial rejection reference, so every run keeps its own evidence
    "initial_rejection": {k: initial_summary[k] for k in
                          ["table", "decision", "gate_message", "failing_return_ids", "affected_records_table"]},
}
export_json(run_record, f"runs/run_{RUN_ID}.json")
print(exports, snapshot_path)
print(json.dumps(run_record, indent=2))

# COMMAND ----------

# DBTITLE 1,4c. Compare the two latest complete runs
# 4c. Compare the two most recent run snapshots exactly (IDs and values, per table) and record the result.
def load_snapshot(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_table = {}
    for r in rows:
        by_table.setdefault(r["output_table"], []).append((r["return_id"], r["tax_year"], r["deduction_to_income"]))
    return {t: sorted(v) for t, v in by_table.items()}

snapshots = sorted(RUNS_DIR.glob("run_*.csv"))
if len(snapshots) < 2:
    print(f"{len(snapshots)} run snapshot(s) so far - run the whole notebook again to produce the second run.")
else:
    prev_path, curr_path = snapshots[-2], snapshots[-1]
    prev, curr = load_snapshot(prev_path), load_snapshot(curr_path)
    per_table = {}
    for t in OUTPUT_TABLES:
        a, b = prev.get(t, []), curr.get(t, [])
        per_table[t] = {
            "rows_run1": len(a), "rows_run2": len(b),
            "duplicate_ids_run1": len(a) - len({r[0] for r in a}),
            "duplicate_ids_run2": len(b) - len({r[0] for r in b}),
            "identical_ids_and_values": a == b,
        }
    ok = all(v["identical_ids_and_values"] and v["duplicate_ids_run1"] == 0 and v["duplicate_ids_run2"] == 0
             for v in per_table.values())
    def run_ref(p):
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        return {"run_ts_utc": meta["run_ts_utc"], "notebook_url": meta["notebook_url"], "snapshot": meta["snapshot"]}
    comparison = {
        "status": "MATCH" if ok else "MISMATCH",
        "run1": run_ref(prev_path),
        "run2": run_ref(curr_path),
        "per_table": per_table,
        "method": "Each run exports every row (output_table, return_id, tax_year, deduction_to_income) of the four "
                  "saved tables; the two latest snapshots are compared row by row per table after sorting.",
    }
    export_json(comparison, "run_comparison.json")
    print(json.dumps(comparison, indent=2))
    assert ok, "the two runs differ - investigate before submitting"

# COMMAND ----------

# DBTITLE 1,4d. Handoff
# MAGIC %md
# MAGIC ### Handoff
# MAGIC * **Inputs:** `training.to6_challenge3_inputs.returns_corrected` (accepted); `returns_initial` rejected (evidence in `initial_rejected_records`); upstream `parsed_source` per `source_lineage`.
# MAGIC * **Outputs** (assigned output schema `training.to6_c3_<hash>` printed by the setup cell): `return_features`, `features_train` (2022), `features_validation` (2023), `features_test` (2024), `initial_rejected_records`.
# MAGIC * **Ratio:** `deduction_to_income = deduction / income`; NULL when income is 0. Only `return_id`, `tax_year`, `deduction_to_income` are delivered.
# MAGIC * **Checks:** completeness, uniqueness and nonnegative-amount rules gate every feature write; readback of the saved table against a Python recomputation, a nonzero example and zero-income cases; splits checked for year, coverage and exactly-once; two complete runs compared on IDs and values (`outputs/run_comparison.json`).
# MAGIC * **Not verified:** no labels or model, so prediction-time availability and leakage are not assessed; the SQL that produced the deliveries from `parsed_source` was not inspected.
# MAGIC
# MAGIC Full details are in `SUBMISSION.md`.