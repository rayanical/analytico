# Open-source column interpretation research

Research checked 2026-10-03 against first-party documentation and source. Moving `master` links describe inspected source, not a pinned dependency contract. These projects inform design; they have not been benchmarked as replacement classifiers on our datasets.

## Metabase: separate physical type, semantic metadata and UI hints

Metabase keeps database data types separate from semantic labels. A numeric column can explicitly be Quantity, Score, Currency or Category; text can be Description or Entity key. Semantic labels affect charts and formatting without casting stored values. Metadata can be edited or left unset. [Official semantic-type documentation](https://www.metabase.com/docs/latest/data-modeling/semantic-types).

Its automatic `infer-is-category` uses a distinct-count threshold of 30, but **`can-be-category?` restricts this path to Text/TextLike**, excludes known PK/FK, and the classifier requires an unset semantic type and some non-null values. The source describes Category as a frontend widget hint rather than a backend calculation rule. This does not support automatically treating low-cardinality numeric measurements as categories. [Category classifier](https://github.com/metabase/metabase/blob/master/src/metabase/analyze/classifiers/category.clj), [eligibility helper](https://github.com/metabase/metabase/blob/master/src/metabase/sync/util.clj).

`infer-semantic-type` also uses fingerprints: URL/email/JSON validation fractions have a 0.95 threshold; state validation uses 0.7. `can-edit-semantic-type?` preserves a user's existing selection while allowing refinement within the current analysis. This is evidence-driven inference with heuristics, not guaranteed business understanding. [Text fingerprint classifier](https://github.com/metabase/metabase/blob/master/src/metabase/analyze/classifiers/text_fingerprint.clj).

**Useful pattern for Analytico:** separate compatible storage types from optional semantics; retain evidence/provenance and overrides. Cardinality can select dropdowns or chart presentation without deciding summability or proving identity.

## Apache Superset: columns and aggregate metrics are separate objects

`TableColumn` stores physical `type`, grouping/filtering flags, temporal metadata and descriptions. `SqlMetric` separately stores an aggregate expression, metric type, formatting and currency. `TableColumn.is_numeric` uses engine-derived generic type; `is_temporal` honors explicit temporal metadata. Nothing in these inspected properties converts repeated numbers into categories or high-cardinality text into identifiers. This observation is limited to this model path, not a claim about every Superset feature. [Source: TableColumn, SqlMetric and their properties](https://github.com/apache/superset/blob/master/superset/connectors/sqla/models.py).

**Useful pattern for Analytico:** being numeric is a capability, not an instruction to sum. Separate reusable metric/aggregation definitions from column labels. Superset's dataset metadata does not automatically discover every business definition from an arbitrary CSV.

## YData Profiling: configurable heuristics, warnings and explicit schema

Its current official settings still use a numeric low-cardinality heuristic: `vars.num.low_categorical_threshold` defaults to 5 and can be disabled with 0. Conversely, categorical cardinality above 50 produces a warning; it does not establish identifier semantics. Users can supply partial `type_schema` and let remaining columns be inferred. String-length/character/word statistics are available, with some disabled by default. [Official settings and schema overrides](https://docs.profiling.ydata.ai/latest/advanced_settings/available_settings/).

**Useful pattern for Analytico:** make inference policy explicit and inspectable, keep warnings separate from classification, and accept user/source schema. Replacing our cutoff with another library's cutoff would not solve the underlying ambiguity. Full profiling is additional work; no performance benefit is established by this research.

## DuckDB: candidate elimination for physical types

DuckDB tests whether sampled values convert to candidate types, removing failed candidates and choosing the highest-priority survivor. VARCHAR is the fallback. Sampling defaults to 20,480 rows, can read the whole file with `sample_size=-1`, and can sample different positions in a seekable file. `all_varchar` preserves text for a separate conversion policy. Explicit type overrides are supported. This detects parseable types, not business roles or additive measures. [Official CSV auto-detection documentation](https://duckdb.org/docs/current/data/csv/auto_detection).

**For Analytico:** retain our source-text DuckDB ingestion and full-column conversion validation. A different CSV sniffer is not the main fix for passenger counts being mislabeled as categories. Do not replace safe source preservation with eager numeric casts that could lose identifier formatting.

## Frictionless: configurable schema inference with validation boundaries

Frictionless supports candidate field types and inference confidence: default confidence 0.9 allows an integer proposal when nine of ten values fit; setting 1 requires all considered values to conform. The default inference sample is 100 rows. Physical type, metadata and user schema are configurable. A 100% fit within a sample must not be interpreted as a proof for an unseen complete column. [Official Detector documentation](https://framework.frictionlessdata.io/docs/framework/detector.html).

**For Analytico:** use sample-based proposals followed by complete-column validation. Do not import the 90% threshold into a policy that silently changes or drops nonconforming data. This is a schema/validation pattern, not a ready-made business classifier.

## Vega-Lite: meaning belongs to the encoding, not just the stored value

Vega-Lite explicitly distinguishes primitive values from measurement types: the same numeric values can be quantitative, nominal or ordinal. An encoding can specify its intended measurement type; defaults also depend on aggregation, binning, scale and time units. A temporal field can be rendered as ordinal month categories. Thus analytical usage is partly a property of the question/chart, not a single universally correct label attached to the source column. [Official type documentation](https://vega.github.io/vega-lite/docs/type.html).

**For Analytico:** make grouping, numeric operations and preferred automatic usage separate capabilities. An integer may be usable as a dimension and, when appropriate, a measure. A low distinct count can favor a bar chart without establishing that summation is meaningless.

## Direct check in our repository

`intelligence.detect_semantic_type` and `DiskDataset._semantic_type` duplicate numeric cardinality and text cardinality rules. Fewer than 20 distinct numeric values selects categorical unless the uniqueness ratio is above 0.5; 50 or more distinct text values selects identifier. `_is_coded_numeric_column` adds another low-cardinality rule using `max(20, 5% of rows)`. Automatic chart selection further favors a category whose distinct count is closest to 10. These policies combine meaning, safety and presentation.

A diagnostic on the real UCI bank dataset reproduced the weakness without fabricated values: `campaign` (number of contacts) has 48 distinct values across 45,211 rows and is labeled metric. The first 500 real rows contain five distinct counts and are labeled categorical. Both are the same documented quantity. This subset check is not an accuracy benchmark; it demonstrates that a sample/filtered export can change the semantic label merely by changing observed diversity.

## Recommendation from these projects

Adopt their separation of concerns rather than importing their thresholds:

1. Preserve a factual physical type and compact profile independently of analytical meaning.
2. Keep low/high cardinality as evidence or presentation policy; it alone cannot prove category, identifier or additive measure.
3. Represent role, semantic subtype and aggregation separately, including unknown values and provenance.
4. Preserve explicit schema/user overrides across repeated inference.
5. Benchmark false confident decisions and automatic chart correctness, not just whether every column received a label.

The deterministic engine can confidently validate syntax and structure. Arbitrary codes, units and business metrics still need source definitions, user metadata or cautious model inference. None of these inspected projects demonstrates universal near-perfect semantic accuracy.

## Proposed first implementation

Use the existing DuckDB/pandas ingestion adapters and schema metadata rather than adding an entire profiling or BI dependency. Introduce one shared interpretation policy consuming compact facts from both engines, with distinct outputs for physical type, semantic evidence, available operations and preferred automatic usage.

Remove the two blanket implications: low numeric cardinality proves category; high text cardinality proves identifier. Preserve leading-zero/source-format safety, explicit role overrides and the existing row-position guard. Cardinality may rank display options, limit filters or recommend grouping, but must not alone relabel a number or deny an otherwise valid explicit numeric query. Missing meaning remains missing, and unsupported aggregation must not be silently invented.

For ambiguous numeric fields, a distribution/frequency view is an available initial insight; a confidently labeled code remains a dimension. The user can keep charting while optional Luna/schema metadata supplies meaning in the background. Review stays optional.

Compare this candidate with both current deterministic output and Luna on the complete four real fixtures, then add held-out datasets, prefix/subset stability and engine parity checks. Score role precision/coverage, incorrect automatic aggregations, chart correctness, full-column validation cost and memory. Do not promise a higher percentage before measuring it. No product behavior was changed by this research task.

## Implementation result

The shared policy is now implemented. See [the measured before/after and broader rule audit](deterministic-policy-benchmark-2026-10-03.md). Wrong assigned roles fell, but exact role coverage also fell; the report preserves that tradeoff and the remaining heuristics.

The [follow-up library comparison](column-inference-libraries-2026-10-03.md) distinguishes deterministic type systems, profilers, schema validators and learned semantic classifiers before selecting a Luna benchmark challenger.
