from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    array_contains,
    col,
    coalesce,
    concat,
    element_at,
    explode,
    length,
    lower,
    lit,
    rand,
    regexp_replace,
    size,
    split,
    sort_array,
    trim,
    when,
    arrays_zip,
)

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("StackOverflow_LLM_Preprocessing") \
    .getOrCreate()

# 2. Define S3 Paths
INPUT_PATH = "s3://netid-25jpkj-cloud-storage-project/raw-data/data"
OUTPUT_BASE_PATH = "s3://netid-25jpkj-cloud-storage-project/processed-data/"
EDA_STATS_PATH = "s3://netid-25jpkj-cloud-storage-project/eda-stats/"

# Quality / sampling (tune for your budget and rubric)
MIN_ANSWER_SCORE = 10
SAMPLE_SEED = 42
SAMPLE_N = 100_000
TAG_THRESHOLD = 100_000

REQUIRED_COLUMNS = [
    "title",
    "question",
    "answers",
    "answers_scores",
]

print("Reading raw Parquet data from S3...")
df = spark.read.parquet(INPUT_PATH)

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(
        f"Parquet schema missing expected columns {missing}. "
        f"Found: {df.columns}. "
        "This pipeline targets Hugging Face `suriyagunasekar/stackoverflow-with-meta-data`."
    )

# 3. Data Cleaning & best-answer selection
# HF schema: parallel arrays `answers` and `answers_scores` — pick the highest-scored answer.
print("Selecting highest-scored answer per row and filtering...")

pairs = arrays_zip(col("answers_scores"), col("answers"))
sorted_pairs = sort_array(pairs, asc=False)
best = element_at(sorted_pairs, 1)

with_best = (
    df.withColumn("_best", best)
    .withColumn("answer_body", col("_best").getField("answers"))
    .withColumn(
        "score",
        coalesce(col("_best").getField("answers_scores").cast("long"), lit(-1)),
    )
    .withColumn("question_body", coalesce(col("question"), lit("")))
    .drop("_best")
)

clean_df = (
    with_best.filter(size(col("answers")) > 0)
    .filter(col("score") > MIN_ANSWER_SCORE)
    .filter(col("title").isNotNull() & (length(trim(col("title"))) > 0))
    .filter(col("answer_body").isNotNull() & (length(trim(col("answer_body"))) > 0))
)

# 4. Choose training scope based on tag volume
if "tags" in clean_df.columns:
    tags_dtype = clean_df.schema["tags"].dataType.simpleString()

    if tags_dtype.startswith("array"):
        exploded_tags = clean_df.select(explode(col("tags")).alias("tag"))
    else:
        # Convert string-like tags (e.g. "<python><pandas>") into rows.
        normalized_tags = split(
            regexp_replace(
                regexp_replace(coalesce(col("tags").cast("string"), lit("")), "^<|>$", ""),
                "><",
                "|",
            ),
            "\\|",
        )
        exploded_tags = clean_df.select(explode(normalized_tags).alias("tag"))

    top_tag_row = (
        exploded_tags
        .filter(col("tag").isNotNull() & (length(trim(col("tag"))) > 0))
        .groupBy("tag")
        .count()
        .orderBy(col("count").desc(), col("tag").asc())
        .first()
    )

    if top_tag_row and top_tag_row["count"] >= TAG_THRESHOLD:
        selected_tag = top_tag_row["tag"]
        selected_tag_count = top_tag_row["count"]
        print(
            f"Top tag is '{selected_tag}' with {selected_tag_count} problems (>= {TAG_THRESHOLD}). "
            f"Using only '{selected_tag}' problems for fine-tuning."
        )

        if tags_dtype.startswith("array"):
            has_selected_tag = array_contains(col("tags"), selected_tag)
        else:
            has_selected_tag = lower(coalesce(col("tags").cast("string"), lit(""))).contains(
                selected_tag.lower()
            )

        training_scope = f"tag_{selected_tag}"
        training_df = clean_df.filter(has_selected_tag)
    else:
        if top_tag_row:
            print(
                f"Top tag '{top_tag_row['tag']}' has {top_tag_row['count']} problems (< {TAG_THRESHOLD}). "
                "Using all problems for fine-tuning."
            )
        else:
            print("No usable tags found. Using all problems for fine-tuning.")
        training_scope = "all"
        training_df = clean_df
else:
    print("No `tags` column found. Falling back to all problems for fine-tuning.")
    training_scope = "all"
    training_df = clean_df

OUTPUT_JSON_PATH = f"{OUTPUT_BASE_PATH}llm_training_data_{training_scope}/"

# 5. Formatting for LLM fine-tuning (instruction / response)
print("Formatting data into instruction-response pairs...")
formatted_df = training_df.withColumn(
    "text",
    concat(
        lit("<s>[INST] "),
        trim(col("title")),
        lit("\n\n"),
        trim(col("question_body")),
        lit(" [/INST] "),
        trim(col("answer_body")),
        lit(" </s>"),
    ),
)

# Random sample (fixed seed) before splitting — document N in your report
print(f"Sampling {SAMPLE_N} rows (seed={SAMPLE_SEED})...")
sampled_df = formatted_df.orderBy(rand(seed=SAMPLE_SEED)).limit(SAMPLE_N)

# 6. Train / validation / test split (single rand column — stable thresholds)
with_len = sampled_df.withColumn("text_length", length(col("text")))
split_rand = with_len.withColumn("_r_split", rand(seed=SAMPLE_SEED))
final_df = (
    split_rand.withColumn(
        "split",
        when(col("_r_split") < 0.8, "train")
        .when(col("_r_split") < 0.9, "validation")
        .otherwise("test"),
    )
    .drop("_r_split")
)

# 7. Save processed data — partitionBy preserves splits for training / eval
print("Saving processed JSON to S3 (partitioned by split)...")
print(f"Output path: {OUTPUT_JSON_PATH}")
(
    final_df.select("split", "text")
    .write.mode("overwrite")
    .partitionBy("split")
    .json(OUTPUT_JSON_PATH)
)

# 8. EDA outputs for the report
print("Calculating EDA statistics...")

length_stats = final_df.select("text_length").summary("min", "25%", "50%", "75%", "max")
length_stats.write.mode("overwrite").csv(EDA_STATS_PATH + "length_distribution/")

split_stats = final_df.groupBy("split").count()
split_stats.write.mode("overwrite").csv(EDA_STATS_PATH + "split_distribution/")

score_stats = final_df.select("score").summary("min", "25%", "50%", "75%", "max")
score_stats.write.mode("overwrite").csv(EDA_STATS_PATH + "chosen_answer_score_distribution/")

if "tags" in final_df.columns:
    tag_df = final_df.select(size(col("tags")).alias("tag_count"))
    tag_stats = tag_df.summary("min", "25%", "50%", "75%", "max")
    tag_stats.write.mode("overwrite").csv(EDA_STATS_PATH + "tag_count_distribution/")

print("Pipeline finished successfully!")
spark.stop()
