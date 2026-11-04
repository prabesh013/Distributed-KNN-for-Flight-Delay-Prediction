# Scalable K-Nearest Neighbor Classification for Flight Delay Prediction Using Apache Spark

**CS 6240: Large-Scale Parallel Data Processing**
Northeastern University
Author: Prabesh Nepal

## Project Overview

This project implements **exact K-Nearest Neighbor (KNN)** classification for large-scale flight delay prediction using:

- a sequential Scala implementation
- a distributed Apache Spark implementation
- Amazon EMR for large-scale distributed experiments

The dataset is the **2024 U.S. Bureau of Transportation Statistics (BTS) Reporting Carrier On-Time Performance dataset**.

The project focuses primarily on **scalability and parallel performance**, rather than prediction accuracy alone.

For `N` training records and `M` evaluation records, exact KNN requires approximately:

```text
N x M
```

distance comparisons.

For the largest experiment in this project:

```text
Training records:   5,808,482
Evaluation records: 1,000

Distance comparisons:
5,808,482 x 1,000
= 5,808,482,000
```

The Spark implementation preserves **exact KNN semantics**.

Every evaluation record is compared against every selected training record. Each Spark partition keeps only its local top-K candidates for each evaluation record. Those candidates are then merged to recover the exact global top-K neighbors.

This avoids transferring every pairwise distance through the Spark shuffle while still producing the same predictions as sequential exact KNN.

---

## Results at a Glance

| Experiment                          |                              Result |
| ----------------------------------- | ----------------------------------: |
| Sequential vs Spark agreement       | 1,000 / 1,000 predictions identical |
| Selected K                          |                                   5 |
| Validation F1 for K = 5             |                              0.7303 |
| Best local Spark partition count    |                                   4 |
| EMR input scaling                   |                Approximately linear |
| Largest workload                    |    5.8 billion distance comparisons |
| EMR 1 to 4 worker speedup           |                               2.53x |
| EMR 1 to 4 worker runtime reduction |                               60.5% |
| December test accuracy              |                              0.9180 |
| December test precision             |                              0.8763 |
| December test recall                |                              0.7342 |
| December test F1                    |                              0.7990 |
| December majority baseline          |                              0.7780 |

---

## Technology Stack

- **Scala:** 2.12.17
- **Apache Spark:** 3.3.2
- **Amazon EMR:** 6.11.1
- **Java:** Java 11
- **Build Tool:** Maven
- **Automation:** GNU Make
- **Cloud Storage:** Amazon S3
- **Cluster Platform:** Amazon EMR
- **AWS Interface:** AWS CLI
- **Storage Format:** Apache Parquet

---

## Project Structure

```text
.
├── Makefile
├── pom.xml
├── README.md
│
├── src/
│   └── main/
│       └── scala/
│           └── knn/
│               ├── DataCheck.scala
│               ├── SequentialKNN.scala
│               ├── SparkKNN.scala
│               └── FullPreprocess.scala
│
├── data/
│   └── processed/
│       ├── train.parquet
│       ├── validation.parquet
│       └── test.parquet
│
├── output/
│   ├── partitions/
│   ├── scaling/
│   ├── k-validation.log
│   ├── sequential-small.log
│   ├── spark-small.log
│   ├── sequential-predictions.txt
│   └── spark-predictions.txt
│
└── evidence/
    └── aws/
```

Large raw BTS CSV files and processed Parquet datasets are not intended to be committed to GitHub.

---

# Dataset

## Source

The project uses the **Bureau of Transportation Statistics Reporting Carrier On-Time Performance dataset** for calendar year 2024.

The raw data consists of 12 monthly CSV files.

### Dataset Size

```text
Raw records:       7,079,061
Raw columns:             110
After cleaning:    6,965,247
```

---

## Temporal Split

The cleaned dataset is split chronologically.

| Split      | Period                       |   Records |
| ---------- | ---------------------------- | --------: |
| Training   | January through October 2024 | 5,808,482 |
| Validation | November 2024                |   571,933 |
| Test       | December 2024                |   584,832 |

The temporal split prevents future flight data from being used while developing or tuning the classifier.

The **November validation set** is used for selecting parameters such as K.

The **December test set** remains untouched until the final evaluation.

---

# Prediction Target

The classification target is:

```text
ArrDel15
```

where:

```text
0 = arrival delay less than 15 minutes
1 = arrival delay at least 15 minutes
```

Cancelled and diverted flights are removed during preprocessing.

---

# Features

Seven features are used for KNN distance calculation:

```text
DepDelayMinutes
CRSElapsedTime
Distance
DepTimeSin
DepTimeCos
DayOfWeekSin
DayOfWeekCos
```

## Feature Description

### DepDelayMinutes

Number of minutes by which the flight departed later than its scheduled departure time.

The prediction task is defined after departure information is available. Therefore, `DepDelayMinutes` is a valid predictor of eventual arrival delay for this task. Arrival-delay fields and actual arrival information are not used as input features.

### CRSElapsedTime

Scheduled elapsed flight time.

### Distance

Flight distance.

### DepTimeSin and DepTimeCos

Departure time is cyclic.

For example, 23:59 and 00:01 are very close in time even though their numeric values appear far apart.

Departure time is therefore converted into sine and cosine features.

### DayOfWeekSin and DayOfWeekCos

Day of week is also cyclic.

Sunday and Monday should be considered adjacent, so day of week is represented using sine and cosine transformations.

---

# Feature Standardization

Continuous features operate on very different numeric scales.

For example:

```text
DepDelayMinutes  -> tens or hundreds
CRSElapsedTime   -> hundreds
Distance         -> hundreds or thousands
```

Without scaling, large-valued features such as distance could dominate Euclidean distance.

Features are therefore standardized.

The standardization statistics are calculated using the **training set only**.

Validation and test data use the mean and standard deviation learned from the training set.

This prevents statistical leakage from future data.

---

# RecordId

A deterministic `RecordId` is generated during preprocessing.

`RecordId` is used for:

- deterministic sampling
- reproducible experiments
- deterministic nearest-neighbor tie breaking
- sequential versus Spark correctness comparison

`RecordId` is **not a KNN feature** and does not participate in distance calculations.

---

# Preprocessing Pipeline

The preprocessing pipeline is implemented in:

```text
src/main/scala/knn/FullPreprocess.scala
```

The preprocessing job performs the following steps:

1. Load all 12 monthly BTS CSV files.
2. Remove cancelled flights.
3. Remove diverted flights.
4. Remove rows containing invalid or missing required values.
5. Select the required columns.
6. Generate deterministic `RecordId` values.
7. Generate cyclic departure-time features.
8. Generate cyclic day-of-week features.
9. Split data chronologically into training, validation, and test sets.
10. Calculate feature statistics using the training set only.
11. Standardize features.
12. Write the processed datasets as Parquet files.

The final cleaned dataset contains:

```text
6,965,247 flights
```

---

# Exact KNN Algorithm

For every evaluation record:

1. Compute its distance to every training record.
2. Maintain only the nearest K records.
3. Use deterministic tie breaking when distances are equal.
4. Count the labels among the K nearest neighbors.
5. Predict the majority class.

Squared Euclidean distance is used:

```text
distance =
(x1 - y1)^2 +
(x2 - y2)^2 +
...
(xd - yd)^2
```

The square root is unnecessary because square root preserves the ordering of distances.

---

## Time Complexity

For:

```text
N = number of training records
M = number of evaluation records
d = number of features
```

the computation is approximately:

```text
O(N x M x d)
```

This project uses:

```text
d = 7
```

Since the number of features remains constant, the dominant workload is approximately:

```text
O(N x M)
```

This computational cost makes exact KNN a useful algorithm for studying distributed parallel processing.

---

# Sequential KNN

The sequential implementation is located in:

```text
src/main/scala/knn/SequentialKNN.scala
```

For each evaluation record, the program scans the selected training records and maintains a bounded priority queue containing only the K closest neighbors.

The sequential implementation serves two purposes:

1. provide a baseline for runtime comparisons
2. provide a correctness reference for the Spark implementation

---

# Distributed Spark KNN

The distributed implementation is located in:

```text
src/main/scala/knn/SparkKNN.scala
```

## Distributed Design

The training dataset is distributed across Spark partitions.

The relatively small evaluation sample is broadcast to the workers.

Conceptually:

```text
                      Evaluation Sample
                             |
                             |
                        Broadcast
                             |
            ---------------------------------
            |               |               |
            v               v               v
       Partition 1     Partition 2     Partition 3
       Training        Training        Training
       Records         Records         Records
            |               |               |
            v               v               v
        Local Top-K      Local Top-K      Local Top-K
            \               |               /
             \              |              /
              -----------------------------
                           |
                           v
                    Global Top-K Merge
                           |
                           v
                       Prediction
```

Each partition:

1. receives the evaluation records through Spark broadcast
2. scans its local training records
3. computes distances
4. maintains a bounded top-K structure for each evaluation record
5. emits only its local nearest-neighbor candidates

The local candidates from all partitions are then merged.

---

# Why Local Top-K Is Exact

Suppose there are `P` Spark partitions.

Each partition returns its local K nearest training records for a query.

Any training record that is not among the top K records within its own partition cannot belong to the global top K.

Therefore, it is sufficient to merge:

```text
P x K
```

candidates per evaluation record rather than transferring all:

```text
N
```

computed distances.

This reduces intermediate communication while preserving exact KNN results.

---

# Prerequisites

## Local Environment

Install:

- Java
- Maven
- Apache Spark 3.3.2
- Scala-compatible Spark runtime
- GNU Make

Verify the installation:

```bash
java -version
mvn -version
spark-submit --version
make --version
```

The Makefile is configured to use:

```makefile
local.master=local[4]
```

Local processed datasets are expected at:

```text
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
```

---

# Build

Build the application JAR:

```bash
make jar
```

Equivalent Maven commands:

```bash
mvn clean package
cp target/spark-demo-1.0.jar spark-demo.jar
```

The generated JAR is:

```text
spark-demo.jar
```

Clean the build:

```bash
make clean
```

---

# Local Sequential Execution

Run sequential exact KNN:

```bash
make local
```

Default parameters are:

```makefile
train.size=10000
eval.size=1000
knn.k=5
```

Parameters can be overridden.

Example:

```bash
make local \
  train.size=50000 \
  eval.size=1000 \
  knn.k=5
```

The validation dataset is used as the default evaluation dataset during model development.

---

# Local Spark Execution

Run the distributed implementation using Spark local mode:

```bash
make spark-local
```

Example:

```bash
make spark-local \
  train.size=200000 \
  eval.size=1000 \
  knn.k=5 \
  spark.partitions=4
```

The evaluation sample is broadcast while the training dataset remains partitioned.

---

# Correctness Verification

Before performing scalability experiments, the distributed implementation was verified against the sequential implementation.

Run:

```bash
make compare-small
```

The stronger comparison used:

```bash
make compare-small \
  compare.train.size=10000 \
  compare.eval.size=1000 \
  compare.k=5 \
  compare.partitions=4
```

This experiment:

1. runs sequential KNN
2. runs Spark KNN
3. uses the same deterministic training sample
4. uses the same deterministic evaluation sample
5. extracts aggregate metrics
6. extracts every individual prediction
7. compares the two prediction files

Important output files include:

```text
output/sequential-small.log
output/spark-small.log

output/sequential-metrics.txt
output/spark-metrics.txt

output/sequential-predictions.txt
output/spark-predictions.txt
```

## Result

```text
Sequential predictions: 1000
Spark predictions:      1000
Identical predictions:  1000
```

All **1,000 predictions matched exactly**.

This confirms that the Spark implementation preserves the behavior of the sequential exact KNN algorithm.

---

# K Selection

Candidate K values:

```text
1
3
5
7
11
15
```

Generate the combined validation log:

```bash
make k-validation-log
```

Results are saved to:

```text
output/k-validation.log
```

Validation results:

|   K | Validation F1 |
| --: | ------------: |
|   1 |        0.6651 |
|   3 |        0.7003 |
|   5 |    **0.7303** |
|   7 |        0.7251 |
|  11 |        0.7200 |
|  15 |        0.7093 |

The final selected value was:

```text
K = 5
```

`K = 5` was selected because it produced the highest validation F1 score among the tested values. The December test set was not used for K selection.

Evidence:

```text
output/k-validation.log
```

---

# Sequential Scalability Experiment

Run:

```bash
make sequential-scaling
```

Training sizes:

```text
10,000
25,000
50,000
100,000
200,000
```

Evaluation size remains fixed:

```text
1,000
```

Results are saved to:

```text
output/scaling/sequential.csv
```

Individual logs are stored under:

```text
output/scaling/sequential-*.log
```

This experiment measures how sequential KNN runtime changes as the training dataset grows.

---

# Local Spark Scalability Experiment

Run:

```bash
make spark-scaling
```

The experiment uses the same training sizes as the sequential experiment.

Results are stored in:

```text
output/scaling/spark.csv
```

Individual logs are stored under:

```text
output/scaling/spark-*.log
```

Measured local runtimes with 1,000 evaluation records and `K = 5` were:

| Training Records | Sequential Runtime | Spark Runtime |
| ---------------: | -----------------: | ------------: |
|           10,000 |           0.2430 s |      0.8267 s |
|           25,000 |           0.3577 s |      1.3259 s |
|           50,000 |           0.4974 s |      1.0872 s |
|          100,000 |           1.1401 s |      1.2951 s |
|          200,000 |           2.2767 s |      1.3488 s |

Spark is slower on the smaller local workloads because startup and scheduling overhead dominate. At 200,000 training records, Spark is approximately **1.69x faster** than the sequential implementation.

Evidence:

```text
output/scaling/sequential.csv
output/scaling/spark.csv
output/scaling/sequential-*.log
output/scaling/spark-*.log
```

---

# Spark Partition Experiment

Run:

```bash
make partition-experiments
```

The experiment tests:

```text
1 partition
2 partitions
4 partitions
8 partitions
```

while keeping the KNN workload constant.

Results are written to:

```text
output/partitions/results.csv
```

Logs are written to:

```text
output/partitions/spark-p1.log
output/partitions/spark-p2.log
output/partitions/spark-p4.log
output/partitions/spark-p8.log
```

## Result

The partition experiment used a fixed workload of 200,000 training records, 1,000 evaluation records, and `K = 5`.

| Spark Partitions |      Runtime |
| ---------------: | -----------: |
|                1 |     2.8305 s |
|                2 |     1.7991 s |
|                4 | **1.4060 s** |
|                8 |     1.5752 s |

Four partitions produced the best local runtime among the tested values. Increasing the partition count to eight increased overhead enough to make the run slightly slower.

Evidence:

```text
output/partitions/results.csv
output/partitions/spark-p1.log
output/partitions/spark-p2.log
output/partitions/spark-p4.log
output/partitions/spark-p8.log
```

---

# Full 2024 Preprocessing on Amazon EMR

The complete 2024 dataset consists of 12 monthly BTS CSV files.

Before running AWS targets, make sure aws.bucket.name and aws.subnet.id in the Makefile refer to resources available in your AWS account. These values can also be overridden from the command line. In the commands below, <bucket> represents the configured S3 bucket.

```
make aws-preprocess \
  aws.bucket.name=<YOUR_BUCKET> \
  aws.subnet.id=<YOUR_SUBNET>
```

Upload the raw files to:

```text
s3://<bucket>/raw/2024/csv/
```

Verify:

```bash
aws s3 ls s3://<bucket>/raw/2024/csv/
```

Run preprocessing:

```bash
make aws-preprocess
```

The Makefile:

1. builds the application JAR
2. uploads the JAR to S3
3. creates an Amazon EMR cluster
4. runs `knn.FullPreprocess`
5. writes the processed data to S3
6. automatically terminates the EMR cluster

Expected output:

```text
s3://<bucket>/processed/2024/train/
s3://<bucket>/processed/2024/validation/
s3://<bucket>/processed/2024/test/
```

Verify:

```bash
aws s3 ls s3://<bucket>/processed/2024/
```

---

# Running Distributed KNN on Amazon EMR

The primary AWS target is:

```bash
make aws
```

The EMR configuration fixes executor resources during worker-scaling experiments.

```text
spark.dynamicAllocation.enabled=false
spark.executor.cores=2
spark.executor.memory=2g
```

One executor is requested per core worker.

This allows the number of worker machines to change while executor size remains fixed.

Example validation run:

```bash
make aws \
  aws.eval=processed/2024/validation \
  train.size=5808482 \
  eval.size=1000 \
  knn.k=5 \
  spark.partitions=8 \
  aws.core.num.nodes=2
```

---

# EMR Input-Size Scaling

This experiment measures runtime while increasing only the number of training records. The experiment uses a deterministic sample of `1,000` November validation records, `K = 5`, `8` Spark partitions, `2` EMR core workers using `m4.large` instances, `2` executor cores, and `2` GB executor memory. All settings remain fixed while only `train.size` changes.

```
make aws \
  aws.eval=processed/2024/validation \
  train.size=<TRAIN_SIZE> \
  eval.size=1000 \
  knn.k=5 \
  spark.partitions=8 \
  aws.core.num.nodes=2
```

| Training Records | Distance Comparisons |    Runtime |
| ---------------: | -------------------: | ---------: |
|          500,000 |              0.500 B |  13.8914 s |
|        1,000,000 |              1.000 B |  25.3095 s |
|        2,000,000 |              2.000 B |  49.0768 s |
|        4,000,000 |              4.000 B |  96.5839 s |
|        5,808,482 |              5.808 B | 139.1833 s |

Runtime increased approximately linearly with training size. This is consistent with the expected `O(N x M)` behavior of exact KNN when the evaluation size and feature dimension are fixed.

The largest input-scaling run performed:

```text
5,808,482 x 1,000 = 5,808,482,000
```

exact distance comparisons.

---

# Worker Scaling Experiment

Worker scaling uses the following fixed configuration:

```text
Training records:   5,808,482
Evaluation records: 1,000
K:                   5
Spark partitions:    8
Instance type:       m4.large
Executor cores:      2
Executor memory:     2 GB
```

Only the number of core worker machines changes.

## 1 Worker

```bash
make aws \
  aws.eval=processed/2024/test \
  train.size=5808482 \
  eval.size=1000 \
  knn.k=5 \
  spark.partitions=8 \
  aws.core.num.nodes=1
```

## 2 Workers

```bash
make aws \
  aws.eval=processed/2024/test \
  train.size=5808482 \
  eval.size=1000 \
  knn.k=5 \
  spark.partitions=8 \
  aws.core.num.nodes=2
```

## 4 Workers

```bash
make aws \
  aws.eval=processed/2024/test \
  train.size=5808482 \
  eval.size=1000 \
  knn.k=5 \
  spark.partitions=8 \
  aws.core.num.nodes=4
```

Speedup is calculated using:

```text
S(p) = T(1) / T(p)
```

Parallel efficiency is:

```text
E(p) = S(p) / p
```

## Result

| Core Workers |    Runtime | Speedup | Parallel Efficiency |
| -----------: | ---------: | ------: | ------------------: |
|            1 | 135.1441 s |  1.000x |               1.000 |
|            2 |  91.3559 s |  1.479x |               0.740 |
|            4 |  53.3944 s |  2.531x |               0.633 |

Scaling from one worker to four workers reduced runtime by approximately **60.5%** and produced a **2.53x speedup**. All three worker configurations produced identical model metrics.

The speedup is substantial but not perfectly linear because distributed execution also includes:

- task scheduling
- serialization
- communication
- aggregation
- executor coordination
- fixed Spark overhead

---

# Final Held-Out Test

The final experiment evaluates previously untouched December 2024 flight records.

Configuration:

```text
Training records:     5,808,482
Evaluation records:       1,000
K:                            5
Spark partitions:             8
Core workers:                 2
Distance comparisons: 5,808,482,000
```

The December records were not used for K selection or other model tuning.

## Final Result

```text
True Negative:  755
False Positive: 23
False Negative: 59
True Positive:  163
```

Classification metrics:

```text
Spark KNN runtime: 91.3559 seconds

Accuracy:          0.9180
Precision:         0.8763
Recall:            0.7342
F1 Score:          0.7990
Baseline Accuracy: 0.7780
```

The two-worker final run therefore completed 5,808,482,000 exact distance comparisons in 91.3559 seconds. The classifier achieved **91.80% accuracy**, compared with the **77.80% majority-class baseline**.

The complete December split contains 584,832 records. Evaluating all December records against all 5,808,482 training records with exact brute-force KNN would require approximately 3.4 trillion distance comparisons, so the held-out evaluation uses a deterministic sample of 1,000 December records.

---

# Monitoring EMR Jobs

Running:

```bash
make aws
```

or:

```bash
make aws-preprocess
```

returns an EMR cluster ID.

Example:

```text
j-XXXXXXXXXXXXX
```

Check step status:

```bash
aws emr list-steps \
  --cluster-id <CLUSTER_ID> \
  --query 'Steps[*].[Name,Id,Status.State]' \
  --output table
```

Check cluster status:

```bash
aws emr describe-cluster \
  --cluster-id <CLUSTER_ID> \
  --query 'Cluster.Status' \
  --output json
```

Clusters are created using:

```text
--auto-terminate
```

so they automatically terminate after the configured steps finish.

---

# Retrieving EMR Logs

EMR logs are stored under:

```text
s3://<bucket>/knn-logs/
```

Locate the Spark driver output:

```bash
LOG_KEY=$(aws s3 ls \
  s3://<bucket>/knn-logs/<CLUSTER_ID>/containers/ \
  --recursive \
  | awk '/_01_000001\/stdout.gz$/ {print $4; exit}')
```

Download it:

```bash
aws s3 cp \
  "s3://<bucket>/$LOG_KEY" \
  /tmp/knn-run.gz
```

Extract the important experiment metrics:

```bash
gunzip -c /tmp/knn-run.gz | grep -E \
'Actual training sample:|Actual evaluation sample:|Distance comparisons:|Spark KNN runtime:|True Negative:|False Positive:|False Negative:|True Positive:|Accuracy:|Precision:|Recall:|F1 Score:|Baseline accuracy:'
```

---

# Experimental Output

Local experiment results are stored under:

```text
output/
```

Important files include:

```text
output/k-validation.log

output/sequential-small.log
output/spark-small.log

output/sequential-predictions.txt
output/spark-predictions.txt

output/scaling/sequential.csv
output/scaling/spark.csv

output/partitions/results.csv
```

AWS experiment evidence can be stored under:

```text
evidence/aws/
```

Large raw CSV and processed Parquet datasets should not be committed to GitHub.

---

# Reproducibility

The project uses deterministic sampling so sequential and Spark experiments operate on identical selected records. The deterministic `RecordId` is also used for exact tie breaking when two training records have equal distances.

## Experimental Controls

For controlled comparisons:

- use the same training sample size
- use the same evaluation sample size
- use the same deterministic evaluation sample
- use the same `K` value
- use the same evaluation split
- keep the partition count fixed unless partition count is the variable being studied
- keep executor cores and executor memory fixed during worker scaling
- change only the experimental variable being measured

The worker-scaling experiment fixes the workload at 5,808,482 training records, 1,000 evaluation records, `K = 5`, eight Spark partitions, two executor cores, and 2 GB executor memory. Only the number of EMR core workers changes.

## Recommended Reproduction Order

1. Build the application with `make jar`.
2. Verify sequential and Spark correctness with `make compare-small compare.train.size=10000 compare.eval.size=1000 compare.k=5 compare.partitions=4`.
3. Run K selection with `make k-validation-log`.
4. Run local sequential scaling with `make sequential-scaling`.
5. Run local Spark scaling with `make spark-scaling`.
6. Run partition experiments with `make partition-experiments`.
7. Upload the 2024 raw BTS CSV files and run `make aws-preprocess`.
8. Run EMR input-size experiments using the November validation split, `eval.size=1000`, `knn.k=5`, `spark.partitions=8`, and `2` EMR core workers, changing only `train.size`.
9. Run worker scaling with 1, 2, and 4 core workers while keeping the remaining configuration fixed.
10. Run the final held-out December evaluation only after model selection is complete.

The December test sample must not be used for selecting `K` or tuning the model.

---

# Key Findings

The experiments demonstrate several important characteristics of exact KNN on distributed systems.

### 1. Distributed correctness

The Spark implementation produced:

```text
1000 / 1000
```

predictions identical to sequential exact KNN.

Therefore, distributing the training data and merging partition-local top-K candidates does not change the semantics of the algorithm.

### 2. Sequential execution is competitive for small workloads

Spark introduces fixed costs such as:

- Spark initialization
- task creation
- scheduling
- serialization
- partition management

For small datasets, these costs can exceed the benefit of parallel execution.

### 3. Spark becomes more useful as workload increases

Exact KNN generates a large amount of independent distance computation.

As the training dataset grows, the amount of useful parallel work becomes large enough to better amortize Spark's overhead.

### 4. More partitions are not always better

The local experiments found:

```text
4 partitions
```

to be the best configuration among the tested values.

Too few partitions can underutilize available CPU resources, while too many partitions introduce additional scheduling overhead.

### 5. Exact KNN scales approximately linearly with input size

With evaluation size fixed, increasing training size increases the number of distance comparisons proportionally.

The EMR experiments showed approximately linear scaling from about:

```text
0.5 billion
```

to:

```text
5.8 billion
```

distance comparisons.

### 6. Additional workers improve performance

Increasing the EMR cluster from one worker to four workers produced:

```text
2.53x speedup
```

and approximately:

```text
60.5% runtime reduction
```

The scaling is useful but not perfectly linear because Spark includes distributed-system overhead.

---

# Limitations

This project intentionally uses **exact brute-force KNN** to study large-scale parallel processing.

As a result, prediction requires comparing every evaluation record with every selected training record.

For extremely large datasets or production machine-learning systems, approximate nearest-neighbor approaches may provide much lower latency.

The project does not attempt to replace exact KNN with an approximate method because doing so would change the computational problem being studied.

The evaluation sample is also intentionally limited in size during the largest experiments because the number of comparisons grows as:

```text
training_size x evaluation_size
```

Even 1,000 evaluation records against the full training dataset already requires more than:

```text
5.8 billion
```

distance computations.

---

# Clean Up

Clean local Maven and JAR artifacts:

```bash
make clean
```

EMR clusters use auto-termination, but cluster status should still be checked after experiments.

Large datasets and logs stored in S3 should be removed after the project is complete if they are no longer required.

---

# Repository

GitHub Repository:

https://github.com/nepalpra/LSPDP-Project (Private Khoury Server)
https://github.com/prabesh013/Distributed-KNN-for-Flight-Delay-Prediction (Public Github Server)

