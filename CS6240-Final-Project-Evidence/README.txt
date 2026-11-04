CS6240 Final Project Evidence
Scalable Exact KNN Classification for Flight Delay Prediction

Author: Prabesh Nepal

AWS Evidence
------------

worker1/
  EMR cluster: j-0839924MDV5YIWF55DN
  Workers: 1
  Training records: 5,808,482
  Evaluation records: 1,000
  K: 5
  Spark partitions: 8
  Runtime: 135.1441 seconds

worker2-final-test/
  EMR cluster: j-08702083PGNQ6V2LN7MH
  Workers: 2
  Training records: 5,808,482
  Evaluation records: 1,000 December test records
  K: 5
  Spark partitions: 8
  Runtime: 91.3559 seconds

  Final test metrics:
  Accuracy:  0.9180
  Precision: 0.8763
  Recall:    0.7342
  F1 Score:  0.7990
  Baseline accuracy: 0.7780

  predictions.txt contains the individual final-test predictions.

worker4/
  EMR cluster: j-09120981JLDQ1VTHFCSN
  Workers: 4
  Training records: 5,808,482
  Evaluation records: 1,000
  K: 5
  Spark partitions: 8
  Runtime: 53.3944 seconds

Worker Scaling
--------------
1 worker: 135.1441 s
2 workers: 91.3559 s, 1.479x speedup
4 workers: 53.3944 s, 2.531x speedup

All three AWS runs produced identical classification metrics.

Local Evidence
--------------
k-validation.log
  K-selection experiment.

sequential-small.log
spark-small.log
  Sequential/Spark correctness experiment.

sequential-predictions.txt
spark-predictions.txt
  Individual predictions used to verify exact agreement between
  sequential and distributed implementations.

sequential.csv
spark.csv
  Local input-size scaling results.

partition-results.csv
  Spark partition-count experiment.
