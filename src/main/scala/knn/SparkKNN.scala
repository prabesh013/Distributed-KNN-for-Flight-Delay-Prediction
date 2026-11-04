package knn

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, xxhash64}

import scala.collection.mutable.PriorityQueue


object SparkKNN {

  // ============================================================
  // Data structures
  // ============================================================

  case class Flight(
    recordId: Long,
    features: Array[Double],
    label: Int
  )


  case class Neighbor(
    distance: Double,
    recordId: Long,
    label: Int
  )


  // ============================================================
  // Neighbor ordering
  //
  // Scala PriorityQueue is a MAX heap.
  //
  // The "worst" neighbor should therefore be at the top:
  //
  // 1. Larger distance is worse.
  // 2. If distances are equal, larger RecordId is worse.
  //
  // This matches SequentialKNN.
  // ============================================================

  implicit val neighborOrdering: Ordering[Neighbor] =
    new Ordering[Neighbor] {

      override def compare(
        a: Neighbor,
        b: Neighbor
      ): Int = {

        val distanceComparison =
          java.lang.Double.compare(
            a.distance,
            b.distance
          )

        if (distanceComparison != 0) {

          distanceComparison

        }
        else {

          java.lang.Long.compare(
            a.recordId,
            b.recordId
          )
        }
      }
    }


  // ============================================================
  // Feature configuration
  // ============================================================

  val recordIdColumn: String =
    "RecordId"


  val featureColumns: Array[String] = Array(
    "DepDelayMinutes",
    "CRSElapsedTime",
    "Distance",
    "DepTimeSin",
    "DepTimeCos",
    "DayOfWeekSin",
    "DayOfWeekCos"
  )


  val targetColumn: String =
    "ArrDel15"


  // ============================================================
  // Squared Euclidean distance
  // ============================================================

  def squaredEuclideanDistance(
    a: Array[Double],
    b: Array[Double]
  ): Double = {

    var sum = 0.0
    var i = 0

    while (i < a.length) {

      val difference =
        a(i) - b(i)

      sum +=
        difference * difference

      i += 1
    }

    sum
  }


  // ============================================================
  // Is candidate better than current worst neighbor?
  // ============================================================

  def isBetter(
    candidate: Neighbor,
    currentWorst: Neighbor
  ): Boolean = {

    val distanceComparison =
      java.lang.Double.compare(
        candidate.distance,
        currentWorst.distance
      )

    if (distanceComparison < 0) {

      true

    }
    else if (distanceComparison > 0) {

      false

    }
    else {

      candidate.recordId <
        currentWorst.recordId
    }
  }


  // ============================================================
  // Add neighbor to bounded Top-K heap
  // ============================================================

  def addNeighbor(
    queue: PriorityQueue[Neighbor],
    candidate: Neighbor,
    k: Int
  ): Unit = {

    if (queue.size < k) {

      queue.enqueue(candidate)

    }
    else if (
      isBetter(
        candidate,
        queue.head
      )
    ) {

      queue.dequeue()

      queue.enqueue(candidate)
    }
  }


  // ============================================================
  // Merge two local Top-K arrays
  //
  // Each Spark partition produces at most K neighbors for each
  // evaluation flight.
  //
  // When partitions are combined, we keep only the global K best.
  // ============================================================

  def mergeTopK(
    left: Array[Neighbor],
    right: Array[Neighbor],
    k: Int
  ): Array[Neighbor] = {

    val queue =
      PriorityQueue.empty[Neighbor]

    var i = 0

    while (i < left.length) {

      addNeighbor(
        queue,
        left(i),
        k
      )

      i += 1
    }


    i = 0

    while (i < right.length) {

      addNeighbor(
        queue,
        right(i),
        k
      )

      i += 1
    }


    queue.dequeueAll.toArray
  }


  // ============================================================
  // Majority vote
  // ============================================================

  def vote(
    neighbors: Array[Neighbor],
    k: Int
  ): Int = {

    var delayedVotes = 0
    var i = 0

    while (i < neighbors.length) {

      if (neighbors(i).label == 1) {
        delayedVotes += 1
      }

      i += 1
    }


    if (delayedVotes > k / 2) {
      1
    }
    else {
      0
    }
  }


  // ============================================================
  // Deterministic sampling
  //
  // This must match SequentialKNN exactly.
  //
  // RecordId is NOT part of the KNN features.
  // It is only the deterministic secondary ordering key.
  // ============================================================

  def deterministicSample(
    df: DataFrame,
    sampleSize: Int
  ): DataFrame = {

    val columnsForHash =
      (featureColumns :+ targetColumn)
        .map(col)


    df
      .withColumn(
        "sampleKey",
        xxhash64(
          columnsForHash: _*
        )
      )
      .orderBy(
        col("sampleKey"),
        col(recordIdColumn)
      )
      .limit(sampleSize)
      .drop("sampleKey")
  }


  // ============================================================
  // Convert Spark Row to Flight
  // ============================================================

  def rowToFlight(
    row: Row
  ): Flight = {

    val recordId =
      row.getLong(0)


    val features =
      new Array[Double](
        featureColumns.length
      )


    var i = 0

    while (i < featureColumns.length) {

      features(i) =
        row.getDouble(i + 1)

      i += 1
    }


    val labelIndex =
      featureColumns.length + 1


    val label =
      row
        .getLong(labelIndex)
        .toInt


    Flight(
      recordId,
      features,
      label
    )
  }


  // ============================================================
  // Select columns needed by KNN
  // ============================================================

  def selectModelColumns(
    df: DataFrame
  ): DataFrame = {

    val selectedColumns =
      Array(recordIdColumn) ++
      featureColumns ++
      Array(targetColumn)


    df.select(
      selectedColumns.map(col): _*
    )
  }


  // ============================================================
  // Collect evaluation data
  //
  // Evaluation sample is intentionally small enough to broadcast.
  // ============================================================

  def collectFlights(
    df: DataFrame
  ): Array[Flight] = {

    selectModelColumns(df)
      .collect()
      .map(rowToFlight)
  }


  // ============================================================
  // Evaluation metrics
  // ============================================================

  def evaluate(
    evaluationData: Array[Flight],
    predictions: Array[Int]
  ): Unit = {

    var truePositive = 0L
    var trueNegative = 0L
    var falsePositive = 0L
    var falseNegative = 0L

    var i = 0

    while (i < evaluationData.length) {

      val actual =
        evaluationData(i).label

      val predicted =
        predictions(i)


      if (
        actual == 1 &&
        predicted == 1
      ) {

        truePositive += 1

      }
      else if (
        actual == 0 &&
        predicted == 0
      ) {

        trueNegative += 1

      }
      else if (
        actual == 0 &&
        predicted == 1
      ) {

        falsePositive += 1

      }
      else if (
        actual == 1 &&
        predicted == 0
      ) {

        falseNegative += 1
      }


      i += 1
    }


    val total =
      truePositive +
      trueNegative +
      falsePositive +
      falseNegative


    val accuracy =
      if (total > 0) {

        (
          truePositive +
          trueNegative
        ).toDouble / total

      }
      else {
        0.0
      }


    val precision =
      if (
        truePositive +
        falsePositive > 0
      ) {

        truePositive.toDouble /
          (
            truePositive +
            falsePositive
          )

      }
      else {
        0.0
      }


    val recall =
      if (
        truePositive +
        falseNegative > 0
      ) {

        truePositive.toDouble /
          (
            truePositive +
            falseNegative
          )

      }
      else {
        0.0
      }


    val f1 =
      if (
        precision +
        recall > 0
      ) {

        2.0 *
          precision *
          recall /
          (
            precision +
            recall
          )

      }
      else {
        0.0
      }


    println()
    println("============================")
    println("CONFUSION MATRIX")
    println("============================")

    println(
      s"True Negative:  $trueNegative"
    )

    println(
      s"False Positive: $falsePositive"
    )

    println(
      s"False Negative: $falseNegative"
    )

    println(
      s"True Positive:  $truePositive"
    )


    println()
    println("============================")
    println("METRICS")
    println("============================")

    println(
      f"Accuracy:  $accuracy%.4f"
    )

    println(
      f"Precision: $precision%.4f"
    )

    println(
      f"Recall:    $recall%.4f"
    )

    println(
      f"F1 Score:  $f1%.4f"
    )


    // ============================================================
    // Majority-class baseline
    // ============================================================

    var notDelayed = 0L

    i = 0

    while (i < evaluationData.length) {

      if (
        evaluationData(i).label == 0
      ) {

        notDelayed += 1
      }

      i += 1
    }


    val baselineAccuracy =
      notDelayed.toDouble /
      evaluationData.length


    println()
    println("============================")
    println("BASELINE")
    println("============================")

    println(
      "Baseline model: always predict NOT delayed"
    )

    println(
      f"Baseline accuracy: $baselineAccuracy%.4f"
    )
  }


  // ============================================================
  // Main
  // ============================================================

  def main(
    args: Array[String]
  ): Unit = {

    // Required:
    //
    // args(0) = training Parquet
    // args(1) = evaluation Parquet
    //
    // Optional:
    //
    // args(2) = training sample size
    // args(3) = evaluation sample size
    // args(4) = K
    // args(5) = Spark partitions


    if (
      args.length < 2 ||
      args.length > 6
    ) {

      System.err.println(
        """
          |Usage:
          |
          |knn.SparkKNN \
          |<train parquet> \
          |<evaluation parquet> \
          |[train size] \
          |[evaluation size] \
          |[k] \
          |[partitions]
          |
          |Example:
          |
          |knn.SparkKNN \
          |data/processed/train.parquet \
          |data/processed/validation.parquet \
          |1000 \
          |100 \
          |5 \
          |4
          |""".stripMargin
      )

      System.exit(1)
    }


    val trainPath =
      args(0)

    val evaluationPath =
      args(1)


    val trainSize =
      if (args.length >= 3) {
        args(2).toInt
      }
      else {
        10000
      }


    val evaluationSize =
      if (args.length >= 4) {
        args(3).toInt
      }
      else {
        1000
      }


    val k =
      if (args.length >= 5) {
        args(4).toInt
      }
      else {
        5
      }


    if (trainSize <= 0) {
      throw new IllegalArgumentException(
        "Training size must be greater than 0."
      )
    }


    if (evaluationSize <= 0) {
      throw new IllegalArgumentException(
        "Evaluation size must be greater than 0."
      )
    }


    if (k <= 0) {
      throw new IllegalArgumentException(
        "K must be greater than 0."
      )
    }


    if (k > trainSize) {
      throw new IllegalArgumentException(
        "K cannot be larger than the training sample."
      )
    }


    // ============================================================
    // Spark
    // ============================================================

    val spark =
      SparkSession
        .builder()
        .appName("Distributed Spark KNN")
        .getOrCreate()


    spark
      .sparkContext
      .setLogLevel("WARN")


    val sc =
      spark.sparkContext


    val numPartitions =
      if (args.length >= 6) {

        args(5).toInt

      }
      else {

        sc.defaultParallelism
      }


    if (numPartitions <= 0) {

      throw new IllegalArgumentException(
        "Number of partitions must be greater than 0."
      )
    }


    println()
    println("============================")
    println("DISTRIBUTED SPARK KNN")
    println("============================")

    println(
      s"Training path: $trainPath"
    )

    println(
      s"Evaluation path: $evaluationPath"
    )

    println(
      s"Training sample size: $trainSize"
    )

    println(
      s"Evaluation sample size: $evaluationSize"
    )

    println(
      s"K: $k"
    )

    println(
      s"Spark partitions: $numPartitions"
    )


    // ============================================================
    // Read Parquet
    // ============================================================

    println()
    println("Reading Parquet data...")


    val fullTrainDF =
      spark
        .read
        .parquet(trainPath)


    val fullEvaluationDF =
      spark
        .read
        .parquet(evaluationPath)


    // ============================================================
    // Deterministic samples
    // ============================================================

    println(
      "Selecting deterministic training sample..."
    )


    val trainSampleDF =
      deterministicSample(
        fullTrainDF,
        trainSize
      )


    println(
      "Selecting deterministic evaluation sample..."
    )


    val evaluationSampleDF =
      deterministicSample(
        fullEvaluationDF,
        evaluationSize
      )


    // ============================================================
    // Evaluation sample goes to driver
    // ============================================================

    println(
      "Collecting evaluation sample to driver..."
    )


    val evaluationData =
      collectFlights(
        evaluationSampleDF
      )


    // ============================================================
    // Training sample stays distributed
    // ============================================================

    println(
      "Preparing distributed training RDD..."
    )


    val trainingRDD =
      selectModelColumns(
        trainSampleDF
      )
        .rdd
        .map(rowToFlight)
        .repartition(numPartitions)
        .cache()


    // count() materializes the cached RDD before timing begins.
    val actualTrainingSize =
      trainingRDD.count()


    val actualEvaluationSize =
      evaluationData.length


    if (k > actualTrainingSize) {

      throw new IllegalArgumentException(
        "K cannot be larger than the actual training sample."
      )
    }


    println()
    println(
      s"Actual training sample: $actualTrainingSize"
    )

    println(
      s"Actual evaluation sample: $actualEvaluationSize"
    )


    val comparisons =
      actualTrainingSize *
      actualEvaluationSize.toLong


    println()
    println(
      f"Distance comparisons: $comparisons%,d"
    )


    // ============================================================
    // Broadcast evaluation flights
    // ============================================================

    println()
    println(
      "Broadcasting evaluation flights..."
    )
    
    val broadcastEvaluation =
      sc.broadcast(
        evaluationData
      )


    val startTime =
      System.nanoTime()


    

    // ============================================================
    // LOCAL TOP K PER PARTITION
    //
    // Each worker receives:
    //
    //   some training flights
    //   +
    //   all evaluation flights
    //
    // For every evaluation flight, each partition keeps only
    // its local K nearest training flights.
    // ============================================================

    val localTopK =
      trainingRDD
        .mapPartitions { partition =>

          val queries =
            broadcastEvaluation.value


          val localQueues =
            Array.fill(
              queries.length
            )(
              PriorityQueue.empty[Neighbor]
            )


          while (partition.hasNext) {

            val trainingFlight =
              partition.next()


            var queryIndex = 0

            while (
              queryIndex < queries.length
            ) {

              val query =
                queries(queryIndex)


              val distance =
                squaredEuclideanDistance(
                  trainingFlight.features,
                  query.features
                )


              val candidate =
                Neighbor(
                  distance,
                  trainingFlight.recordId,
                  trainingFlight.label
                )


              addNeighbor(
                localQueues(queryIndex),
                candidate,
                k
              )


              queryIndex += 1
            }
          }


          localQueues
            .iterator
            .zipWithIndex
            .map {
              case (queue, queryIndex) =>

                (
                  queryIndex,
                  queue.dequeueAll.toArray
                )
            }
        }


    // ============================================================
    // GLOBAL TOP K
    //
    // For each evaluation flight:
    //
    // partition 1 top K
    // partition 2 top K
    // partition 3 top K
    // ...
    //
    // are merged into one global top K.
    // ============================================================

    val globalTopK =
      localTopK
        .reduceByKey(
          (
            left: Array[Neighbor],
            right: Array[Neighbor]
          ) =>
            mergeTopK(
              left,
              right,
              k
            ),
          numPartitions
        )
        .collect()


    // ============================================================
    // Predictions
    // ============================================================

    val predictions =
      new Array[Int](
        evaluationData.length
      )


    var i = 0

    while (i < globalTopK.length) {

      val queryIndex =
        globalTopK(i)._1

      val neighbors =
        globalTopK(i)._2


      predictions(queryIndex) =
        vote(
          neighbors,
          k
        )


      i += 1
    }


    val endTime =
      System.nanoTime()


    val elapsedSeconds =
      (
        endTime -
        startTime
      ) / 1000000000.0


    // ============================================================
    // Runtime
    // ============================================================

    println()
    println("============================")
    println("RUNTIME")
    println("============================")

    println(
      f"Spark KNN runtime: $elapsedSeconds%.4f seconds"
    )

    def printPredictions(
        evaluationData: Array[Flight],
        predictions: Array[Int]
    ): Unit = {

        println()
        println("============================")
        println("INDIVIDUAL PREDICTIONS")
        println("============================")

        var i = 0

        while (i < evaluationData.length) {

            val flight =
            evaluationData(i)

            println(
            s"PREDICTION RecordId=${flight.recordId} Actual=${flight.label} Predicted=${predictions(i)}"
            )

            i += 1
        }
    }


    // ============================================================
    // Evaluation
    // ============================================================

    evaluate(
      evaluationData,
      predictions
    )
    printPredictions(
        evaluationData,
        predictions
    )

    println()
    println(
      "Distributed Spark KNN completed successfully."
    )


    broadcastEvaluation.destroy()

    trainingRDD.unpersist()

    spark.stop()
  }
}