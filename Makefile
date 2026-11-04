# ============================================================
# KNN Flight Delay Prediction
# CS6240 Final Project
# ============================================================


# ============================================================
# Local environment
# ============================================================

hadoop.root=/usr/local/hadoop


# ============================================================
# Application
# ============================================================

app.name=KNN Flight Delay

jar.name=spark-demo.jar
maven.jar.name=spark-demo-1.0.jar

data.check.job.name=knn.DataCheck
sequential.job.name=knn.SequentialKNN
spark.job.name=knn.SparkKNN


# ============================================================
# KNN experiment parameters
# ============================================================

# Default sample sizes
train.size=10000
eval.size=1000

# K selected using validation data
knn.k=5

# Used only for K-selection experiments
k.values=1 3 5 7 11 15

# Number of partitions for distributed Spark KNN
spark.partitions=4


# ============================================================
# Local data
# ============================================================

local.master=local[4]

local.train=data/processed/train.parquet
local.validation=data/processed/validation.parquet
local.test=data/processed/test.parquet

# Validation is the default evaluation dataset.
eval.data=${local.validation}

local.output=output


# ============================================================
# HDFS configuration
# ============================================================

hdfs.user.name=$(shell whoami)

hdfs.data.dir=/user/${hdfs.user.name}/knn-data

hdfs.train=${hdfs.data.dir}/train.parquet
hdfs.validation=${hdfs.data.dir}/validation.parquet
hdfs.test=${hdfs.data.dir}/test.parquet

# Validation is the default HDFS evaluation dataset.
hdfs.eval=${hdfs.validation}

hdfs.output=/user/${hdfs.user.name}/knn-output


# ============================================================
# AWS configuration
#
# These values come from the earlier course setup.
# Review them before actually running EMR.
# ============================================================

aws.emr.release=emr-6.11.1

aws.bucket.name=cs6240-final-project-prabesh-nepal-2026

aws.raw.2024=raw/2024/csv/*.csv
aws.processed.2024=processed/2024
aws.data.dir=data

aws.train=processed/2024/train
aws.validation=processed/2024/validation
aws.test=processed/2024/test

aws.eval=${aws.validation}

aws.output=knn-output
aws.log.dir=knn-logs

aws.core.num.nodes=2
aws.primary.num.nodes=1
aws.instance.type=m4.large
aws.subnet.id=subnet-0ba2edeb24dbf2685
# ============================================================
# AWS EMR - Full 2024 preprocessing
# ============================================================

aws-preprocess: jar upload-app-aws
	aws emr create-cluster \
		--ec2-attributes SubnetId=${aws.subnet.id} \
		--name "BTS Full 2024 Preprocessing" \
		--release-label ${aws.emr.release} \
		--instance-groups '[{"InstanceCount":${aws.core.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":${aws.primary.num.nodes},"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Hadoop Name=Spark \
		--steps Type=CUSTOM_JAR,Name="Full 2024 Preprocessing",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster","--class","knn.FullPreprocess","s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.raw.2024}","s3://${aws.bucket.name}/${aws.processed.2024}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--configurations '[{"Classification":"hadoop-env","Configurations":[{"Classification":"export","Configurations":[],"Properties":{"JAVA_HOME":"/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties":{}},{"Classification":"spark-env","Configurations":[{"Classification":"export","Configurations":[],"Properties":{"JAVA_HOME":"/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties":{}}]' \
		--use-default-roles \
		--enable-debugging \
		--auto-terminate

# ============================================================
# Build
# ============================================================

jar:
	mvn clean package
	cp target/${maven.jar.name} ${jar.name}


# ============================================================
# Data check
# ============================================================

data-check: jar
	spark-submit \
		--class ${data.check.job.name} \
		--master ${local.master} \
		--name "KNN Data Check" \
		${jar.name} \
		${local.train} \
		${local.validation}


# ============================================================
# Sequential KNN
# ============================================================

# Run an already-built JAR.
run-sequential:
	spark-submit \
		--class ${sequential.job.name} \
		--master ${local.master} \
		--name "Sequential KNN" \
		${jar.name} \
		${local.train} \
		${eval.data} \
		${train.size} \
		${eval.size} \
		${knn.k}


# Build first, then run sequential KNN.
local: jar
	$(MAKE) --no-print-directory run-sequential \
		eval.data=${eval.data} \
		train.size=${train.size} \
		eval.size=${eval.size} \
		knn.k=${knn.k}


# Optional alias if you still want to use "run-local".
run-local: run-sequential


# ============================================================
# Distributed Spark KNN
# ============================================================

# Run already-built Spark KNN JAR.
run-spark:
	spark-submit \
		--class ${spark.job.name} \
		--master ${local.master} \
		--name "Distributed Spark KNN" \
		${jar.name} \
		${local.train} \
		${eval.data} \
		${train.size} \
		${eval.size} \
		${knn.k} \
		${spark.partitions}


# Build first, then run distributed Spark KNN.
spark-local: jar
	$(MAKE) --no-print-directory run-spark \
		eval.data=${eval.data} \
		train.size=${train.size} \
		eval.size=${eval.size} \
		knn.k=${knn.k} \
		spark.partitions=${spark.partitions}


# ============================================================
# K-value validation experiments
# ============================================================

# Build once and run all candidate K values.
#
# Uses:
#   train.parquet
#   validation.parquet
#
# Does NOT use final test data.
k-experiments: jar
	@for k in ${k.values}; do \
		echo ""; \
		echo "========================================"; \
		echo "Sequential KNN Validation Experiment"; \
		echo "Train Size      = ${train.size}"; \
		echo "Validation Size = ${eval.size}"; \
		echo "K               = $$k"; \
		echo "========================================"; \
		$(MAKE) --no-print-directory run-sequential \
			eval.data=${local.validation} \
			train.size=${train.size} \
			eval.size=${eval.size} \
			knn.k=$$k; \
	done


# Save the full K experiment log and show only useful results.
k-validation-log:
	@mkdir -p ${local.output}
	@$(MAKE) --no-print-directory k-experiments \
		> ${local.output}/k-validation.log 2>&1
	@echo "Saved full log to ${local.output}/k-validation.log"
	@echo ""
	@grep -E \
		"Sequential KNN Validation Experiment|Train Size|Validation Size|K[[:space:]]*=|KNN runtime|Accuracy:|Precision:|Recall:|F1 Score:|Baseline accuracy" \
		${local.output}/k-validation.log


# ============================================================
# Sequential vs Spark correctness comparison
# ============================================================

# Small correctness test.
#
# Defaults:
#   train = 1000
#   eval  = 100
#   K     = 5
#
# You can override these from the command line.
compare.train.size=1000
compare.eval.size=100
compare.k=5
compare.partitions=4


compare-small: jar
	@mkdir -p ${local.output}

	@echo "Running Sequential KNN..."
	@$(MAKE) --no-print-directory run-sequential \
		eval.data=${local.validation} \
		train.size=${compare.train.size} \
		eval.size=${compare.eval.size} \
		knn.k=${compare.k} \
		> ${local.output}/sequential-small.log 2>&1

	@echo "Running Distributed Spark KNN..."
	@$(MAKE) --no-print-directory run-spark \
		eval.data=${local.validation} \
		train.size=${compare.train.size} \
		eval.size=${compare.eval.size} \
		knn.k=${compare.k} \
		spark.partitions=${compare.partitions} \
		> ${local.output}/spark-small.log 2>&1


	@grep -E \
		"Actual training sample|Actual evaluation sample|Distance comparisons|True Negative|False Positive|False Negative|True Positive|Accuracy:|Precision:|Recall:|F1 Score:|Baseline accuracy" \
		${local.output}/sequential-small.log \
		> ${local.output}/sequential-metrics.txt

	@grep -E \
		"Actual training sample|Actual evaluation sample|Distance comparisons|True Negative|False Positive|False Negative|True Positive|Accuracy:|Precision:|Recall:|F1 Score:|Baseline accuracy" \
		${local.output}/spark-small.log \
		> ${local.output}/spark-metrics.txt


	@grep '^PREDICTION ' \
		${local.output}/sequential-small.log \
		| sort \
		> ${local.output}/sequential-predictions.txt

	@grep '^PREDICTION ' \
		${local.output}/spark-small.log \
		| sort \
		> ${local.output}/spark-predictions.txt


	@echo ""
	@echo "========================================"
	@echo "SEQUENTIAL RESULTS"
	@echo "========================================"
	@cat ${local.output}/sequential-metrics.txt


	@echo ""
	@echo "========================================"
	@echo "SPARK RESULTS"
	@echo "========================================"
	@cat ${local.output}/spark-metrics.txt


	@echo ""
	@echo "========================================"
	@echo "AGGREGATE METRIC CHECK"
	@echo "========================================"

	@if diff -q \
		${local.output}/sequential-metrics.txt \
		${local.output}/spark-metrics.txt \
		> /dev/null; then \
		echo "MATCH: Sequential and Spark metrics are identical."; \
	else \
		echo "DIFFERENCE FOUND: Metrics do not match."; \
		echo ""; \
		diff -u \
			${local.output}/sequential-metrics.txt \
			${local.output}/spark-metrics.txt || true; \
	fi


	@echo ""
	@echo "========================================"
	@echo "INDIVIDUAL PREDICTION CHECK"
	@echo "========================================"

	@seq_count=$$(wc -l < ${local.output}/sequential-predictions.txt | tr -d ' '); \
	spark_count=$$(wc -l < ${local.output}/spark-predictions.txt | tr -d ' '); \
	echo "Sequential predictions: $$seq_count"; \
	echo "Spark predictions:      $$spark_count"; \
	echo "Expected predictions:   ${compare.eval.size}"; \
	echo ""; \
	if [ "$$seq_count" -ne "${compare.eval.size}" ]; then \
		echo "ERROR: Sequential output does not contain the expected number of predictions."; \
		exit 1; \
	fi; \
	if [ "$$spark_count" -ne "${compare.eval.size}" ]; then \
		echo "ERROR: Spark output does not contain the expected number of predictions."; \
		exit 1; \
	fi; \
	if diff -q \
		${local.output}/sequential-predictions.txt \
		${local.output}/spark-predictions.txt \
		> /dev/null; then \
		echo "EXACT MATCH: Every individual prediction is identical."; \
	else \
		echo "DIFFERENCE FOUND: Individual predictions do not match."; \
		echo ""; \
		diff -u \
			${local.output}/sequential-predictions.txt \
			${local.output}/spark-predictions.txt || true; \
		exit 1; \
	fi


	@echo ""
	@echo "========================================"
	@echo "CORRECTNESS VERIFICATION COMPLETE"
	@echo "========================================"
	@echo "SequentialKNN and SparkKNN produced"
	@echo "the same predictions for every evaluation record."

	@echo ""
	@echo "Files:"
	@echo "  ${local.output}/sequential-small.log"
	@echo "  ${local.output}/spark-small.log"
	@echo "  ${local.output}/sequential-predictions.txt"
	@echo "  ${local.output}/spark-predictions.txt"

# ============================================================
# Sequential scalability experiment
# ============================================================

scaling.train.sizes=10000 25000 50000 100000 200000
scaling.eval.size=1000


sequential-scaling: jar
	@mkdir -p ${local.output}/scaling

	@echo "train_size,eval_size,k,comparisons,runtime_seconds,accuracy,f1" \
		> ${local.output}/scaling/sequential.csv

	@for n in ${scaling.train.sizes}; do \
		echo ""; \
		echo "========================================"; \
		echo "Sequential Scaling Experiment"; \
		echo "Train Size = $$n"; \
		echo "Eval Size  = ${scaling.eval.size}"; \
		echo "K          = ${knn.k}"; \
		echo "========================================"; \
		log="${local.output}/scaling/sequential-$$n.log"; \
		$(MAKE) --no-print-directory run-sequential \
			eval.data=${local.validation} \
			train.size=$$n \
			eval.size=${scaling.eval.size} \
			knn.k=${knn.k} \
			> "$$log" 2>&1; \
		comparisons=$$(grep "Distance comparisons:" "$$log" \
			| tail -1 \
			| awk '{print $$3}' \
			| tr -d ','); \
		runtime=$$(grep "^KNN runtime:" "$$log" \
			| tail -1 \
			| awk '{print $$3}'); \
		accuracy=$$(grep "^Accuracy:" "$$log" \
			| tail -1 \
			| awk '{print $$2}'); \
		f1=$$(grep "^F1 Score:" "$$log" \
			| tail -1 \
			| awk '{print $$3}'); \
		echo "$$n,${scaling.eval.size},${knn.k},$$comparisons,$$runtime,$$accuracy,$$f1" \
			>> ${local.output}/scaling/sequential.csv; \
		echo "Runtime = $$runtime seconds"; \
	done

	@echo ""
	@echo "Results:"
	@cat ${local.output}/scaling/sequential.csv


# ============================================================
# Spark scalability experiment
# ============================================================

spark-scaling: jar
	@mkdir -p ${local.output}/scaling

	@echo "train_size,eval_size,k,partitions,comparisons,runtime_seconds,accuracy,f1" \
		> ${local.output}/scaling/spark.csv

	@for n in ${scaling.train.sizes}; do \
		echo ""; \
		echo "========================================"; \
		echo "Spark Scaling Experiment"; \
		echo "Train Size = $$n"; \
		echo "Eval Size  = ${scaling.eval.size}"; \
		echo "K          = ${knn.k}"; \
		echo "Partitions = ${spark.partitions}"; \
		echo "========================================"; \
		log="${local.output}/scaling/spark-$$n.log"; \
		$(MAKE) --no-print-directory run-spark \
			eval.data=${local.validation} \
			train.size=$$n \
			eval.size=${scaling.eval.size} \
			knn.k=${knn.k} \
			spark.partitions=${spark.partitions} \
			> "$$log" 2>&1; \
		comparisons=$$(grep "Distance comparisons:" "$$log" \
			| tail -1 \
			| awk '{print $$3}' \
			| tr -d ','); \
		runtime=$$(grep "^Spark KNN runtime:" "$$log" \
			| tail -1 \
			| awk '{print $$4}'); \
		accuracy=$$(grep "^Accuracy:" "$$log" \
			| tail -1 \
			| awk '{print $$2}'); \
		f1=$$(grep "^F1 Score:" "$$log" \
			| tail -1 \
			| awk '{print $$3}'); \
		echo "$$n,${scaling.eval.size},${knn.k},${spark.partitions},$$comparisons,$$runtime,$$accuracy,$$f1" \
			>> ${local.output}/scaling/spark.csv; \
		echo "Runtime = $$runtime seconds"; \
	done

	@echo ""
	@echo "Results:"
	@cat ${local.output}/scaling/spark.csv

# ============================================================
# Spark partition experiment
# ============================================================

partition.values=1 2 4 8
partition.train.size=200000
partition.eval.size=1000


partition-experiments: jar
	@mkdir -p ${local.output}/partitions

	@echo "partitions,train_size,eval_size,k,runtime_seconds,accuracy,f1" \
		> ${local.output}/partitions/results.csv

	@for p in ${partition.values}; do \
		echo ""; \
		echo "========================================"; \
		echo "Spark Partition Experiment"; \
		echo "Train Size = ${partition.train.size}"; \
		echo "Eval Size  = ${partition.eval.size}"; \
		echo "K          = ${knn.k}"; \
		echo "Partitions = $$p"; \
		echo "========================================"; \
		log="${local.output}/partitions/spark-p$$p.log"; \
		$(MAKE) --no-print-directory run-spark \
			eval.data=${local.validation} \
			train.size=${partition.train.size} \
			eval.size=${partition.eval.size} \
			knn.k=${knn.k} \
			spark.partitions=$$p \
			> "$$log" 2>&1; \
		runtime=$$(grep "^Spark KNN runtime:" "$$log" \
			| tail -1 \
			| awk '{print $$4}'); \
		accuracy=$$(grep "^Accuracy:" "$$log" \
			| tail -1 \
			| awk '{print $$2}'); \
		f1=$$(grep "^F1 Score:" "$$log" \
			| tail -1 \
			| awk '{print $$3}'); \
		echo "$$p,${partition.train.size},${partition.eval.size},${knn.k},$$runtime,$$accuracy,$$f1" \
			>> ${local.output}/partitions/results.csv; \
		echo "Runtime = $$runtime seconds"; \
	done

	@echo ""
	@echo "Results:"
	@cat ${local.output}/partitions/results.csv

# ============================================================
# Final test
#
# Do not use these for tuning.
# ============================================================

final-test-sequential: jar
	$(MAKE) --no-print-directory run-sequential \
		eval.data=${local.test} \
		train.size=${train.size} \
		eval.size=${eval.size} \
		knn.k=${knn.k}


final-test-spark: jar
	$(MAKE) --no-print-directory run-spark \
		eval.data=${local.test} \
		train.size=${train.size} \
		eval.size=${eval.size} \
		knn.k=${knn.k} \
		spark.partitions=${spark.partitions}


# ============================================================
# Local output
# ============================================================

clean-local-output:
	rm -rf ${local.output}/*


# ============================================================
# HDFS
# ============================================================

start-hdfs:
	${hadoop.root}/sbin/start-dfs.sh


stop-hdfs:
	${hadoop.root}/sbin/stop-dfs.sh


# WARNING:
# This resets the local HDFS filesystem.
format-hdfs: stop-hdfs
	rm -rf /tmp/hadoop*
	${hadoop.root}/bin/hdfs namenode -format


# ============================================================
# YARN
# ============================================================

start-yarn: stop-yarn
	${hadoop.root}/sbin/start-yarn.sh


stop-yarn:
	${hadoop.root}/sbin/stop-yarn.sh


# ============================================================
# Initialize HDFS
# ============================================================

init-hdfs: start-hdfs
	${hadoop.root}/bin/hdfs dfs \
		-mkdir -p /user/${hdfs.user.name}

	${hadoop.root}/bin/hdfs dfs \
		-mkdir -p ${hdfs.data.dir}


# ============================================================
# Upload data to HDFS
# ============================================================

upload-input-hdfs: init-hdfs
	${hadoop.root}/bin/hdfs dfs \
		-rm -f ${hdfs.train}

	${hadoop.root}/bin/hdfs dfs \
		-rm -f ${hdfs.validation}

	${hadoop.root}/bin/hdfs dfs \
		-rm -f ${hdfs.test}

	${hadoop.root}/bin/hdfs dfs \
		-put ${local.train} ${hdfs.train}

	${hadoop.root}/bin/hdfs dfs \
		-put ${local.validation} ${hdfs.validation}

	${hadoop.root}/bin/hdfs dfs \
		-put ${local.test} ${hdfs.test}


clean-hdfs-output:
	${hadoop.root}/bin/hdfs dfs \
		-rm -r -f ${hdfs.output}


# ============================================================
# Pseudo-cluster Spark KNN
# ============================================================

# First run:
# initialize HDFS, upload data, start YARN, run SparkKNN.
pseudo: jar upload-input-hdfs start-yarn
	spark-submit \
		--class ${spark.job.name} \
		--master yarn \
		--deploy-mode cluster \
		--name "Distributed Spark KNN" \
		${jar.name} \
		${hdfs.train} \
		${hdfs.eval} \
		${train.size} \
		${eval.size} \
		${knn.k} \
		${spark.partitions}


# Quick run when HDFS data already exists.
pseudoq: jar start-hdfs start-yarn
	spark-submit \
		--class ${spark.job.name} \
		--master yarn \
		--deploy-mode cluster \
		--name "Distributed Spark KNN" \
		${jar.name} \
		${hdfs.train} \
		${hdfs.eval} \
		${train.size} \
		${eval.size} \
		${knn.k} \
		${spark.partitions}


# ============================================================
# AWS S3
# ============================================================

make-bucket:
	aws s3 mb s3://${aws.bucket.name}


upload-input-aws:
	aws s3 cp \
		${local.train} \
		s3://${aws.bucket.name}/${aws.train}

	aws s3 cp \
		${local.validation} \
		s3://${aws.bucket.name}/${aws.validation}

	aws s3 cp \
		${local.test} \
		s3://${aws.bucket.name}/${aws.test}


upload-app-aws:
	aws s3 cp \
		${jar.name} \
		s3://${aws.bucket.name}/${jar.name}


delete-output-aws:
	aws s3 rm \
		s3://${aws.bucket.name}/${aws.output} \
		--recursive


# ============================================================
# AWS EMR
#
# Review release and instance configuration before running.
# ============================================================

aws: jar upload-app-aws
	aws emr create-cluster \
		--ec2-attributes SubnetId=${aws.subnet.id} \
		--name "KNN Flight Delay Spark Cluster" \
		--release-label ${aws.emr.release} \
		--instance-groups '[{"InstanceCount":${aws.core.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":${aws.primary.num.nodes},"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Hadoop Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster","--conf","spark.dynamicAllocation.enabled=false","--conf","spark.executor.cores=2","--conf","spark.executor.memory=2g","--num-executors","${aws.core.num.nodes}","--class","${spark.job.name}","s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.train}","s3://${aws.bucket.name}/${aws.eval}","${train.size}","${eval.size}","${knn.k}","${spark.partitions}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--configurations '[{"Classification":"hadoop-env","Configurations":[{"Classification":"export","Configurations":[],"Properties":{"JAVA_HOME":"/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties":{}},{"Classification":"spark-env","Configurations":[{"Classification":"export","Configurations":[],"Properties":{"JAVA_HOME":"/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties":{}}]' \
		--use-default-roles \
		--enable-debugging \
		--auto-terminate


# ============================================================
# Hadoop configuration switching
# ============================================================

switch-standalone:
	cp config/standalone/*.xml \
		${hadoop.root}/etc/hadoop


switch-pseudo:
	cp config/pseudo/*.xml \
		${hadoop.root}/etc/hadoop


# ============================================================
# Cleanup
# ============================================================

clean:
	mvn clean
	rm -f ${jar.name}


# ============================================================
# PHONY targets
# ============================================================

.PHONY: \
	jar \
	data-check \
	run-sequential \
	run-local \
	local \
	run-spark \
	spark-local \
	k-experiments \
	k-validation-log \
	compare-small \
	final-test-sequential \
	final-test-spark \
	clean-local-output \
	start-hdfs \
	stop-hdfs \
	format-hdfs \
	start-yarn \
	stop-yarn \
	init-hdfs \
	upload-input-hdfs \
	clean-hdfs-output \
	pseudo \
	pseudoq \
	make-bucket \
	upload-input-aws \
	upload-app-aws \
	delete-output-aws \
	aws \
	switch-standalone \
	switch-pseudo \
	clean \
	sequential-scaling \
	spark-scaling \
	partition-experiments \
	aws-preprocess