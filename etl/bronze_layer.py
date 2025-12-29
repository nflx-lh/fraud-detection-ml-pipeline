
"""
Bronze Layer ETL Pipeline Module using PySpark

This module handles the ingestion of raw data files into the Bronze layer
of a Medallion Architecture data pipeline.

It reads raw data files from the data/raw/ folder and saves them as Parquet files
in the datamart/bronze/ folder without transformations.
"""
import os
from typing import List, Dict, Callable, Optional
import pandas as pd
import ijson
import tempfile

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format
from etl.utils.utils import logger

class BronzeLayer():
    def __init__(self, jobs: List, spark: Optional[SparkSession] = None, dates_list: Optional[List] = None):
        self.function_map: Dict[str, Callable] = {
            "process_transactions": self.process_transactions, 
            "process_cards": self.process_cards_users,
            "process_users": self.process_cards_users,
            "process_mcc": self.process_mcc_fraud,
            "process_labels": self.process_mcc_fraud,
            }
        self.jobs = jobs
        self.dates_list = dates_list
        self.spark = spark
        
    def run_bronze_stage(self):
        if not self.jobs:
            raise 
        else:
            logger.info("Bronze stages in progress")
        for job in self.jobs:
            process_name = job.get('process_name')
            partition_name = job.get('partition_name')
            input_path = job.get('input_path')
            output_path = job.get('output_path')
            try: 
                # logger.info(f"{process_name} in progress") 
                #
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                     
                # call the correct function for the correct tables
                if process_name == "process_transactions":
                    full_df = self.spark.read.option("header", True).option("inferSchema", True).csv(input_path)
                    full_df = full_df.withColumn("year_month", date_format(col("date"), "yyyy-MM")).cache()
                    for date_str in self.dates_list:
                        self.function_map[process_name](date_str, full_df, output_path, partition_name)
                    full_df.unpersist(blocking=True)

                elif process_name == "process_cards" or process_name == "process_users": 
                    self.function_map[process_name](input_path, output_path, partition_name, self.spark)
                elif process_name == "process_mcc" or process_name == "process_labels": 
                    self.function_map[process_name](input_path, output_path, partition_name, process_name, self.spark)
            except Exception as e:
                logger.error(f"Error during {process_name}: {e}")
                raise 
        logger.info("Bronze stages done") 
        
    # transactions csv, need to backfill
    def process_transactions(self, snapshot_date_str, full_df, bronze_directory, partition_str):
        # prepare arguments
        bronze_partition_str = "bronze_" + partition_str + "_"
        year_month = snapshot_date_str[:7]

        # load, convert date to just year month and day, filter based on snapshot date
        df_filtered = full_df.filter(col("year_month") == year_month)
        logger.info(f"{snapshot_date_str} row count: {df_filtered.count()}")

        # # save bronze table to datamart - IRL connect to database to write
        # partition_name = bronze_partition_str + snapshot_date_str.replace('-','_') + '.csv'
        # filepath = os.path.join(bronze_directory, partition_name)
        
        # df_filtered.toPandas().to_csv(filepath, index=False)
        # logger.info(f"saved to: {filepath}")

        # save bronze table to datamart - IRL connect to database to write
        partition_name = bronze_partition_str + snapshot_date_str.replace('-', '_') + '.parquet'
        filepath = os.path.join(bronze_directory, partition_name)

        df_filtered.write.mode('overwrite').parquet(filepath)
        logger.info(f"saved to: {filepath}")
                    
    # ingest cards csv
    def process_cards_users(self, input_dir, output_dir, partition_str, spark):

        bronze_partition_str = "bronze_" + partition_str
        df = self.load_csv_file(spark, input_dir)

        # partition_name = bronze_partition_str + '.csv'
        # filepath = output_dir + partition_name

        # df.toPandas().to_csv(filepath, index=False)
        # logger.info(f"Successfully saved {partition_str} → {filepath}")

        partition_name = bronze_partition_str + '.parquet'
        filepath = os.path.join(output_dir, partition_name)

        df.write.mode('overwrite').parquet(filepath)
        logger.info(f"saved to: {filepath}")

        # free memory
        df.unpersist()

    def process_mcc_fraud(self, input_dir, output_dir, partition_str, process_name, spark, batch_size=10000):
        bronze_partition_str = "bronze_" + partition_str
        partition_name = bronze_partition_str + ".parquet"
        output_path = os.path.join(output_dir, partition_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_files = []
            batch_records = []

            with open(input_dir, 'r') as f:
                if process_name == "process_labels":
                    parser = ijson.kvitems(f, 'target')
                    to_row = lambda k, v: {'transaction_id': k, 'is_fraud': v}
                else:
                    parser = ijson.kvitems(f, '')
                    to_row = lambda k, v: {'mcc_code': k, 'mcc_description': v}

                for i, (key, value) in enumerate(parser, 1):
                    batch_records.append(to_row(key, value))

                    if i % batch_size == 0:
                        tmp_file = os.path.join(tmp_dir, f"batch_{i}.parquet")
                        pd.DataFrame(batch_records).to_parquet(tmp_file, index=False)
                        batch_files.append(tmp_file)
                        batch_records.clear()

                # Write any remaining records
                if batch_records:
                    tmp_file = os.path.join(tmp_dir, "batch_final.parquet")
                    pd.DataFrame(batch_records).to_parquet(tmp_file, index=False)
                    batch_files.append(tmp_file)

            # Combine all batch files into a single Parquet using Spark
            df = spark.read.parquet(*batch_files)
            df.write.mode("overwrite").parquet(output_path)

    def load_csv_file(self, spark, file_path):
        try:
            return (spark.read
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .csv(str(file_path)))
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return None
        

