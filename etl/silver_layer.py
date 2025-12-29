import json
import os
from typing import List, Dict, Callable, Optional
import pandas as pd
import csv
import ijson

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, date_format, to_date, concat_ws, lit, trim, col, lower, regexp_replace, when
from pyspark.sql.types import (
    DateType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from etl.utils.utils import logger

class SilverLayer():
    def __init__(self, jobs: List, spark: Optional[SparkSession] = None, dates_list: Optional[List] = None):
        self.function_map: Dict[str, Callable] = {
            "process_cards_data": self.process_cards_data, 
            "process_users_data": self.process_users_data,
            "process_mcc_data": self.process_mcc_data,
            "process_labels_data": self.process_labels_data,
            "process_transactions_data": self.process_transactions_data
            }
        self.jobs = jobs
        self.dates_list = dates_list
        self.spark = spark
    
    # main function that runs all stages
    def run_silver_stage(self):
        if not self.jobs:
            raise
        else:
            logger.info(f"Silver stages in progress")
        for job in self.jobs:
            process_name = job.get('process_name')
            partition_name = job.get('partition_name')
            input_path = job.get('input_path')
            output_path = job.get('output_path')
            try: 
                # logger.info(f"{process_name} in progress") 
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                if process_name == "process_transactions_data":
                    for snap_shot in self.dates_list:
                        self.function_map[process_name](snap_shot, input_path, output_path, partition_name, self.spark)
                else:
                    self.function_map[process_name](input_path, output_path, partition_name, self.spark)
            except Exception as e:
                logger.error(f"Error during {process_name}: {e}")
                raise 
        logger.info(f"Silver stages done") 

    # process the cards dataset
    def process_cards_data(self, bronze_cards_directory, silver_cards_directory, partition_str, spark):

        if not os.path.exists(silver_cards_directory):
            os.makedirs(silver_cards_directory)   

        # connect to bronze attributes and clean them
        bronze_partition_str = "bronze_" + partition_str 
        partition_name = bronze_partition_str + '.parquet'  

        filepath = bronze_cards_directory + partition_name
        df = spark.read.parquet(filepath)

        # clean the string columns which should be dates, 2 columns 'expires' and 'acct_open_date' convert to date
        df = df.withColumn("expires",to_date(concat_ws("/", lit("01"), trim(col("expires"))),"dd/MM/yyyy"))
        df = df.withColumn("acct_open_date",to_date(concat_ws("/", lit("01"), trim(col("acct_open_date"))),"dd/MM/yyyy"))

        # trim and lowercase the categorical columns
        categorical_cols = ["card_brand", "card_type", "has_chip", "card_on_dark_web"]
        for col_name in categorical_cols:
            df = df.withColumn(col_name,trim(lower(col(col_name))).cast(StringType()))
        
        # remove the $ sign for the credit_limt
        df = df.withColumn("credit_limit", regexp_replace(col("credit_limit"), "\\$", ""))

        # type cast all the columns
        column_type_map = {
            "id": StringType(),
            "client_id": StringType(),
            "card_number": StringType(),
            "card_brand": StringType(),
            "card_type": StringType(),
            "has_chip": StringType(),
            "card_on_dark_web": StringType(),
            "cvv": StringType(),
            "credit_limit": IntegerType(),
            "num_cards_issued": IntegerType(),
            "year_pin_last_changed": IntegerType(),
        }
        
        for column, new_type in column_type_map.items():
            df = df.withColumn(column, col(column).cast(new_type))    

        # save silver table - IRL connect to database to write
        silver_partition_name = "silver_" + partition_str + '.parquet'
        filepath = silver_cards_directory + silver_partition_name
        df.write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        logger.info(f"saved to: {filepath}")

    # process the users dataset
    def process_users_data(self, bronze_users_directory, silver_users_directory, partition_str, spark):

        if not os.path.exists(silver_users_directory):
            os.makedirs(silver_users_directory)   

        # connect to bronze attributes and clean them
        bronze_partition_str = "bronze_" + partition_str 
        partition_name = bronze_partition_str + '.parquet'  

        filepath = bronze_users_directory + partition_name
        df = spark.read.parquet(filepath)

        # trim and lowercase the categorical columns
        df = df.withColumn("gender",(lower(col("gender"))).cast(StringType()))
        df = df.withColumn("address",trim(lower(col("address"))).cast(StringType()))

        # remove the $ sign for columns with money
        df = df.withColumn("per_capita_income", regexp_replace(col("per_capita_income"), "\\$", ""))
        df = df.withColumn("yearly_income", regexp_replace(col("yearly_income"), "\\$", ""))
        df = df.withColumn("total_debt", regexp_replace(col("total_debt"), "\\$", ""))

        # type cast all the columns
        column_type_map = {
            "id": StringType(),
            "gender": StringType(),
            "address": StringType(),
            "current_age": IntegerType(),
            "retirement_age": IntegerType(),
            "birth_year": IntegerType(),
            "birth_month": IntegerType(),
            "credit_score": IntegerType(),
            "num_credit_cards": IntegerType(),
            "per_capita_income": IntegerType(),
            "yearly_income": IntegerType(),
            "total_debt": IntegerType()
        }
        
        for column, new_type in column_type_map.items():
            df = df.withColumn(column, col(column).cast(new_type))    

        # save silver table - IRL connect to database to write
        silver_partition_name = "silver_" + partition_str + '.parquet'
        filepath = silver_users_directory + silver_partition_name
        df.write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        logger.info(f"saved to: {filepath}")
        
        # free memory
        df.unpersist()

    # process the mcc dataset
    def process_mcc_data(self, bronze_mcc_directory, silver_mcc_directory, partition_str, spark):

        if not os.path.exists(silver_mcc_directory):
            os.makedirs(silver_mcc_directory)   

        # connect to bronze attributes and clean them
        bronze_partition_str = "bronze_" + partition_str 
        partition_name = bronze_partition_str + '.parquet'  

        filepath = bronze_mcc_directory + partition_name
        df = spark.read.parquet(filepath)

        # trim and lowercase the categorical columns
        df = df.withColumn("mcc_code",(lower(col("mcc_code"))).cast(StringType()))
        df = df.withColumn("mcc_description",trim(lower(col("mcc_description"))).cast(StringType()))
 
        # save silver table - IRL connect to database to write
        silver_partition_name = "silver_" + partition_str + '.parquet'
        filepath = silver_mcc_directory + silver_partition_name
        df.write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        logger.info(f"saved to: {filepath}")

        # free memory
        df.unpersist()

    # process the labels dataset
    def process_labels_data(self, bronze_labels_directory, silver_labels_directory, partition_str, spark):

        if not os.path.exists(silver_labels_directory):
            os.makedirs(silver_labels_directory)   

        # connect to bronze attributes and clean them
        bronze_partition_str = "bronze_" + partition_str 
        partition_name = bronze_partition_str + '.parquet'  

        filepath = bronze_labels_directory + partition_name
        df = spark.read.parquet(filepath)

        # trim and lowercase the categorical columns
        df = df.withColumn("transaction_id",(lower(col("transaction_id"))).cast(StringType()))
        df = df.withColumn("is_fraud",trim(lower(col("is_fraud"))).cast(StringType()))
 
        # save silver table - IRL connect to database to write
        silver_partition_name = "silver_" + partition_str + '.parquet'
        filepath = silver_labels_directory + silver_partition_name
        df.write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        logger.info(f"saved to: {filepath}")

        # free memory
        df.unpersist()
        
    def process_transactions_data(self, snapshot_date_str, bronze_transactions_directory, silver_transactions_directory, partition_str, spark):
            
            if not os.path.exists(silver_transactions_directory):
                os.makedirs(silver_transactions_directory)    

            # connect to bronze attributes and clean them
            bronze_partition_str = "bronze_" + partition_str + '_' + snapshot_date_str.replace('-','_')
            partition_name = bronze_partition_str + '.parquet'  

            filepath = bronze_transactions_directory + partition_name
            df = spark.read.parquet(filepath)

            # clean merchant state and id columns
            df = df.withColumn("merchant_city",(lower(col("merchant_city"))).cast(StringType()))
            df = df.withColumn("merchant_state",when(col("merchant_state").isNull(), "online").otherwise(lower(trim(col("merchant_state")))).cast(StringType()))

            # clean the zip column there are empty rows because the purchase was online, fill with 0
            df = df.withColumn("zip", when(col("zip").isNull(), 0).otherwise(col("zip")))

            # process errors column
            df = df.withColumn("errors", when(col("errors").isNull(), "no_error").otherwise( trim(lower(col("errors")))))

            # remove the $ sign for columns with money
            df = df.withColumn("amount", regexp_replace(col("amount"), "\\$", "").cast(FloatType()))
    
            column_type_map = {
                "id": StringType(),
                "client_id": StringType(),
                "card_id": StringType(),
                "merchant_id": StringType(),
                "mcc": StringType(),
                "zip": StringType(),
                "errors": StringType(),
                "date": TimestampType(),
            }
            

            for column, new_type in column_type_map.items():
                df = df.withColumn(column, col(column).cast(new_type))    
            # save silver table - IRL connect to database to write
            silver_partition_name = "silver_" + partition_str + "_" + snapshot_date_str.replace('-','_') + '.parquet'
            filepath = silver_transactions_directory + silver_partition_name
            df .write.mode("overwrite").parquet(filepath)
            # df.toPandas().to_parquet(filepath,
            #           compression='gzip')
            logger.info(f"saved to: {filepath}")
            # free memory
            df.unpersist()
