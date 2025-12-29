import json
import os
from typing import List, Dict, Callable, Optional
import pandas as pd
import csv
import ijson

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, date_format, to_date, concat_ws, lit, trim, col, lower, regexp_replace, when, to_date, datediff, year, round
from pyspark.sql.types import (
    DateType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from etl.utils.utils import logger

class GoldLayer():
    def __init__(self, jobs: List, spark: Optional[SparkSession] = None, dates_list: Optional[List] = None):
        self.function_map: Dict[str, Callable] = {
            "process_feature_store": self.process_feature_store, 
            "process_label_store": self.process_label_store,
            }
        self.jobs = jobs
        self.dates_list = dates_list
        self.spark = spark
    
    # main function that runs all stages
    def run_gold_stage(self):
        if not self.jobs:
            raise
        else:
            logger.info(f"Gold stages in progress")
        for job in self.jobs:
            process_name = job.get('process_name')
            partition_name = job.get('partition_name')
            input_path = job.get('input_path')
            output_path = job.get('output_path')
            try: 
                # logger.info(f"{process_name} in progress") 
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                if process_name == "process_feature_store":
                    for snap_shot in self.dates_list:
                        self.function_map[process_name](snap_shot, output_path, partition_name, self.spark)
                else:
                    self.function_map[process_name](input_path, output_path, partition_name, self.spark)
            except Exception as e:
                logger.error(f"Error during {process_name}: {e}")
                raise 
        logger.info(f"Gold stages done") 

    @staticmethod
    def add_gold_layer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for gold layer features.
        
        ########### IMPORTANT ###########
        As these features are time dependent on the transaction date, any features added here need to be
        done so during the real time inference stage.

        """

        df = df.withColumn("date", to_date("date"))         
        df = df.withColumn("acct_opened_months",round(datediff("date", "acct_open_date") / 30.0, 1))
        df = df.withColumn("year_of_tr", year("date"))
        df = df.withColumn("yrs_since_pin_changed", col("year_of_tr") - col("year_pin_last_changed"))
        df = df.drop("year_of_tr") 

        return df

    def process_feature_store(self, snapshot_date_str, gold_feature_directory, partition_str, spark):
        combined_df = self.join(snapshot_date_str, spark)

        # add your feature engineering here if any
        # Add gold features
        combined_df = GoldLayer.add_gold_layer_features(combined_df)

        # write to feature store
        gold_partition_name = "gold_" + partition_str + "_" + snapshot_date_str.replace('-','_') + '.parquet' 
        filepath = gold_feature_directory + gold_partition_name
        combined_df.write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        logger.info(f"saved to: {filepath}")
        # free memory
        combined_df.unpersist()



    # Join mcc, cards, user for each snapshot_date transaction
    def join(self, snapshot_date_str, spark): 

        # connect to silver transaction table
        partition_name = "silver_transactions_" + snapshot_date_str.replace('-','_') + '.parquet' 

        filepath = "datamart/silver/transactions/" + partition_name
        df_transactions = spark.read.parquet(filepath)

        # connect to silver mcc table
        df_mcc = spark.read.parquet("datamart/silver/mcc/silver_mcc.parquet")

        # connect to silver cards table
        df_cards = spark.read.parquet("datamart/silver/cards/silver_cards.parquet")

        # connect to silver users table
        df_users = spark.read.parquet("datamart/silver/users/silver_users.parquet")

        # perform the joins
        # left join transactions with users on transactions.client_id = users.id 
        df_joined = df_transactions.join(df_users, df_transactions.client_id == df_users.id, how="left")

        # drop duplicate rows
        df_joined = df_joined.drop(df_users.id)  

        # left join again with transactions.card_id = cards_data.id
        df_joined = df_joined.join(df_cards, df_joined.card_id == df_cards.id, how="left")

        # drop duplicate rows
        df_joined = df_joined.drop(df_cards.id)
        df_joined = df_joined.drop(df_cards.client_id)
        
        # left join again with transactions.mcc = mcc.mcc_code
        df_joined = df_joined.join(df_mcc, df_joined.mcc == df_mcc.mcc_code, how="left")
        df_joined = df_joined.drop(df_mcc.mcc_code)
        
        logger.info(f"Join Completed")

        return df_joined

    def process_label_store(self, silver_labels_directory, gold_labels_directory, partition_str, spark):
        
        if not os.path.exists(gold_labels_directory):
            os.makedirs(gold_labels_directory)    

        # connect to silver labels and clean them
        silver_partition_str = "silver_" + partition_str + '.parquet' 

        filepath = silver_labels_directory + silver_partition_str

        df = spark.read.parquet(filepath)
        
        # save gold label table - IRL connect to database to write
        gold_partition_name = "gold_" + partition_str + '.parquet'
        filepath = gold_labels_directory + gold_partition_name
        
        df.write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        logger.info(f"saved to: {filepath}")
        # free memory
        df.unpersist()
