import json
import os
from typing import List, Dict, Callable, Optional
import pandas as pd
import csv
import ijson
import redis

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
    TimestampType,
)
from etl.utils.utils import logger
from etl.utils.config import config
from etl.gold_layer import GoldLayer

class OnlineFeatureLayer():
    def __init__(self, jobs: List, spark: Optional[SparkSession] = None, dates_list: Optional[List] = None):
        self.function_map: Dict[str, Callable] = {
            "process_online_feature_store": self.process_online_feature_store, 
            }
        self.jobs = jobs
        self.dates_list = dates_list
        self.spark = spark
    
    # main function that runs all stages
    def run_online_feature_stage(self):
        if not self.jobs:
            raise
        else:
            logger.info(f"Online feature loading from Silver data mart in progress")
        for job in self.jobs:
            process_name = job.get('process_name')
            
            try: 
                if process_name == "process_online_feature_store":
                    input_path = job.get('input_path')       
                    redis_host = job.get('output_redis.host', 'redis')
                    redis_port = job.get('output_redis.port', 6379)    
                    redis_db = job.get('output_redis.db', 0)
                    redis_password = job.get('output_redis.password', None)
                    redis_conn = redis.Redis(host=redis_host, port=redis_port, db=redis_db, password=redis_password)
                    self.function_map[process_name](input_path, self.spark, redis_conn)
                    
            except Exception as e:
                logger.error(f"Error during {process_name}: {e}")
                raise 
        logger.info(f"Online feature store stage done") 

    def load_to_redis(self, redis_conn, df, key_column='card_number'):
        """Load Pandas DataFrame to Redis for online feature store"""
                
        # Replace None with empty string for Redis compatibility
        df = df.fillna('')  
        for index, row in df.iterrows():
            key_id = row[key_column]
            # Convert the row (Pandas Series) to a dictionary, then to a JSON string
            row_dict = row.to_dict()
            redis_conn.set(str(key_id), json.dumps(row_dict))        

    def process_online_feature_store(self, input_path, spark, redis_conn):
        
        # connect to silver cards table
        df_cards = spark.read.parquet(input_path["card_feature_store"])

        # connect to silver users table
        df_users = spark.read.parquet(input_path["user_feature_store"])

        # Join the two DataFrames on df_cards.client_id = df_users.id
        df_users = df_users.withColumnRenamed("id", "client_id")
        df_card_user = df_cards.join(df_users, "client_id", "inner")

        # Convert date columns to string format if they are in datetime format
        # Get all columns in the dataframe that are of type DateType or TimestampType
        # Redis does not support date or timestamp types. Need to convert them to string.
        date_columns = [field.name for field in df_card_user.schema.fields if isinstance(field.dataType, (DateType, TimestampType))]
        # Convert date columns to string format
        for date_col in date_columns:
            df_card_user = df_card_user.withColumn(date_col, date_format(col(date_col), 'yyyy-MM-dd'))
        
        # Add feature from gold layer. Need transaction date....
        # df_card_user = GoldLayer.add_gold_layer_features(df_card_user)
        pdf_card_user = df_card_user.toPandas()

        # Store the DataFrame in the Redis online feature store with card_number as the key
        key_column = 'card_number'
        self.load_to_redis(redis_conn, pdf_card_user, key_column)
        

