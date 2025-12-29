import pyspark

from etl.bronze_layer import BronzeLayer
from etl.silver_layer import SilverLayer
from etl.gold_layer import GoldLayer
from etl.online_feature_layer import OnlineFeatureLayer

from etl.utils.utils import generate_first_of_month_dates, logger
from etl.utils.config import config

#Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
start_date_str = config.get('start_date')
end_date_str = config.get('end_date')

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

# create gold datalake    
try:
    # get gold jobs
    gold_jobs = config.get('gold')
    # instantiate gold object class
    gold = GoldLayer(gold_jobs, dates_list=dates_str_lst, spark=spark)
    #run gold jobs
    gold.run_gold_stage()
except Exception as e:
    logger.exception(f"Unexpected error during Gold stage: {e}")
    raise
