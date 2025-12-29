import logging
from datetime import datetime
import os
from monitoring.utils.config import config


name = config.get("name")
logger = logging.getLogger("Basic Logger")
logger.setLevel(logging.INFO)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if not logger.hasHandlers():
    os.makedirs("logs", exist_ok=True)
    # set format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # print logs
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # save logs
    filename = f"logs/{timestamp}_{name}.log"
    filehandler = logging.FileHandler(filename)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    logger.info(f"Monitoring Pipeline started by {name}")


# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def read_file(filepath, spark):

    df = spark.read.option("header", "true").parquet(filepath)
    print("row_count:",df.count())
    print(df.printSchema())

    df.show()
