from ml.data_loader import DataLoader
from ml.preprocessor import Preprocessor
from ml.model_manager import ModelManager
from ml.utils.config import ConfigReader
from ml.utils.utils import logger
import argparse
import pandas as pd

def train_model_pipeline(config_path: str):
    config = ConfigReader(config_path)
    print(config.get("model_config"))
    # load data from the specified start date from the gold layer
    try:
        logger.info(f"Loading Data from Gold Layer")
        data_loader = DataLoader(**config.get("dataloader_config"))
        df = data_loader.load_gold_parquet()

        # df = pd.read_parquet("gold_data.parquet")
    except Exception as e:
        logger.exception(f"Unexpected error during data ingestion: {e}")

    # # preprocess the data
    try:
        logger.info(f"Starting preprocessing stage")
        preprocessor = Preprocessor(df, **config.get("preprocessor_config"), **config.get("model_config"))
        processed_data = preprocessor.preprocess()
        X_train, X_test, y_train, y_test = processed_data['split']
        #print(X_train)
        if processed_data['oot']:
            oot_data = processed_data['oot'] # this is a nest list of feature label pairs for oot testing later
        else:
            oot_data = []
    except Exception as e:
        logger.exception(f"Unexpected error during data preprocessing: {e}")


    # train and test
    try:
        logger.info(f"Starting training stage")
        model_name = config.get("model_config.model_name")
        logger.info(f"Model Name: {model_name}")
        # Overwrite the model name in the config
        optuna_config = ConfigReader(f'ml/optuna_config/{model_name}.yaml')
        model_manager = ModelManager(**config.get("model_config"), optuna_config = optuna_config.get("optuna_config"), oot=oot_data,
                                    col_preprocessor=preprocessor.transform_pipeline)
        model_manager.train_evaluate(X_train, X_test, y_train, y_test)
    except Exception as e:
        logger.exception(f"Unexpected error during training: {e}")

"""
need to add the dataset logging logic in mlflow
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run ML Pipeline")
    parser.add_argument("--config_path", type=str, default="ml/ml_conf.yaml")
    args = parser.parse_args()

    train_model_pipeline(config_path=args.config_path)