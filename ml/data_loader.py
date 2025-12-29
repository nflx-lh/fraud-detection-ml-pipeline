# src/data_loader.py

import os
import pandas as pd
from typing import List
from pathlib import Path
from datetime import datetime
from ml.utils.utils import logger

class DataLoader():
    """
    This class handles the loading of data from the gold layer and concatenating the features with labels, the features are chosen from a start date 
    to the present date, and the relevant labels are joined to the transaction ids
    """

    def __init__(self, start_date, gold_feature_dir, gold_label_dir = None):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.gold_feature_dir = Path(gold_feature_dir)
        self.gold_label_dir = Path(gold_label_dir)

    def load_gold_parquet(self) -> pd.DataFrame:
        """
        Load and concatenate feature snapshots with labels.
        """
        dfs = []
        for file in self.gold_feature_dir.iterdir():
            file_name = file.stem
            date_str = datetime.strptime(file_name.replace("gold_features_", ""), '%Y_%m_%d')
            # get all data from the start date to the end
            if date_str >= self.start_date:
                dfs.append(pd.read_parquet(file))
            

        df_all_features = pd.concat(dfs, axis=0, ignore_index=True)

        # Align the column name first
        df_all_features = df_all_features.rename(columns={"id": "transaction_id"})  

        # Load labels
        df_labels = pd.read_parquet(self.gold_label_dir)

        # Merge on common key, e.g., "transaction_id"
        df_combined = pd.merge(df_all_features, df_labels, on="transaction_id", how="inner")

        if df_combined.empty:
            raise ValueError("Merged dataframe is empty. Check key column alignment between features and labels.")
        else:
            logger.info("Data Loaded Successfully")

        return df_combined
