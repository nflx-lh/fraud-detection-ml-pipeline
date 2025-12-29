import pandas as pd
from pathlib import Path
from typing import List
from ml.utils.utils import logger
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class Preprocessor():
    """
    This class handles the preprocessing of the loaded data from the gold layer, 
    it handles the dropping of columns, train test split, and oot splits, also handles smote
    """
    def __init__(self, df:pd.DataFrame, use_smote:bool, oot_splits:int = None, oot_period:int = None, columns_to_keep = None,**kwargs):
        self.df = df
        self.use_smote = use_smote
        self.oot_splits = oot_splits
        self.oot_period = oot_period
        self.oot = [] # a list of dataframes
        self.columns_to_keep = columns_to_keep
        self.transform_pipeline = None
        self.kwargs = kwargs
    def preprocess(self):
        """
        handles the entire preprocessing flow, returns a dictionary containing the split data with the oot if required
        """
        # pre split processing steps
        self.pre_split_processing()

        # train test split  
        y = self.df['is_fraud']
        X = self.df.drop(columns=['is_fraud'])

        # perform the train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # post split processing, fit on train. self.transform_pipeline is used to export out to MLFlow
        self.transform_pipeline = self._fit(X_train)

        # transform on train and test
        X_train = self.transform_pipeline.transform(X_train)
        X_test = self.transform_pipeline.transform(X_test)

        # Apply SMOTE only on the training set
        if self.use_smote:
            X_train, y_train = self._smote(X_train, y_train)
            
        # handle oot
        if self.oot_splits and self.oot_period:
            processed_oot = self.oot_preprocess(self.oot, self.transform_pipeline)

        logger.info("Preprocessing Done")

        return {
            'split': (X_train, X_test, y_train, y_test),
            'oot': processed_oot if self.oot_splits and self.oot_period else None
        }

    def _smote(self, X, y):
        """
        smote function, returns the balanced x and y
        """
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)

        return X_res, y_res

    def oot_preprocess(self, oot_list: List[pd.DataFrame], pipeline:Pipeline):
        """
        handles the processing of the oot sets, each oot set is a test set, just need to separate labels from features, and perform transformations of features, 
        returns a nested list of feature & label pairs for each oot set
        """
        temp = []
        for df in oot_list:
            y = df['is_fraud']
            X = df.drop(columns=['is_fraud'])
            X = pipeline.transform(X)
            temp.append((X, y))
        
        return temp

    def pre_split_processing(self):
        """
        handle operations like drop duplicates, separate into oot if required, convert labels into correct type (0 and 1 for binary classification), dropping columns here too
        """
        # drop duplicates
        self.df = self.df.drop_duplicates()

        # replace labels with 0 and 1
        self.df['is_fraud'] = self.df['is_fraud'].map({'yes': 1, 'no': 0})

        # before oot, sort the columns based on the date
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Sort by datetime ascending
        self.df = self.df.sort_values(by='date', ascending=True)

        # handle oot if needed
        if self.oot_splits and self.oot_period:
            self.split_oot()

        # select columns to use
        self.df = self.df[self.columns_to_keep]

        # need to save a copy of the training set for data drift detection during inference
        experiment_name = self.kwargs.get("experiment_name", "default_experiment")
        run_name = self.kwargs.get("run_name", "default_run")
        file_name = f"{experiment_name}_{run_name}.csv"
        folder_dir = Path("monitoring/reference_data")
        folder_dir.mkdir(parents=True,exist_ok=True)
        file_path = folder_dir/file_name

        self.df.to_csv(file_path, index=False)


    def split_oot(self):
        """"
        handles the oot splits, returns the remaining data set and the oot 
        """
        for split in range(self.oot_splits):
            latest_date = self.df['date'].max()
            
            cutoff = latest_date - pd.Timedelta(days=self.oot_period)
            oot_mask = self.df['date'] >= cutoff

            # select last oot_period rows in dataframe
            self.oot.append(self.df[oot_mask].copy())
            self.df = self.df[~oot_mask]

    
    # separate the fit function so that we can only use it on the train set
    def _fit(self, X):
        """
        handles fitting of the imputer and scaler to the train dataset, returns a fitted pipeline which can be used to transform data
        """
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

        pipeline = full_pipeline.fit(X)

        return pipeline
    
        

    
