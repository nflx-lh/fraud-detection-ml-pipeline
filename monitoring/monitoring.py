from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *
from evidently import ColumnMapping

from evidently.ui.workspace import Workspace
from typing import List

#evidently ui --port 8080 --host 0.0.0.0

class Monitoring:
    """"
    Uses evidently AI, creates a workspace and project, 
    
    """
    def __init__(
        self,
        reference_data_path: str = None,
        inference_data_path: str = None,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None,
        project_name: str = None,
    ):
        self.reference_data_path = reference_data_path
        self.inference_data_path = inference_data_path
        self.project_name = project_name
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.ws = Workspace.create("workspace")

    def run_monitoring(self):
        # get the data from the inference and reference data path
        reference_df, inference_df = self.get_data()
        # create a new project
        project = self.create_project()

        # get the drift report
        report = self.check_drift(reference_df, inference_df)

        # add the report to the workspace
        self.ws.add_report(project.id, report)

    # implements the datadrift calculation, returns a report
    def check_drift(self, reference_df, inference_df):
        column_mapping = ColumnMapping()
        total_features = self.numerical_features + self.categorical_features

        inference_df = inference_df[total_features]
        reference_df = reference_df[total_features]

        column_mapping.numerical_features = self.numerical_features
        column_mapping.categorical_features = self.categorical_features

        data_drift_report = Report(metrics=[
            DataDriftPreset(cat_stattest = 'psi'),
        ])

        data_drift_report.run(reference_data=reference_df, current_data=inference_df, column_mapping = column_mapping)

        return data_drift_report

    def get_data(self):

        reference_file_path = Path(self.reference_data_path)

        inference_file_path = Path(self.inference_data_path)

        if not reference_file_path.exists():
            raise FileNotFoundError("No reference dataset")
        else:
            reference_df = pd.read_csv(reference_file_path, header=0)


        if not inference_file_path.exists():
            raise FileNotFoundError("No inference dataset")
        else:
            inference_df = pd.read_csv(inference_file_path, header=0)

        return reference_df, inference_df

    # creates a project under the current workspace
    def create_project(self):
        project = self.ws.create_project(self.project_name)
        project.description = "CS611 Demo Sample Project"
        project.save()

        return project
