from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import os
import mlflow
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline
from ml.utils.utils import logger
import warnings

warnings.filterwarnings("ignore", message=".*Failed to infer the model signature.*")

class ModelManager():
    """
    A flexible class for model experimentation that allows for quick model swapping
    and hyperparameter tuning using optuna
    """
    
    def __init__(
        self, 
        model_name: str,
        model_path: str = None,
        experiment_name:str = None,
        run_name:str = None,
        n_trials = None,
        optuna_config: dict = None,
        oot: list = [],
        col_preprocessor = None
    ):
        self.model_name = model_name
        self.best_params = None  
        self.model_path = model_path
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.n_trials = n_trials
        # mapping function to map model names to actual sklearn model object
        self.function_map = {'logistic': LogisticRegression,'xgboost': XGBClassifier, 'mlp':MLPClassifier}
        # assign the model object to the attribute model
        self.model = self.function_map[model_name]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.tunable_params = optuna_config.get("optuna_config.tunable_params", {})
        self.static_params = optuna_config.get("optuna_config.static_params", {})
        self.oot = oot
        self.preprocessor = col_preprocessor
        self.results = []

    # maiin function that trains using optuna and tests the model on the test set
    def train_evaluate(self, X_train, X_test, y_train, y_test):

        # assign class variables
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # get the experiment ID
        experiment_id = self.get_or_create_experiment()

        # Set the current active MLflow experiment
        mlflow.set_experiment(experiment_id=experiment_id)

        # override Optuna's default logging to ERROR only
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        
        self.run_optuna_study(experiment_id, self.run_name, self.n_trials)

        logger.info("Training Done")

    # helper function to create experiment
    def get_or_create_experiment(self):
        """
        Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

        This function checks if an experiment with the given name exists within MLflow.
        If it does, the function returns its ID. If not, it creates a new experiment
        with the provided name and returns its ID.

        Parameters:
        - experiment_name (str): Name of the MLflow experiment.

        Returns:
        - str: ID of the existing or newly created MLflow experiment.
        """

        if experiment := mlflow.get_experiment_by_name(self.experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(self.experiment_name)
    

    # define a logging callback that will report on only new challenger parameter configurations if a
    # trial has usurped the state of 'best conditions'
    def champion_callback(self, study, frozen_trial):
        """
        Logging callback that will report when a new trial iteration improves upon existing
        best trial values.

        Note: This callback is not intended for use in distributed computing systems such as Spark
        or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
        workers or agents.
        The race conditions with file system state management for distributed trials will render
        inconsistent values with this callback.
        """

        winner = study.user_attrs.get("winner", None)

        if study.best_value and winner != study.best_value:
            study.set_user_attr("winner", study.best_value)
            if winner:
                improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
                print(
                    f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                    f"{improvement_percent: .4f}% improvement"
                )
            else:
                print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


    # plot the confusion matrix
    def plot_cm(self, model, X, y):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        # Create the confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap='Blues', ax=ax)

        plt.close(fig)  # Prevent auto-showing in notebooks

        return fig

    def objective(self, trial):
        with mlflow.start_run(run_name = 'optuna_trials', nested=True):
            # Define hyperparameters
            params = self.suggest_params_from_config(trial, self.tunable_params)
            params.update(self.static_params)

            X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train)

            # Train model
            clf = self.model(**params)

            # Fit model
            clf.fit(X_train, y_train)

            # Predict probabilities
            preds = clf.predict_proba(X_val)[:, 1]

            # Calculate loss/metric
            loss = log_loss(y_val, preds)
            auc = roc_auc_score(y_val, preds)

            #calculate f1 scores
            pred_labels = (preds >= 0.5).astype(int)
            f1 = f1_score(y_val, pred_labels)

            # Log params and metrics
            mlflow.log_params(params)
            mlflow.log_metric("log_loss", loss)
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("f1_score", f1)
            
        return f1  # or use -auc if you want to maximize it
    
    # helper function to load in configs for optuna
    def suggest_params_from_config(self, trial, tunable_params):
        params = {}
        for param_name, param_info in tunable_params.items():
            ptype = param_info.get("type")
            low = param_info.get("low")
            high = param_info.get("high")
            categorical = param_info.get("categorical")  # for categorical options
            log = param_info.get("log", False)            # whether to use log scale
            default = param_info.get("default")            # optional default value (not used by Optuna but could be handy)

            if ptype == "int":
                if log:
                    # Optuna does not directly support log scale for int, so treat as float then cast
                    val = trial.suggest_float(param_name, low, high, log=True)
                    params[param_name] = int(round(val))
                else:
                    params[param_name] = trial.suggest_int(param_name, low, high)

            elif ptype == "float":
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)

            elif ptype == "categorical":
                if categorical is None:
                    raise ValueError(f"Categorical param '{param_name}' requires 'categorical' key with list of options")
                params[param_name] = trial.suggest_categorical(param_name, categorical)

            else:
                raise ValueError(f"Unsupported param type: {ptype} for param '{param_name}'")

        return params


    def run_optuna_study(self, experiment_id, run_name, n_trials):
        """
        handles the optuna study, finds the best hyper params for the model, trains the model on it and evals it on the test set, logs the model as well
        """
        # Initiate the parent run and call the hyperparameter tuning child run logic
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            # Initialize the Optuna study
            study = optuna.create_study(direction="maximize")

            # Execute the hyperparameter optimization trials.
            # Note the addition of the `champion_callback` inclusion to control our logging
            study.optimize(self.objective, n_trials=n_trials, callbacks=[self.champion_callback])

            mlflow.log_params(study.best_params)
            mlflow.log_metric("lowest_loss", study.best_value)
            

            # Re-initialize the best model with best params
            best_params = study.best_params.copy()
            best_params.update(self.static_params)

            self.best_model = self.model(**best_params)
            self.best_model.fit(self.X_train, self.y_train)

            #log f1 score for the train set
            y_pred = self.best_model.predict(self.X_train)
            f1 = f1_score(self.y_train, y_pred)
            self.results.append(f1)

            #log f1 score for the test set
            y_pred = self.best_model.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred)

            mlflow.log_metric("f1_score", f1)
            self.results.append(f1)

            # Confusion matrix plot
            cm_fig = self.plot_cm(self.best_model, self.X_test, self.y_test)
            cm_fig.savefig(f"{run_name}_cm.png")

            mlflow.log_artifact(f"{run_name}_cm.png")
            os.remove(f"{run_name}_cm.png")

            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                       ('model', self.best_model)])

            # Log model to MLflow.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=self.X_train[0:1],
                registered_model_name=None,
                metadata={"model_data_version": 1},
            )
            warnings.resetwarnings()
            if self.oot:
                self.evaluate_on_oot()
            
            # plot the f1 score off the test set and oot set
            plt = self.plot_final_resutls()
            plt.savefig(f"{run_name}_all_f1.png")

            mlflow.log_artifact(f"{run_name}_all_f1.png")
            os.remove(f"{run_name}_all_f1.png")

    def plot_final_resutls(self):
        """"
        will plot the f1 score for the train, test and each oot split, logs artifact to the parent run
        """
        x = list(range(len(self.results)))
        x_labels = list(range(len(self.results)))
        x_labels[0] = "train f1"
        x_labels[1] = "test f1"

        for i in range(2, len(x_labels)):
            x_labels[i] = f"oot {i-1} f1"
        # Line plot
        plt.plot(x, self.results, marker='o', linestyle='-', color='blue')

        # Apply custom x-axis labels
        plt.xticks(ticks=x, labels=x_labels)

        # Add labels and title
        plt.ylabel('F1 Score')
        plt.title('Model Performance')

        return plt
    def evaluate_on_oot(self):
        """
        Evaluates the best model on each OOT (Out-Of-Time) dataset.
        Logs the F1 score for each split into MLflow.
        """

        for i, (X_oot, y_oot) in enumerate(self.oot):
            run_name = f"oot_run_{i}"
            logger.info(f"Evaluating OOT split {i}")

            y_pred = self.best_model.predict(X_oot)
            oot_f1 = f1_score(y_oot, y_pred)
            self.results.append(oot_f1)
            # Confusion matrix plot
            cm_fig = self.plot_cm(self.best_model, X_oot, y_oot)
            cm_fig.savefig(f"{run_name}_cm.png")
            # log the metrics
            with mlflow.start_run(run_name = run_name, nested=True):

                mlflow.log_artifact(f"{run_name}_cm.png")
                os.remove(f"{run_name}_cm.png")

                mlflow.log_metric(f"oot_f1_score_{i}", oot_f1)
            
            logger.info(f"Logged OOT F1 score for split {i}: {oot_f1:.4f}")

