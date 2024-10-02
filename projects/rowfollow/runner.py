import sys

import mlflow

import rowfollow_maml
import maml_config

tracking_uri = sys.argv[1]
experiment_name = sys.argv[2]
config_file_path = sys.argv[3]

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    rowfollow_maml.main(*maml_config.load_configuration(config_file_path))
