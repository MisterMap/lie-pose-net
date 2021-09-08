from lieposenet.master_mlflow_experiment import MasterMlflowExperiment
import argparse
from clearml import Task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run many experiments in ClearML server")
    parser.add_argument("-repo", default="https://github.com/MisterMap/lie-pose-net", help="Repository")
    parser.add_argument("-branch", default="main")
    parser.add_argument("-script", default="run_experiment.py")
    parser.add_argument("-queue", default="default")
    parser.add_argument("-project_name", default="lie-pose-net")
    parser.add_argument("-parameters")
    args = parser.parse_args()

    task = Task.init("master-lie-pose-net", "Run multiple experiments", reuse_last_task_id=False)

    experiment = MasterMlflowExperiment(**vars(args))
    experiment.remote_run_experiment()
