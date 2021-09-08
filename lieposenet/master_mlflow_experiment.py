import json
import itertools
from clearml import Task


class MasterMlflowExperiment(object):
    def __init__(self, repo, branch, script, queue, project_name, parameters):
        self._parameters = json.loads(parameters)
        self._parameter_setups = self.make_parameter_setups(self._parameters)
        self._repo = repo
        self._branch = branch
        self._script = script
        self._queue = queue
        self._project_name = project_name

    @staticmethod
    def make_parameter_setups(parameter_dict):
        keys = list(parameter_dict.keys())
        print(parameter_dict)
        iterators = [parameter_dict[key] if isinstance(parameter_dict[key], list
                                                       ) else [parameter_dict[key]] for key in keys]

        parameter_setups = []
        for parameters in itertools.product(*iterators):
            parameter_setups.append({})
            for key, parameter in zip(keys, parameters):
                parameter_setups[-1][key] = parameter
        return parameter_setups

    def remote_run_experiment(self):
        for parameter_setup in self._parameter_setups:
            print(parameter_setup)
            task = Task.create(
                project_name=f"{self._project_name}",
                task_name=f'remote_task',
                repo=self._repo,
                branch=self._branch,
                script=self._script,
                requirements_file="../requirements.txt"
            )
            task.set_parent(Task.current_task().id)
            task.connect(parameter_setup)
            Task.enqueue(task, self._queue)
