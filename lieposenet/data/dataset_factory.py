from .seven_scenes import SevenScenes
from .odometry_seven_scenes import OdometrySevenScenes


class DatasetFactory(object):
    @staticmethod
    def make_dataset(parameters, **kwargs):
        if parameters.name == "seven_scenes":
            return SevenScenes(**parameters, **kwargs)
        elif parameters.name == "odom_seven_scenes":
            return OdometrySevenScenes(**parameters, **kwargs)
        else:
            raise ValueError("Unknown model name: {}".format(parameters.name))
