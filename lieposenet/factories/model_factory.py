from ..models.pose_net import PoseNet
from ..criterions.pose_net_criterion import PoseNetCriterion
from ..criterions.se3_criterion import SE3Criterion
from ..criterions.poe_se3_criterion import POESE3Criterion
from ..criterions.parametrized_poe_se3_criterion import ParametrizedPOESE3Criterion
from torchvision import models


def remove_name_key(parameters):
    result = {}
    for key, value in parameters.items():
        if key != "name":
            result[key] = value
    return result


class ModelFactory(object):
    def make_model(self, parameters):
        feature_extractor = self.make_feature_extractor(parameters.feature_extractor)
        criterion = self.make_criterion(parameters.criterion)

        if parameters.name == "pose_net":
            return PoseNet(parameters, feature_extractor, criterion)
        else:
            raise ValueError("Unknown model name: {}".format(parameters.name))

    @staticmethod
    def make_feature_extractor(parameters):
        if parameters.name == "resnet34":
            return models.resnet34(pretrained=parameters.pretrained)
        else:
            raise ValueError("Unknown feature_extractor name: {}".format(parameters.name))

    @staticmethod
    def make_criterion(parameters):
        if parameters.name == "pose_net_criterion":
            return PoseNetCriterion(**remove_name_key(parameters))
        elif parameters.name == "se3":
            return SE3Criterion()
        if parameters.name == "poe_se3":
            return POESE3Criterion(**remove_name_key(parameters))
        if parameters.name == "param_poe_se3":
            return ParametrizedPOESE3Criterion(**remove_name_key(parameters))
        else:
            raise ValueError("Unknown criterion name: {}".format(parameters.name))
