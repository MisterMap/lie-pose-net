from pytorch_lightning.core.saving import get_filesystem, rank_zero_warn
import yaml
from pytorch_lightning.utilities.parsing import AttributeDict


def make_attribute_dict(dictionary):
    if type(dictionary) != dict:
        return dictionary
    new_dictionary = {}
    for key, value in dictionary.items():
        new_dictionary[key] = make_attribute_dict(value)
    return AttributeDict(**new_dictionary)


def load_hparams_from_yaml(config_yaml: str):
    fs = get_filesystem(config_yaml)
    if not fs.exists(config_yaml):
        rank_zero_warn(f"Missing Tags: {config_yaml}.", RuntimeWarning)
        return {}

    with fs.open(config_yaml, "r") as fp:
        tags = yaml.full_load(fp)
    return make_attribute_dict(tags)
