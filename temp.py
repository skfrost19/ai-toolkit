# import yaml
# import re

# yaml_path = "config/modal_lora_config.yaml"

# with open(yaml_path, "r") as file:
#     data = file.read()
#     # Uncomment the trigger_word line if it is commented
#     data = re.sub(r"#\s*(trigger_word:)", r"\1", data)
#     data = yaml.safe_load(data)
#     print(data)


def update_yaml(args_dict):
    from ruamel.yaml import YAML
    import re

    yaml = YAML()
    yaml.preserve_quotes = True

    yaml_path = args_dict["config_file_list"][0]
    with open(yaml_path, "r") as file:
        data = file.read()

    # Uncomment the trigger_word line if it is commented
    data = re.sub(r"#\s*(trigger_word:)", r"\1", data)

    yaml_data = yaml.load(data)

    if args_dict["trigger_word"]:
        yaml_data["config"]["process"][0]["trigger_word"] = args_dict["trigger_word"]
    yaml_data["config"]["name"] = args_dict["lora_name"]
    yaml_data["config"]["process"][0]["volume_name"] = args_dict["modal_volume_name"]

    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)


import argparse

parser = argparse.ArgumentParser()

# require at least one config file
parser.add_argument(
    "config_file_list",
    nargs="+",
    type=str,
    help="Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially",
)

# flag to continue if a job fails
parser.add_argument(
    "-r",
    "--recover",
    action="store_true",
    help="Continue running additional jobs even if a job fails",
)

# optional name replacement for config file
parser.add_argument(
    "-n",
    "--name",
    type=str,
    default=None,
    help="Name to replace [name] tag in config file, useful for shared config file",
)
parser.add_argument(
    "--trigger_word",
    type=str,
    default=None,
    help="Trigger word to be added to captions.",
)
parser.add_argument(
    "--lora_name",
    type=str,
    default="my_first_flux_lora_v1",
    help="Name for the LoRA model, this will be the folder as well as the model name.",
)
parser.add_argument(
    "--modal_volume_name",
    type=str,
    default="flux-lora-models",
    help="Name of the modal volume.",
)
args = parser.parse_args()

args_dict = vars(args)

print(args_dict)

# take the name of config file passed as argument and pass it to
# update_yaml(args_dict)

# import yaml

# with open("config/modal_lora_config.yaml", "r") as file:
#     data = yaml.safe_load(file)
#     print(data)
