def update_yaml(args_dict):
    from ruamel.yaml import YAML
    import re

    yaml = YAML()
    yaml.preserve_quotes = True

    yaml_path = "config/modal_lora_config.yaml"
    with open(yaml_path, "r") as file:
        data = file.read()

    if args_dict["trigger_word"]:
        # Uncomment the trigger_word line if it is commented
        data = re.sub(r"#\s*(trigger_word:)", r"\1", data)

    yaml_data = yaml.load(data)

    if args_dict["trigger_word"]:
        yaml_data["config"]["process"][0]["trigger_word"] = args_dict["trigger_word"]
    yaml_data["config"]["name"] = args_dict["lora_name"]
    yaml_data["config"]["process"][0]["volume_name"] = args_dict["modal_volume_name"]

    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # optional name replacement for config file [lora_name]
    parser.add_argument(
        "-l",
        "--lora_name",
        type=str,
        required=True,
        help="LoRA folder as well as the name of the LoRAs to be created after fine-tuning. It should be unique.e.g. my_first_flux_lora_v1",
    )

    # trigger word replacement for config file [trigger_word]
    parser.add_argument(
        "-t",
        "--trigger_word",
        type=str,
        required=True,
        help="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
    )

    # convert it to dictionary and pass it to the function to update
    args_dict = vars(parser.parse_args())
    update_yaml(args_dict)

    # if everything is successful, run a modal job
    "modal run --detach run_modal.py --config-file-list=/root/ai-toolkit/{CONFIG_PATH}"
    CONFIG_PATH = args_dict["config_file_list"].split("/")[-1]
    # if everything is successful, run a modal job
    import os

    os.system(
        f"modal run --detach run_modal.py --config-file-list=/root/ai-toolkit/{CONFIG_PATH}"
    )
