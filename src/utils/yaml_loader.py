import yaml

def load_model_config(yaml_file="configs/model.yml"):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)