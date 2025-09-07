import torch

import yaml

with open("bert-large.yaml", "r") as file:
    config = yaml.safe_load(file)


checkpoint = torch.load(config["weight_path"], map_location="cpu")
print(checkpoint.keys())

