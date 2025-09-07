import hydra
from omegaconf import DictConfig, OmegaConf

import importlib

from data.base import make_loader
from model import make_model

import wandb

wandb.login(key="0755651ac5ef404426fbce162c5d70ae390e7b9e")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    
    wandb.init(
        project = f"{config.data.name}_{config.model.name_or_path}",
        name = f"{config.editor.name}_{str(config.data.n_edits)}",
        config = OmegaConf.to_container(config, resolve = True)
    )
    print("********** HERE A ****************")
    data_module = importlib.import_module(f"data.{config.data.name}")
    data_class = getattr(data_module, f"{config.data.name.upper()}Dataset")
    print(data_class)
    print(config)
    train_loader, valid_loader = make_loader(config, data_class)

    print("********** HERE B ****************")

    model = make_model(config.model).to(config.model_device)
    print("********** HERE C ****************")

    editor_module = importlib.import_module(f"editor.{config.editor.name}")
    editor_class = getattr(editor_module, config.editor.name.upper())
    editor = editor_class(config, model)
    editor.run(train_loader, valid_loader)
    print("********** HERE D ****************")

if __name__ == "__main__":
    main()
