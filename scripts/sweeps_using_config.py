import typer
from pathlib import Path
from spacy.training.loop import train
from spacy.training.initialize import init_nlp
from spacy import util
import spacy

import yaml
from thinc.api import Config
import wandb


def main(default_config: Path, yaml_config: Path, output_path: Path, project_name=str):
    def train_spacy():
        loaded_local_config = util.load_config(default_config)
        with wandb.init() as run:
            spacy.prefer_gpu()

            sweeps_config = Config(util.dot_to_dict(run.config))
            merged_config = Config(loaded_local_config).merge(sweeps_config)

            nlp = init_nlp(merged_config)
            output_path.mkdir(parents=True, exist_ok=True)
            train(nlp, output_path, use_gpu=True)

    with open(yaml_config) as fp:
        sweep_config = yaml.load(fp, Loader=yaml.SafeLoader)

    loaded_local_config = util.load_config(default_config)

    sweep_id = wandb.sweep(sweep_config, project=loaded_local_config["variables"]["wandb_project_name"])
    wandb.agent(sweep_id, train_spacy, count=20)


if __name__ == "__main__":
    typer.run(main)
