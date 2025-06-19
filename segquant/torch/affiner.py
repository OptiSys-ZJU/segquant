"""
This module provides functionality for processing and training an affiner
based on a given configuration, dataset, and models. It supports both
blockwise and third-party affiner types.
"""

import json
import numpy as np
from segquant.solver.blockwise_affiner import BlockwiseAffiner
from segquant.subset_wrapper import SubsetWrapper
import os
import torch 


def load_affiner(
    config, dataset, model_real, model_quant, latents=None, shuffle=True, thirdparty_affiner=None,
):
    """
    Process the affiner based on the provided configuration and dataset.
    Args:
        config (dict): Configuration dictionary containing stepper and solver settings.
        dataset (Dataset): The dataset to be used for training.
        model_real (Model): The real model to be trained.
        model_quant (Model): The quantized model to be trained.
        latents (optional): Latent variables for the affiner, if applicable.
        shuffle (bool): Whether to shuffle the dataset indices.
        thirdparty_affiner (optional): An external affiner instance, if provided.
    Returns:
        affiner (Tuple(BlockwiseAffiner|thirdparty_affiner)): The trained affiner instance.
    """
    affine_config_path = "../affine_configs"
    affiner_dicts_path = os.path.join(affine_config_path, f"affiner_dicts.pt")
    if not os.path.exists(affine_config_path):
        os.makedirs(affine_config_path)

    # save the config of affiner
    # if affiner_config path not exists, create a new one
    if not os.path.exists(os.path.join(affine_config_path, "affiner_config.json")):
        configs = []
        with open(os.path.join(affine_config_path, "affiner_config.json"), "w") as f:
            configs.append(config)
            json.dump(configs, f, indent=4)
    else:
        with open(os.path.join(affine_config_path, "affiner_config.json"), "r") as f:
            try:
                configs = json.load(f)
            except json.JSONDecodeError:
                configs = []

    # check if the config already exists
    if config in configs:
        print("Config already exists, loading affiner from path: ", affiner_dicts_path)
        if os.path.exists(affiner_dicts_path):
            # set up the affiner
            affiner_dicts = torch.load(affiner_dicts_path)
            affiner_dict = affiner_dicts[config]
            # TODO: not class method, need to fix
            affiner = BlockwiseAffiner.load_state_dict(affiner_dict)
            return affiner
        else:
            raise ValueError(f"Affiner path {affiner_dicts_path} does not exist, which is unexpected")

    else:
        print("Config does not exist, appending a new config to affiner_config.json")
        configs.append(config)
        with open(os.path.join(affine_config_path, "affiner_config.json"), "w") as f2:
            json.dump(configs, f2, indent=4)
        # continue to create a new affiner
        print("[INFO]Creating a new affiner")
    
    # if affiner_config not exists, continue and create a new affiner
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
        dataset = SubsetWrapper(dataset, indices)

    if config["stepper"]["type"] == "blockwise":
        affiner = BlockwiseAffiner(
            max_timestep=config["stepper"]["max_timestep"],
            blocksize=config["solver"]["blocksize"],
            sample_size=config["stepper"]["sample_size"],
            solver_type=config["solver"]["type"],
            solver_config=config["solver"],
            latents=latents,
            recurrent=config["stepper"]["recurrent"],
            noise_target=config["stepper"]["noise_target"],
            enable_latent_affine=config["stepper"]["enable_latent_affine"],
            enable_timesteps=config["stepper"]["enable_timesteps"],
        )
    else:
        if thirdparty_affiner is not None:
            affiner = thirdparty_affiner
        else:
            raise ValueError(f"Unknown stepper type: {config['stepper']['type']}")

    if config["stepper"]["recurrent"]:
        for _ in range(config["stepper"]["max_timestep"]):
            affiner.learning_real(
                model_real=model_real,
                data_loader=dataset.get_dataloader(),
                **config["extra_args"],
            )
            affiner.learning_quant(
                model_quant=model_quant,
                data_loader=dataset.get_dataloader(),
                **config["extra_args"],
            )
            affiner.replay_real(
                model_real=model_real,
                data_loader=dataset.get_dataloader(),
                **config["extra_args"],
            )
            affiner.replay_quant(
                model_quant=model_quant,
                data_loader=dataset.get_dataloader(),
                **config["extra_args"],
            )
    else:
        affiner.learning_real(
            model_real=model_real,
            data_loader=dataset.get_dataloader(),
            **config["extra_args"],
        )
        affiner.learning_quant(
            model_quant=model_quant,
            data_loader=dataset.get_dataloader(),
            **config["extra_args"],
        )

    affiner.finish_learning()

    # save the affiner
    if os.path.exists(affiner_dicts_path):
        affiner_dicts = torch.load(affiner_dicts_path)
        affiner_dicts[config] = affiner.state_dict()
        torch.save(affiner_dicts, affiner_dicts_path)
    else:
        # create a new affiner_dicts.pt
        affiner_dicts = {}
        affiner_dicts[config] = affiner.state_dict()
        torch.save(affiner_dicts, affiner_dicts_path)

    return affiner
