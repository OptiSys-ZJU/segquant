"""
This module provides functionality for processing and training an affiner
based on a given configuration, dataset, and models. It supports both
blockwise and third-party affiner types.
"""

import json
import numpy as np
from segquant.solver.blockwise_affiner import BlockwiseAffiner
from segquant.solver.recurrent_steper import RecurrentSteper
from segquant.subset_wrapper import SubsetWrapper
import os
import torch 
from tqdm import tqdm

def load_affiner(dump_path):
    if not os.path.exists(dump_path):
        return None
    else:
        print(f"[INFO] affiner [{dump_path}] found, loading...")
        pickle_file = torch.load(dump_path)
        return RecurrentSteper.load_from_file(pickle_file, latents=None, device="cuda:0")

def create_affiner(
    config, dataset, model_real, model_quant, latents=None, shuffle=True, dump_path=None
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
    Returns:
        affiner: The trained affiner instance.
    """

    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
        dataset = SubsetWrapper(dataset, indices)

    affine_type = config["type"]
    recurrent = config["config"]["recurrent"]
    max_timestep = config["config"]["max_timestep"]
    if affine_type == "blockwise":
        affiner = BlockwiseAffiner.from_config(config, latents=latents, device="cuda:0")
    elif affine_type == "tac":
        from baseline.tac_diffusion import TACDiffution
        affiner = TACDiffution.from_config(config, latents=latents, device="cuda:0")
    elif affine_type == "ptqd":
        from baseline.ptqd import PTQD
        affiner = PTQD.from_config(config, latents=latents, device="cuda:0")
    else:
        raise ValueError(f"Unknown stepper type: {config['type']}")

    if recurrent:
        for _ in tqdm(range(max_timestep), desc="Learning Affiner Parameters"):
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
    if dump_path is not None:
        affiner.dump(dump_path)

    return affiner
