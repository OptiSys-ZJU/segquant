import torch
from baseline.tac_diffusion import TACDiffution
from dataset.affine.noise_dataset import NoiseDataset
from segquant.torch.blockwise_affine import BlockwiseAffiner


if __name__ == '__main__':
    blocksizes = [1, 2, 4, 8, 16, 32, 64, 128]

    tac_affiner = TACDiffution(max_timestep=60)

    block_affiners = [BlockwiseAffiner(max_timestep=60, blocksize=block) for block in blocksizes]
    dataset = NoiseDataset('../dataset/affine_noise')

    def learning(timestep, real, quant, affiner, log_prefix):
        if hasattr(affiner, 'blocksize'):
            K = torch.ones_like(real)
            b = torch.zeros_like(real)
            init = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

            K, b = affiner.step_learning(timestep, quant, real)
            affine = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

            print(f'[{log_prefix}] [{timestep}] Learning block[{affiner.blocksize}] init: [{init[0]:5f}/{init[1]:5f}], affine: [{affine[0]:5f}/{affine[1]:5f}]')
        else:
            K = torch.ones_like(real)
            init = (affiner.loss(K, quant, real), affiner.error(K, quant, real))

            K = affiner.step_learning(timestep, quant, real)
            affine = (affiner.loss(K, quant, real), affiner.error(K, quant, real))

            print(f'[{log_prefix}] [{timestep}] Learning init: [{init[0]:5f}/{init[1]:5f}], affine: [{affine[0]:5f}/{affine[1]:5f}]')


    ###### learning
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
    learning_sample = 8
    step = 0
    for batch in dataloader:
        assert isinstance(batch, list) and len(batch) == dataset.max_timestep
        for data in batch:
            timestep = int(data["timestep"][0].item())
            quant = data["quant"]
            real = data["real"]
            for affiner in block_affiners:
                learning(timestep, real, quant, affiner, 'Blockwise')
            learning(timestep, real, quant, tac_affiner, 'Tac')

        step += 1
        if step >= learning_sample:
            break
    
    ##### testing
    def testing(timestep, real, quant, affiner, log_prefix):
        if hasattr(affiner, 'blocksize'):
            K = torch.ones_like(real)
            b = torch.zeros_like(real)
            init = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

            K, b = affiner.get_solution(timestep)
            affine = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

            K, b = affiner.step_learning(timestep, quant, real)
            opt = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

            print(f'[{log_prefix}] [{timestep}] block[{affiner.blocksize}] init: [{init[0]:5f}/{init[1]:5f}], affine: [{affine[0]:5f}/{affine[1]:5f}], opt: [{opt[0]:5f}/{opt[1]:5f}]')
        else:
            K = torch.ones_like(real)
            init = (affiner.loss(K, quant, real), affiner.error(K, quant, real))

            K = affiner.get_solution(timestep)
            affine = (affiner.loss(K, quant, real), affiner.error(K, quant, real))

            K = affiner.step_learning(timestep, quant, real)
            opt = (affiner.loss(K, quant, real), affiner.error(K, quant, real))

            print(f'[{log_prefix}] [{timestep}] init: [{init[0]:5f}/{init[1]:5f}], affine: [{affine[0]:5f}/{affine[1]:5f}], opt: [{opt[0]:5f}/{opt[1]:5f}]')

    dataloader = dataset.get_dataloader(batch_size=1, shuffle=True)
    test_sample = 2
    step = 0
    for batch in dataloader:
        assert isinstance(batch, list) and len(batch) == dataset.max_timestep
        for data in batch:
            timestep = int(data["timestep"][0].item())
            quant = data["quant"]
            real = data["real"]

            for affiner in block_affiners:
                testing(timestep, real, quant, affiner, 'Blockwise')
            testing(timestep, real, quant, tac_affiner, 'Tac')
        step += 1
        if step >= test_sample:
            break
