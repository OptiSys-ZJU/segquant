import onnxruntime
from typing import Any, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace
from stable_diff.model.controlnet_sd3 import AbstractSD3ControlNetModel
import inspect
import json
import torch
import numpy as np

class SD3ControlNetONNXModel(AbstractSD3ControlNetModel):
    @classmethod
    def from_config(cls, config_path, onnx_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            model = cls(onnx_path=onnx_path,
                        sample_size=config['sample_size'],
                        patch_size=config['patch_size'],
                        in_channels=config['in_channels'],
                        num_layers=config['num_layers'],
                        attention_head_dim=config['attention_head_dim'],
                        num_attention_heads=config['num_attention_heads'],
                        joint_attention_dim=config['joint_attention_dim'],
                        caption_projection_dim=config['caption_projection_dim'],
                        pooled_projection_dim=config['pooled_projection_dim'],
                        out_channels=config['out_channels'],
                        pos_embed_max_size=config['pos_embed_max_size'])
            return model
    
    def __init__(self,
                onnx_path: str,
                sample_size: int = 128,
                patch_size: int = 2,
                in_channels: int = 16,
                num_layers: int = 18,
                attention_head_dim: int = 64,
                num_attention_heads: int = 18,
                joint_attention_dim: int = 4096,
                caption_projection_dim: int = 1152,
                pooled_projection_dim: int = 2048,
                out_channels: int = 16,
                pos_embed_max_size: int = 96,
                extra_conditioning_channels: int = 0,
                dual_attention_layers: Tuple[int, ...] = (),
                qk_norm: Optional[str] = None,
                pos_embed_type: Optional[str] = "sincos",
                use_pos_embed: bool = True,
                force_zeros_for_pooled_projection: bool = True):
        
        self.config = SimpleNamespace()
        init_params = inspect.signature(self.__init__).parameters.keys()
        for key, value in locals().items():
            if key in init_params:
                setattr(self.config, key, value)

        # 
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.input_names = self.get_input_name(self.onnx_session)
        self.output_names = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_names))
        print("output_name:{}".format(self.output_names))

    def __call__(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        
        res = self.__forward__((hidden_states.cpu().numpy(), controlnet_cond.cpu().numpy(), np.array(conditioning_scale), encoder_hidden_states.cpu().numpy(), pooled_projections.cpu().numpy(), timestep.cpu().numpy()))
        controlnet_block_res_samples = [torch.from_numpy(x).half().to('cuda') for x in res]
        return (controlnet_block_res_samples,)

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def __forward__(self, input):
        input_feeds = {}
        i = 0
        for input_name in self.input_names:
            input_feeds[input_name] = input[i]
            i += 1
        res = self.onnx_session.run(self.output_names, input_feeds)
        return res