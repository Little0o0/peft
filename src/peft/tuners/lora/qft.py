import torch
from torch import nn
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

import qfinetuning as qft
from qfinetuning.nn.modules import Linear8bitScale_Normal

class LoRA_Linear8bitScale_Normal(torch.nn.Module, LoraLayer):
    def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            **kwargs,
        ) -> None:

        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            # if self.merged:
            #     self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        # elif self.merged:
        #     result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    compute_dtype = lora_A.weight.dtype
                    if x.dtype != compute_dtype:
                        x = x.to(compute_dtype)
                output = lora_B(lora_A(dropout(x)))
                if requires_conversion:
                    output = output.to(expected_dtype)
                output = output * scaling
                result = result + output

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

def dispatch_qft_8bit_normal(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
    if loaded_in_8bit and isinstance(target_base_layer, Linear8bitScale_Normal):
        eightbit_kwargs = kwargs.copy()
        eightbit_kwargs.update(
            {
                "outlier_idx": target.state.outlier_idx,
                "S_momentum": target.state.S_momentum,
                "S_init": target.state.S_init,
                "quant_type": target.state.quant_type,
                "layer_name":  target.state.layer_name,
                "grad_precision": target.state.grad_precision
            }
        )
        new_module = LoRA_Linear8bitScale_Normal(target, adapter_name, **eightbit_kwargs)

    return new_module