import torch


class HLS4MLModule(torch.nn.Module):
    """
    Custom PyTorch module class for hls4ml to define custom modules that shouldn't be traced through by torch.FX
    """

    pass


class CustomFXTracer(torch.fx.Tracer):

    def is_leaf_module(self, m, module_qualified_name: str) -> bool:
        """
        Custom Tracer class for hls4ml to define Brevitas modules and custom modules as leaf modules so they are not traced
        through by torch.FX
        """
        import torch

        return (
            isinstance(m, HLS4MLModule)
            or m.__module__.startswith('torch.nn')
            or m.__module__.startswith('torch.ao.nn')
            or m.__module__.startswith('brevitas.nn')
        ) and not isinstance(m, torch.nn.Sequential)
