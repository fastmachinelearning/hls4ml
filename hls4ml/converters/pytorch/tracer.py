import torch


class CustomFXTracer(torch.fx.Tracer):

    def is_leaf_module(self, m, module_qualified_name: str) -> bool:
        """
        Custom Tracher class for hls4ml to define brevitas modules as leaf modules so they are not traced through by torch.FX
        """
        import torch

        return (
            m.__module__.startswith("torch.nn")
            or m.__module__.startswith("torch.ao.nn")
            or m.__module__.startswith("brevitas.nn")
        ) and not isinstance(m, torch.nn.Sequential)
