import torch
from typing import Optional

class ScalarNormalizer(torch.nn.Module):
    """Scalar normalizer that learns mean and std from data.

    NOTE: Multi-dimensional tensors are flattened before updating
    the running mean/std. This is desired behaviour for force targets.
    """

    def __init__(
        self,
        init_mean: Optional[torch.Tensor | float] = None,
        init_std: Optional[torch.Tensor | float] = None,
        init_num_batches: Optional[int] = 1000,
        online: bool = True,
    ) -> None:
        """Initializes the ScalarNormalizer.

        To enhance training stability, consider setting an init mean + std.
        """
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(1, affine=False, momentum=None)  # type: ignore
        if init_mean is not None:
            assert init_std is not None
            self.bn.running_mean = torch.tensor([init_mean])
            self.bn.running_var = torch.tensor([init_std**2])
            self.bn.num_batches_tracked = torch.tensor([init_num_batches])
        assert isinstance(online, bool)
        self.online = online

    def forward(self, x: torch.Tensor, online: Optional[bool] = None) -> torch.Tensor:
        """Normalize by running mean and std."""
        online = online if online is not None else self.online
        x_reshaped = x.reshape(-1, 1)
        if self.training and online and x_reshaped.shape[0] > 1:
            # hack: call batch norm, but only to update a running mean/std
            self.bn(x_reshaped)

        mu = self.bn.running_mean  # type: ignore
        sigma = torch.sqrt(self.bn.running_var)  # type: ignore
        if sigma < 1e-6:
            raise ValueError("ScalarNormalizer has ~zero std.")

        return (x - mu) / sigma  # type: ignore

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the forward normalization."""
        return x * torch.sqrt(self.bn.running_var) + self.bn.running_mean  # type: ignore

