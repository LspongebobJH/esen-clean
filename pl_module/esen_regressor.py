from types import SimpleNamespace
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from fairchem.core.common.utils import cg_change_mat, irreps_sum
from fairchem.core.models.esen.nn.so3_layers import SO3_Linear
# LinearReferenceEnergy
from fairchem.core.modules.normalization.element_references import \
    LinearReferences
from .reference_energies import REFERENCE_ENERGIES
from .normalizer import ScalarNormalizer


def num_params(model: nn.Module):
    '''print the size of backbone, heads and total'''
    nparams = sum(p.numel() for p in model.parameters())
    return nparams


def full_3x3_to_voigt_6_stress_torch(stress_tensor: torch.Tensor):
    """Convert (N, 3, 3) stress tensor to (N, 6) Voigt form."""
    assert stress_tensor.shape[-2:] == (3, 3)

    return torch.stack([
        stress_tensor[:, 0, 0], 
        stress_tensor[:, 1, 1],
        stress_tensor[:, 2, 2],
        stress_tensor[:, 1, 2],
        stress_tensor[:, 0, 2],
        stress_tensor[:, 0, 1],
    ], dim=1)


def compose_tensor(
    trace: torch.Tensor,
    l2_symmetric: torch.Tensor,
) -> torch.Tensor:
    """Re-compose a tensor from its decomposition

    Args:
        trace: a tensor with scalar part of the decomposition of r2 tensors in the batch
        l2_symmetric: tensor with the symmetric/traceless part of decomposition

    Returns:
        tensor: rank 2 tensor
    """

    if trace.shape[1] != 1:
        raise ValueError("batch of traces must be shape (batch size, 1)")

    if l2_symmetric.shape[1] != 5:
        raise ValueError("batch of l2_symmetric tensors must be shape (batch size, 5)")

    if trace.shape[0] != l2_symmetric.shape[0]:
        raise ValueError(
            "Shape missmatch between trace and l2_symmetric parts. The first dimension is the batch dimension"
        )

    batch_size = trace.shape[0]
    decomposed_preds = torch.zeros(
        batch_size, irreps_sum(2), device=trace.device
    )  # rank 2
    decomposed_preds[:, : irreps_sum(0)] = trace
    decomposed_preds[:, irreps_sum(1) : irreps_sum(2)] = l2_symmetric

    r2_tensor = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(2, device=trace.device),
        decomposed_preds,
    )

    return r2_tensor

def decompose_tensor(stress: torch.Tensor, rank: int=2):
    '''
    Decomposes a rank-2 tensor (e.g., stress) into irreducible components under SO(3)
    using a Clebsch-Gordan (CG) basis transformation. Adds decomposed parts (e.g.,
    isotropic and anisotropic stress) to the data object for use in equivariant ML models.
    '''
    # Only support second-order tensors (e.g., symmetric 3x3 stress)
    if stress.shape[-1] != 3 or stress.shape[-2] != 3:
        raise ValueError('stress must be a 3x3 tensor')

    if rank != 2:
        raise ValueError('rank must be 2')

    # Reshape the input tensor into shape (B, 9), assuming it's symmetric and 3x3
    # Then apply the CG basis transformation: einsum does matrix multiplication
    tensor_decomposition = torch.einsum(
        "ab, cb->ca",                        # (9,9) x (B,9) -> (B,9)
        cg_change_mat(rank).to(stress.device),  # (9,9)
        stress.reshape(-1, irreps_sum(rank))  # Flatten to (B,9)
    )

    # For each component to extract (e.g., l=0 or l=2 irreps)
    stress_isotropic = tensor_decomposition[:, 0:irreps_sum(0)]
    stress_anisotropic = tensor_decomposition[:, irreps_sum(1):irreps_sum(2)]
    
    return stress_isotropic, stress_anisotropic

def mean_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    error_type: Literal["mae", "mse", "huber_0.01"] = "mae",
    batch: Batch = None,
) -> torch.Tensor:
    """Compute MAE or MSE for node or graph targets.

    Args:
        target: The target tensor.
        pred: The prediction tensor.
        error_type: The type of error to compute. Either "mae" or "mse" or "huber_x"
            where x is the delta parameter for the huber loss.
        batch: batch indexes of each atom
    """
    if error_type.startswith("huber"):
        huber_delta = float(error_type.split("_")[1])
        error_type = "huber"  # type: ignore
        assert huber_delta > 0.0, "HUBER_DELTA must be greater than 0.0"

    error_function = {
        "mae": lambda x, y: torch.abs(x - y),
        "mse": lambda x, y: (x - y) ** 2,
        "huber": lambda x, y: F.huber_loss(x, y, reduction="none", delta=huber_delta),
    }[error_type]

    errors = error_function(pred, target)
    errors = errors.mean(dim=-1) if errors.dim() > 1 else errors

    if weights is not None:
        assert len(weights) == len(errors), \
            f"weights length {len(weights)} does not match errors length {len(errors)}"
        errors = errors * weights.view_as(errors)

    if batch is not None:
        error = torch.zeros(
            len(batch["natoms"]),
            device=errors.device,
            dtype=errors.dtype,
        )
        error.index_add_(0, batch["batch"], errors.view(-1))
        error /= batch["natoms"] # mean error of each graph in the batch
        error = error.mean()
    else:
        error = errors.mean()

    return error


class EnergyHead(nn.Module):
    def __init__(
        self, 
        sphere_channels: int,
        hidden_channels: int,
        reduce: str = "mean",
        loss_coeff: float = 1.0,
        loss_type: Literal["mae", "mse", "huber_0.01"] = "mae",
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.reduce = reduce

        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        # load normalizer and energy ref
        ref = REFERENCE_ENERGIES["vasp-shifted"]
        if self.reduce == "mean":
            means = torch.tensor([ref.residual_mean_per_atom])
            stds = torch.tensor([ref.residual_std_per_atom])
        elif self.reduce == "sum":
            means = torch.tensor([ref.residual_mean])
            stds = torch.tensor([ref.residual_std])
        else:
            raise ValueError

        self.normalizer = ScalarNormalizer(init_mean=means, init_std=stds)
        self.reference = LinearReferences(
            element_references=torch.tensor(ref.coefficients, dtype=torch.float32),
        )

        self.loss_coeff = loss_coeff
        self.loss_type = loss_type

    def forward(self, batch: Batch, emb: dict[str, torch.Tensor]):
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze()
        ).view(-1, 1, 1)

        energy = torch.zeros(
            len(batch["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )
        energy.index_add_(0, batch["batch"], node_energy.view(-1))

        if self.reduce == "mean":
            return energy / batch["natoms"]
        elif self.reduce == "sum":
            return energy
        else:
            raise ValueError
    
    def predict(self, batch: Batch, emb: dict[str, torch.Tensor]):
        """
        always predict total energy
        """
        pred = self(batch, emb)
        pred = self.denormalize(pred, batch)
        return pred
    
    def loss(self, batch: Batch, emb: dict[str, torch.Tensor]):
        raw_target = batch['energy'].view(-1) # total energy
        target = self.normalize(raw_target, batch)
        
        pred = self(batch, emb).view(-1)
        assert pred.shape == target.shape, f"pred.shape={pred.shape}, target.shape={target.shape}"
        loss = mean_error(pred, target, error_type=self.loss_type)

        # raw loss to trace
        raw_pred = self.denormalize(pred, batch)
        metrics = {
            "raw_total_energy_mae": torch.abs(raw_pred - raw_target).mean().item(),
            "raw_peratom_energy_mae": torch.abs(
                raw_pred / batch.natoms - \
                raw_target / batch.natoms).mean().item()  # raw loss (energy / atom)
        }
        return SimpleNamespace(loss=loss * self.loss_coeff, log=metrics)

    def denormalize(self, x: torch.Tensor, batch: Batch):
        """
        denormalize and add linear reference
        always return the **TOTAL** energy

        return: total energy
        """
        x = self.normalizer.inverse(x).view(-1) # (bs,)
        if self.reduce == "mean":
            x = x * batch["natoms"]
        return self.reference(x, batch, reshaped=False)
        # x += self.reference(
        #     atom_types=batch["atomic_numbers"], 
        #     n_node=batch["natoms"]
        # ).squeeze(-1)
    
    def normalize(self, x: torch.Tensor, batch: Batch, online: bool=True):
        """
        substract linear reference and normalize
        x: total energy
        """
        # reference = self.reference(batch["atomic_numbers"], n_node=batch["natoms"]).squeeze(-1)
        # x -= reference
        x = self.reference.dereference(x, batch, reshaped=False)
        if self.reduce == "mean":
            x = x / batch["natoms"]
        return self.normalizer(x, online)


class ForcesHead(nn.Module):
    """Node prediction head that can be appended to a base model.

    This head could be added to the foundation model to enable
    auxiliary tasks during pretraining, or added afterwards
    during a finetuning step.
    """
    def __init__(
        self,
        sphere_channels: int,
        train_on_free_atoms: bool = False,
        loss_coeff: float = 1.0,
        loss_type: Literal["mae", "mse", "huber_0.01"] = "mae",
    ):
        super().__init__()
        self.linear = SO3_Linear(sphere_channels, 1, lmax=1)

        self.normalizer = ScalarNormalizer()

        self.train_on_free_atoms = train_on_free_atoms
        self.loss_coeff = loss_coeff
        self.loss_type = loss_type

    def forward(self, batch: Batch, emb: dict[str, torch.Tensor]) -> torch.Tensor:
        forces = self.linear(emb["node_embedding"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        return forces

    def predict(self, batch: Batch, emb: dict[str, torch.Tensor]):
        pred = self(batch, emb)
        return self.normalizer.inverse(pred)

    def loss(self, batch: Batch, emb: dict[str, torch.Tensor]):
        raw_target = batch['forces'].squeeze(-1) # (natoms, 3)
        pred = self(batch, emb)

        if self.train_on_free_atoms:
            # mask = batch["fixed"] == 0
            # pred = pred[mask]
            # target = target[mask]   
            raise NotImplementedError(
                "ForcesHead does not support training on free atoms yet."
            )

        target = self.normalize(raw_target, batch)
        assert pred.shape == target.shape
        
        loss = mean_error(pred, target, error_type=self.loss_type, batch=batch)

        raw_pred = self.denormalize(pred, batch)
        raw_mae = mean_error(raw_pred, raw_target, error_type="mae", batch=batch)

        metrics = {
            "node_mae_raw": raw_mae.item(),
            "node_cosine_sim": torch.cosine_similarity(raw_pred, raw_target, dim=-1)
            .mean()
            .item(),
        }
        return SimpleNamespace(loss=loss * self.loss_coeff, log=metrics)

    def denormalize(self, x: torch.Tensor, batch: Batch):
        """Denormalize the force prediction."""
        return self.normalizer.inverse(x)

    def normalize(self, x: torch.Tensor, batch: Batch, online: bool = True):
        """Normalize the force prediction."""
        return self.normalizer(x, online=online)


class DeNSForcesHead(nn.Module):
    """Node prediction head that can be appended to a base model.

    This head could be added to the foundation model to enable
    auxiliary tasks during pretraining, or added afterwards
    during a finetuning step.
    """
    def __init__(
        self,
        sphere_channels: int,
        train_on_free_atoms: bool = False,
        forces_loss_coeff: float = 1.0,
        denoising_loss_coeff: float = 1.0,
        loss_type: Literal["mae", "mse", "huber_0.01"] = "mae",
    ):
        super().__init__()
        self.forces_linear = SO3_Linear(sphere_channels, 1, lmax=1)
        self.denoising_linear = SO3_Linear(sphere_channels, 1, lmax=1)

        self.forces_normalizer = ScalarNormalizer()
        self.denoising_normalizer = ScalarNormalizer()

        self.train_on_free_atoms = train_on_free_atoms

        self.forces_loss_coeff = forces_loss_coeff
        self.denoising_loss_coeff = denoising_loss_coeff
        self.loss_type = loss_type

    def forward(self, batch: Batch, emb: dict[str, torch.Tensor]) -> torch.Tensor:
        forces = self.forces_linear(emb["node_embedding"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()

        denoising_pos_vec = self.denoising_linear(emb["node_embedding"].narrow(1, 0, 4))
        denoising_pos_vec = denoising_pos_vec.narrow(1, 1, 3)
        denoising_pos_vec = denoising_pos_vec.view(-1, 3).contiguous()

        # NOTE: should always in this form: forces = denoising_pos_vec * a + forces * b
        # s.t. the computational graph remains same
        if hasattr(batch, "denoising_pos_forward") and batch.denoising_pos_forward:
            if hasattr(batch, "noise_mask"):
                noise_mask_tensor = batch.noise_mask.view(-1, 1)
                forces = denoising_pos_vec * noise_mask_tensor + forces * (
                    ~noise_mask_tensor
                )
            else:
                forces = denoising_pos_vec + forces * 0
        else:
            forces = denoising_pos_vec * 0 + forces

        return forces

    def predict(self, batch: Batch, emb: dict[str, torch.Tensor]):
        if hasattr(batch, "denoising_pos_forward") and batch.denoising_pos_forward:
            raise ValueError(
                "Denoising objective should not be used for prediction, only for training."
            )
        
        pred = self(batch, emb) # only forces
        return self.forces_normalizer.inverse(pred)

    def compute_atomwise_denoising_pos_and_force_hybrid_loss(
        self, pred, target, noise_mask, batch: Optional[Batch] = None
    ):       
        force_index = torch.where(noise_mask == 0)
        denoising_pos_index = torch.where(noise_mask == 1)
        mult_tensor = torch.ones_like(noise_mask, dtype=pred.dtype, device=pred.device)
        mult_tensor[force_index] *= self.forces_loss_coeff
        mult_tensor[denoising_pos_index] *= self.denoising_loss_coeff
        loss = mean_error(pred, target, weights=mult_tensor, error_type=self.loss_type, batch=batch)

        return loss

    def loss(self, batch: Batch, emb: dict[str, torch.Tensor]):
        if self.train_on_free_atoms:
            raise NotImplementedError(
                "DeNSForcesHead does not support training on free atoms yet."
            )
        
        pred = self(batch, emb)

        raw_forces_target = batch['forces'].squeeze(-1) # (natoms, 3)
        forces_target = self.forces_normalizer(raw_forces_target)
        assert pred.shape == forces_target.shape, \
            f"pred.shape={pred.shape}, forces_target.shape={forces_target.shape}"

        if hasattr(batch, "denoising_pos_forward") and batch.denoising_pos_forward:
            raw_denoising_target = batch.noise_vec
            denoising_target = self.denoising_normalizer(raw_denoising_target)
            if hasattr(batch, "noise_mask"):
                noise_mask = batch.noise_mask.view(-1, 1)
                target = denoising_target * noise_mask + forces_target * (~noise_mask)
                loss = self.compute_atomwise_denoising_pos_and_force_hybrid_loss(
                    pred, target, noise_mask, batch,
                )
            else:
                target = denoising_target
                loss = mean_error(pred, target, error_type=self.loss_type, batch=batch) * self.denoising_loss_coeff
        else:
            target = forces_target
            loss = mean_error(pred, target, error_type=self.loss_type, batch=batch) * self.forces_loss_coeff

        # raw loss to trace
        raw_forces_pred = self.forces_normalizer.inverse(pred)
        if hasattr(batch, "denoising_pos_forward") and batch.denoising_pos_forward:
            raw_denoising_pred = self.denoising_normalizer.inverse(pred)
            if hasattr(batch, "noise_mask"):
                raw_pred = raw_denoising_pred * batch.noise_mask.unsqueeze(-1) + raw_forces_pred * (~batch.noise_mask.unsqueeze(-1))
                raw_target = raw_denoising_target * batch.noise_mask.unsqueeze(-1) + raw_forces_target * (~batch.noise_mask.unsqueeze(-1))
            else:
                raw_pred = raw_denoising_pred
                raw_target = raw_denoising_target
        else:
            raw_pred = raw_forces_pred
            raw_target = raw_forces_target
        raw_mae = mean_error(raw_pred, raw_target, error_type="mae", batch=batch)
        
        metrics = {
            "node_mae_raw": raw_mae.item(),
            "node_cosine_sim": torch.cosine_similarity(raw_pred, raw_target, dim=-1)
            .mean()
            .item(),
        }
        return SimpleNamespace(loss=loss, log=metrics)
    

class CartBasisStressHead(nn.Module):
    """
    predict the isotropic and anisotropic parts of the stress tensor
    to ensure symmetry and then recompose back to the full stress tensor.
    Use the cartesian basis stress to compute loss.
    """
    def __init__(
        self, 
        sphere_channels: int,
        hidden_channels: int,
        reduce: str = "mean",
        loss_coeff: float = 1.0,
        loss_type: Literal["mae", "mse", "huber_0.01"] = "mae",
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.reduce = reduce

        self.scalar_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )
        self.l2_linear = SO3_Linear(self.sphere_channels, 1, lmax=2)

        self.normalizer = ScalarNormalizer()

        self.loss_coeff = loss_coeff
        self.loss_type = loss_type

    def forward(self, batch: Batch, emb: dict[str, torch.Tensor]) -> torch.Tensor:
        # isotropic part
        node_scalar = self.scalar_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        iso_stress = torch.zeros(
            len(batch["natoms"]),
            device=node_scalar.device,
            dtype=node_scalar.dtype,
        )
        iso_stress.index_add_(0, batch["batch"], node_scalar.view(-1))
        if self.reduce == "mean":
            iso_stress /= batch["natoms"]

        # anisotropic part
        node_l2 = self.l2_linear(emb["node_embedding"].narrow(1, 0, 9))
        node_l2 = node_l2.narrow(1, 4, 5)
        node_l2 = node_l2.view(-1, 5).contiguous()

        aniso_stress = torch.zeros(
            (len(batch["natoms"]), 5),
            device=node_l2.device,
            dtype=node_l2.dtype,
        )
        aniso_stress.index_add_(0, batch["batch"], node_l2)
        
        if self.reduce == "mean":
            aniso_stress /= batch["natoms"].unsqueeze(1)

        stress = compose_tensor(iso_stress.unsqueeze(1), aniso_stress).reshape(-1, 3, 3)

        return stress

    def predict(self, batch: Batch, emb: dict[str, torch.Tensor]) -> torch.Tensor:
        stress = self(batch, emb)
        pred = self.denormalize(stress)
        return pred
    
    def loss(self, batch: Batch, emb: dict[str, torch.Tensor]):
        raw_target = batch['stress'].reshape(-1, 3, 3)
        target = self.normalize(raw_target)

        pred = self(batch, emb)
        assert pred.shape == target.shape
        loss = mean_error(pred, target, error_type=self.loss_type)
        
        # raw loss to trace
        raw_pred = self.denormalize(pred)
        metrics = {
            "stress_mae_raw": torch.abs(
                full_3x3_to_voigt_6_stress_torch(raw_pred) - \
                full_3x3_to_voigt_6_stress_torch(raw_target)
            ).mean().item(),
        }
        return SimpleNamespace(loss=loss * self.loss_coeff, log=metrics)

    def denormalize(self, stress: torch.Tensor):
        stress = self.normalizer.inverse(stress)
        return stress

    def normalize(self, stress: torch.Tensor):
        stress = self.normalizer(stress)
        return stress


class CGBasisStressHead(nn.Module):
    """
    predict the isotropic and anisotropic parts of the stress tensor
    to ensure symmetry and then recompose back to the full stress tensor
    Use the Clebsch-Gordan basis stress to compute loss.
    """
    def __init__(
        self, 
        sphere_channels: int,
        hidden_channels: int,
        reduce: str = "mean",
        loss_coeff: float = 1.0,
        loss_type: Literal["mae", "mse", "huber_0.01"] = "mae",
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.reduce = reduce

        self.scalar_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )
        self.l2_linear = SO3_Linear(self.sphere_channels, 1, lmax=2)

        self.iso_normalizer = ScalarNormalizer()
        self.aniso_normalizer = ScalarNormalizer()

        self.loss_coeff = loss_coeff
        self.loss_type = loss_type

    def forward(self, batch: Batch, emb: dict[str, torch.Tensor]) -> torch.Tensor:
        # isotropic part
        node_scalar = self.scalar_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        iso_stress = torch.zeros(
            len(batch["natoms"]),
            device=node_scalar.device,
            dtype=node_scalar.dtype,
        )
        iso_stress.index_add_(0, batch["batch"], node_scalar.view(-1))
        if self.reduce == "mean":
            iso_stress /= batch["natoms"]

        # anisotropic part
        node_l2 = self.l2_linear(emb["node_embedding"].narrow(1, 0, 9))
        node_l2 = node_l2.narrow(1, 4, 5)
        node_l2 = node_l2.view(-1, 5).contiguous()

        aniso_stress = torch.zeros(
            (len(batch["natoms"]), 5),
            device=node_l2.device,
            dtype=node_l2.dtype,
        )
        aniso_stress.index_add_(0, batch["batch"], node_l2)
        
        if self.reduce == "mean":
            aniso_stress /= batch["natoms"].unsqueeze(1)

        return iso_stress.unsqueeze(1), aniso_stress

    def predict(self, batch: Batch, emb: dict[str, torch.Tensor]) -> torch.Tensor:
        iso_stress, aniso_stress = self(batch, emb)
        pred = self.denormalize(iso_stress, aniso_stress)
        return pred
    
    def loss(self, batch: Batch, emb: dict[str, torch.Tensor]):
        raw_target = batch['stress'].reshape(-1, 3, 3)
        iso_target, aniso_target = self.normalize(raw_target)

        iso_pred, aniso_pred = self(batch, emb)
        assert iso_pred.shape == iso_target.shape, f"iso_pred.shape={iso_pred.shape}, iso_target.shape={iso_target.shape}"
        assert aniso_pred.shape == aniso_target.shape, f"aniso_pred.shape={aniso_pred.shape}, aniso_target.shape={aniso_target.shape}"
        
        loss_iso = mean_error(iso_pred, iso_target, error_type=self.loss_type)
        loss_aniso = mean_error(aniso_pred, aniso_target, error_type=self.loss_type)
        loss = loss_iso + loss_aniso
        
        # raw loss to trace
        raw_pred = self.denormalize(iso_pred, aniso_pred)
        metrics = {
            "stress_mae_raw": torch.abs(
                full_3x3_to_voigt_6_stress_torch(raw_pred) - \
                full_3x3_to_voigt_6_stress_torch(raw_target)
            ).mean().item(),
        }
        return SimpleNamespace(loss=loss * self.loss_coeff, log=metrics)

    def denormalize(self, iso_stress: torch.Tensor, aniso_stress: torch.Tensor):
        iso_stress = self.iso_normalizer.inverse(iso_stress)
        aniso_stress = self.aniso_normalizer.inverse(aniso_stress)
        stress = compose_tensor(iso_stress, aniso_stress)
        stress = stress.reshape(-1, 3, 3)
        return stress

    def normalize(self, stress: torch.Tensor):
        '''
        decomposes the stress tensor into isotropic and anisotropic parts
        and normalizes them separately.
        '''
        iso_stress, aniso_stress = decompose_tensor(stress) # (B,1) and (B,5)
        iso_stress = self.iso_normalizer(iso_stress)
        aniso_stress = self.aniso_normalizer(aniso_stress)

        return iso_stress, aniso_stress


class EsenRegressor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        energy_head: Optional[nn.Module] = None,
        forces_head: Optional[nn.Module] = None,
        stress_head: Optional[nn.Module] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone

        self.compute_energy = energy_head is not None
        self.compute_forces = forces_head is not None
        self.compute_stress = stress_head is not None

        self.energy_head = energy_head
        self.forces_head = forces_head
        self.stress_head = stress_head

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        print("EsenRegressor with direct forces initialized")
        self.print_num_params()

    def forward(self, data: Batch):
        pass

    def predict(self, data: Batch):
        emb = self.backbone(data)
        output = {'embedding': emb}
        if self.compute_energy:
            output["energy"] = self.energy_head.predict(data, emb)
        
        if self.compute_forces:
            output["forces"] = self.forces_head.predict(data, emb)

        if self.compute_stress:
            output["stress"] = self.stress_head.predict(data, emb)

        return output
    
    def loss(self, data: Batch):
        emb = self.backbone(data)
        loss = torch.tensor(0.0)
        metrics = {}
        if self.compute_energy:
            energy_out = self.energy_head.loss(data, emb)
            metrics.update(energy_out.log)
            loss = loss.type_as(energy_out.loss)
            loss += energy_out.loss

        if self.compute_stress:
            stress_out = self.stress_head.loss(data, emb)
            metrics.update(stress_out.log)
            loss = loss.type_as(stress_out.loss)
            loss += stress_out.loss

        if self.compute_forces:
            force_out = self.forces_head.loss(data, emb)
            metrics.update(force_out.log)
            loss = loss.type_as(force_out.loss)
            loss += force_out.loss

        metrics['loss'] = loss.item()
        return SimpleNamespace(loss=loss, log=metrics)

    def print_num_params(self):
        '''print the size of backbone, heads and total'''
        total_nparams = 0
        total_nparams += num_params(self.backbone)
        if self.energy_head is not None:
            total_nparams += num_params(self.energy_head)
            print(f"{self.energy_head.__class__.__name__}: [nparams] Number of Parameters: {num_params(self.energy_head)}")
        if self.forces_head is not None:
            total_nparams += num_params(self.forces_head)
            print(f"{self.forces_head.__class__.__name__}: [nparams] Number of Parameters: {num_params(self.forces_head)}")
        if self.stress_head is not None:
            total_nparams += num_params(self.stress_head)
            print(f"{self.stress_head.__class__.__name__}: [nparams] Number of Parameters: {num_params(self.stress_head)}")
            total_nparams += num_params(self.stress_head)
        print(f"{self.__class__.__name__}: [nparams] Number of Parameters: {total_nparams}")
        return total_nparams

if __name__=="__main__":
    # stress->decompose->compose->stress'
    # test if stress == stress'
    stress = torch.randn(10, 3, 3) 
    stress = (stress + stress.transpose(1, 2)) / 2 # make it symmetric
    stress = stress + torch.eye(3).unsqueeze(0) * 0.1 # add a small isotropic part
    stress = stress / 10.0 # scale down
    iso_stress, aniso_stress = decompose_tensor(stress)
    stress_ = compose_tensor(iso_stress, aniso_stress).reshape(-1, 3, 3)
    print("stress_ == stress", torch.allclose(stress_, stress, atol=1e-5))
