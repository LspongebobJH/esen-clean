from types import SimpleNamespace
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from fairchem.core.common.utils import conditional_grad
from pl_module.normalizer import ScalarNormalizer
from pl_module.esen_regressor import (full_3x3_to_voigt_6_stress_torch,
                                      mean_error)


def num_params(model: nn.Module):
    '''print the size of backbone, heads and total'''
    nparams = sum(p.numel() for p in model.parameters())
    return nparams

class EsenConserRegressor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        energy_head: nn.Module,
        regress_forces: bool = True,
        regress_stress: bool = True,
        forces_loss_type: Literal["mae", "mse", "huber_0.01"] = "mae",
        stress_loss_type: Literal["mae", "mse", "huber_0.01"] = "mae",
        forces_loss_coeff: float = 1.0,
        stress_loss_coeff: float = 1.0,
        train_on_free_atoms: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.direct_forces = False
        self.backbone.regress_stress = regress_stress
        self.backbone.regress_forces = regress_forces

        self.energy_head = energy_head

        self.direct_forces = False
        self.regress_stress = regress_stress
        self.regress_forces = regress_forces
        if self.regress_forces:
            self.forces_normalizer = ScalarNormalizer()
            self.forces_loss_type = forces_loss_type
            self.forces_loss_coeff = forces_loss_coeff
        if self.regress_stress:
            self.stress_normalizer = ScalarNormalizer()
            self.stress_loss_type = stress_loss_type
            self.stress_loss_coeff = stress_loss_coeff

        self.train_on_free_atoms = train_on_free_atoms

        self.energy_key = "energy"
        self.forces_key = "forces"
        self.stress_key = "stress"

        print("EsenConserRegressor with conservative forces initialized")
        self.print_num_params()

    @conditional_grad(torch.enable_grad())
    def forward(self, batch: Batch):
        outputs = {}

        emb = self.backbone(batch)
        outputs['embedding'] = emb
        base_energy = self.energy_head(batch, emb)
        raw_energy = self.energy_head.denormalize(base_energy, batch)
        # NOTE: use fixed mean & std for energy normalization
        outputs[self.energy_key] = self.energy_head.normalize(
            raw_energy, batch, online=False)

        if self.regress_stress:
            grads = torch.autograd.grad(
                [raw_energy.sum()],
                [batch["pos"], emb["displacement"]],
                create_graph=self.training,
            )
            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(batch["cell"]).abs().unsqueeze(-1)
            # cauchy stress = virial stress / volume
            stress = virial / volume.view(-1, 1, 1)
            virial = torch.neg(virial)  # virial stress
            outputs[self.forces_key] = forces
            outputs[self.stress_key] = stress
            batch["cell"] = emb["orig_cell"]
        elif self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    raw_energy, batch["pos"], create_graph=self.training
                )[0]
            )
            outputs[self.forces_key] = forces

        return outputs

    def predict(self, batch: Batch):
        """Predict energy, forces, and stress."""
        preds = self(batch)

        out = {'embedding': preds['embedding']}
        out[self.energy_key] = self.energy_head.denormalize(
            preds[self.energy_key], batch
        )
        if self.regress_forces:
            out[self.forces_key] = preds[self.forces_key]

        if self.regress_stress:
            out[self.stress_key] = preds[self.stress_key]

        return out  # type: ignore

    def loss(self, batch: Batch):
        out = self(batch)
        metrics = {}
        loss = torch.tensor(0.0)

        energy_pred = out[self.energy_key]
        energy_out = self.energy_loss_func(energy_pred, batch)
        metrics.update(energy_out.log)
        loss = loss.type_as(energy_out.loss)
        loss += energy_out.loss

        if self.regress_forces and self.forces_key in batch:
            raw_forces_pred = out[self.forces_key]
            forces_pred = self.forces_normalizer(raw_forces_pred, online=False)
            forces_out = self.forces_loss_func(forces_pred, batch)
            metrics.update(forces_out.log)
            loss = loss.type_as(forces_out.loss)
            loss += forces_out.loss

        if self.regress_stress and self.stress_key in batch:
            raw_stress_pred = out[self.stress_key]
            stress_pred = self.stress_normalizer(raw_stress_pred, online=False)
            stress_out = self.stress_loss_func(stress_pred, batch)
            metrics.update(stress_out.log)
            loss = loss.type_as(stress_out.loss)
            loss += stress_out.loss

        metrics['loss'] = loss.item()
        return SimpleNamespace(loss=loss, log=metrics)

    def energy_loss_func(
        self,
        pred: torch.Tensor,
        batch: Batch,
    ):
        """Energy loss and metrics."""
        pred = pred.view(-1)
        # avoid using squeeze(-1) because sometimes bs=1 and target has shape (1,)
        raw_target = batch[self.energy_key].view(-1)
        assert pred.shape == raw_target.shape, f"{pred.shape} != {raw_target.shape}"

        target = self.energy_head.normalize(raw_target, batch)
        loss = mean_error(pred, target, error_type=self.energy_head.loss_type)

        raw_pred = self.energy_head.denormalize(pred, batch)
        metrics = {
            "raw_total_energy_mae": torch.abs(raw_pred - raw_target).mean().item(),
            "raw_peratom_energy_mae": torch.abs(
                raw_pred / batch.natoms -
                # raw loss (energy / atom)
                raw_target / batch.natoms).mean().item()
        }

        return SimpleNamespace(loss=loss*self.energy_head.loss_coeff, log=metrics)

    def forces_loss_func(
        self,
        pred: torch.Tensor,
        batch: Batch,
    ):
        """Compute forces loss and metrics."""
        pred = pred.squeeze(-1)
        raw_target = batch[self.forces_key].squeeze(-1)  # (natoms, 3)

        # remove before applying normalizer
        if self.train_on_free_atoms:
            raise NotImplementedError

        target = self.forces_normalizer(raw_target)
        assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"

        loss = mean_error(
            pred, target, error_type=self.forces_loss_type, batch=batch)  # type: ignore

        raw_pred = self.forces_normalizer.inverse(pred)
        raw_mae = mean_error(raw_pred, raw_target,
                             error_type="mae", batch=batch)

        metrics = {
            "node_mae_raw": raw_mae.item(),
            "node_cosine_sim": torch.cosine_similarity(raw_pred, raw_target, dim=-1)
            .mean()
            .item(),
        }
        return SimpleNamespace(loss=loss * self.forces_loss_coeff, log=metrics)

    def stress_loss_func(
        self,
        pred: torch.Tensor,
        batch: Batch,
    ):
        # NOTE: pred stress's unit is ev/A^3!
        # NOTE: target's unit must be ev/A^3 as well!
        """Stress loss and metrics."""
        pred = pred.squeeze(-1)
        raw_target = batch[self.stress_key].reshape(-1, 3, 3)
        assert pred.shape == raw_target.shape, f"{pred.shape} != {raw_target.shape}"

        target = self.stress_normalizer(raw_target)
        loss = mean_error(pred, target, error_type=self.stress_loss_type)
        raw_pred = self.stress_normalizer.inverse(pred)
        metrics = {
            "stress_mae_raw": torch.abs(
                full_3x3_to_voigt_6_stress_torch(raw_pred) -
                full_3x3_to_voigt_6_stress_torch(raw_target)
            ).mean().item(),
        }
        return SimpleNamespace(loss=loss * self.stress_loss_coeff, log=metrics)

    def print_num_params(self):
        '''print the size of backbone, heads and total'''
        total_nparams = 0
        total_nparams += num_params(self.backbone)
        if self.energy_head is not None:
            total_nparams += num_params(self.energy_head)
            print(f"{self.energy_head.__class__.__name__}: [nparams] Number of Parameters: {num_params(self.energy_head)}")
        print(f"{self.__class__.__name__}: [nparams] Number of Parameters: {total_nparams}")
        return total_nparams