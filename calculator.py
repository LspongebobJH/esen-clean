from typing import Optional
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

import torch
import torch.nn as nn


class WBMCalculator(Calculator):
    def __init__(
        self,
        model: nn.Module,
        a2g_func: callable,
        device: Optional[torch.device] = None,
        divide_stress_by: float = 1.0,
        **kwargs,
    ):
        """Initializes the calculator.

        Args:
            model (EsenRegressor): The model to use for predictions.
        """
        Calculator.__init__(self, **kwargs)
        self.results = {}  # type: ignore
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)  # type: ignore

        self.a2g = a2g_func

        self.divide_stress_by = divide_stress_by

        # assume all E&F&S are computed
        properties = [
            "energy", "free_energy",
            "forces",
            "stress",
        ]
        self.implemented_properties = properties

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Calculate properties.

        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        data_object = self.a2g(atoms)
        data_object = data_object.to(self.device)

        self.results = {}
        out = self.model.predict(data_object)
        if "energy" in self.implemented_properties:
            self.results["energy"] = float(out["energy"].detach().cpu().item())
            self.results["free_energy"] = self.results["energy"]

        if "forces" in self.implemented_properties:
            self.results["forces"] = out["forces"].detach().cpu().numpy()

        if "stress" in self.implemented_properties:
            raw_stress = out["stress"].detach().cpu().numpy()
            if raw_stress.size == 9: # transform to 6-d vogits rep
                raw_stress = full_3x3_to_voigt_6_stress(raw_stress.reshape(3, 3))
            elif raw_stress.size == 6: # already in 6-d vogits rep
                raw_stress = raw_stress[0] if len(raw_stress.shape) > 1 else raw_stress
            else:
                raise ValueError(
                    f"Invalid stress shape: {raw_stress.shape}. Expected (3, 3) or (6,)."
                )

            # the stress unit must be eV/A^3
            self.results["stress"] = raw_stress / self.divide_stress_by
