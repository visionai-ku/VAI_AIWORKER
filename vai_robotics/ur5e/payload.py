"""The active tool load reported by the UR controller."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ToolPayload:
    """Total mass in kg with its CoG in metres in the tool flange frame."""

    mass_kg: float
    cog_m: tuple[float, float, float]

    def __post_init__(self):
        mass = float(self.mass_kg)
        cog = tuple(float(coordinate) for coordinate in self.cog_m)
        if not math.isfinite(mass) or mass < 0:
            raise ValueError("Payload mass must be finite and non-negative.")
        if len(cog) != 3 or not all(math.isfinite(coordinate) for coordinate in cog):
            raise ValueError("Payload CoG must have three finite flange coordinates.")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "cog_m", cog)
