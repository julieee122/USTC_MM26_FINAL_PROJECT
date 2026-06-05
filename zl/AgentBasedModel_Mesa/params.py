from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ABMParams:
    alpha: float = 3.0
    beta: float = 3.0
    sU: float = 0.2
    sM: float = 0.2
    gammaU: float = 2.0
    gammaM: float = 2.0
    qAU: float = 0.0
    qBU: float = 0.0
    qAM: float = 0.0
    qBM: float = 0.0
    bAU: float = 0.0
    bBU: float = 0.0
    bAM: float = 0.0
    bBM: float = 0.0
    pAU: float = 0.0
    pBU: float = 0.0
    rA: float = 0.0
    rB: float = 0.0
    sigmaU: float = 0.2
    sigmaM: float = 0.2
