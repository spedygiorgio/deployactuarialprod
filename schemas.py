"""Pydantic schemas used by the FastAPI application."""

from typing import Literal

from pydantic import BaseModel, Field

from constants import (
    AREAS,
    BONUS_MALUS_RANGE,
    DENSITY_RANGE,
    DRIV_AGE_RANGE,
    REGIONS,
    VEHICLE_BRANDS,
    VEHICLE_GAS_TYPES,
    VEH_AGE_RANGE,
    VEH_POWER_RANGE,
)


class Insured(BaseModel):
    """Input contract for policyholder features."""

    VehPower: int = Field(
        title="Vehicle Power",
        description="Vehicle power in CV",
        ge=VEH_POWER_RANGE[0],
        le=VEH_POWER_RANGE[1],
        default=5,
    )
    VehAge: int = Field(
        title="Vehicle Age",
        description="Vehicle age in years",
        ge=VEH_AGE_RANGE[0],
        le=VEH_AGE_RANGE[1],
        default=1,
    )
    DrivAge: int = Field(
        title="Driver Age",
        description="Driver age in years",
        ge=DRIV_AGE_RANGE[0],
        le=DRIV_AGE_RANGE[1],
        default=35,
    )
    Density: int = Field(
        title="Density",
        description="Density of inhabitants per km2",
        gt=0,
        le=DENSITY_RANGE[1],
        default=100,
    )
    BonusMalus: int = Field(
        title="Bonus Malus",
        description="Bonus Malus",
        ge=BONUS_MALUS_RANGE[0],
        le=BONUS_MALUS_RANGE[1],
        default=100,
    )
    VehBrand: Literal[
        "B12", "B3", "B2", "B5", "B4", "B6", "B10", "B1", "B13", "B11", "B14"
    ] = Field(
        title="Vehicle Brand",
        description=f"Allowed values: {', '.join(VEHICLE_BRANDS)}",
    )
    VehGas: Literal["Regular", "Diesel"] = Field(
        title="Vehicle Gas",
        description=f"Allowed values: {', '.join(VEHICLE_GAS_TYPES)}",
    )
    Region: Literal[
        "R72", "R91", "R52", "R11", "R94", "R93", "R31", "R82", "R22", "R21", "R42", "R54", "R73", "R41", "R26", "R25", "R24", "R53", "R83", "R23", "R74", "R43"
    ] = Field(
        title="Region",
        description=f"Allowed values: {', '.join(REGIONS)}",
    )
    Area: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(
        title="Area",
        description=f"Allowed values: {', '.join(AREAS)}",
    )


class PredictionResponse(BaseModel):
    """Output contract returned by /predict/."""

    Frequency: float
    Severity: float
    Pure_Premium: float
