"""Centralized constants for API validation and defaults."""

from typing import Final

VEHICLE_BRANDS: Final[tuple[str, ...]] = (
    "B12",
    "B3",
    "B2",
    "B5",
    "B4",
    "B6",
    "B10",
    "B1",
    "B13",
    "B11",
    "B14",
)

VEHICLE_GAS_TYPES: Final[tuple[str, ...]] = (
    "Regular",
    "Diesel",
)

REGIONS: Final[tuple[str, ...]] = (
    "R72",
    "R91",
    "R52",
    "R11",
    "R94",
    "R93",
    "R31",
    "R82",
    "R22",
    "R21",
    "R42",
    "R54",
    "R73",
    "R41",
    "R26",
    "R25",
    "R24",
    "R53",
    "R83",
    "R23",
    "R74",
    "R43",
)

AREAS: Final[tuple[str, ...]] = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
)

VEH_POWER_RANGE: Final[tuple[int, int]] = (1, 20)
VEH_AGE_RANGE: Final[tuple[int, int]] = (0, 120)
DRIV_AGE_RANGE: Final[tuple[int, int]] = (18, 120)
DENSITY_RANGE: Final[tuple[int, int]] = (1, 30000)
BONUS_MALUS_RANGE: Final[tuple[int, int]] = (50, 230)
