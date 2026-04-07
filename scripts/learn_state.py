from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
from multiscat import OptimizationConfig
from multiscat.basis import (
    close_coupling_basis,
    scattering_metadata_from_stacked_delta_x,
    split_scattering_metadata,
)
from multiscat.config import MorseScatteringCondition, momentum_from_angles
from multiscat.multiscat import get_scattering_state
from scipy.constants import angstrom as angstrom_si  # type: ignore[import-untyped]
from scipy.constants import (  # type: ignore[import-untyped]
    atomic_mass,
    electron_volt,
    physical_constants,
)
from slate_quantum import operator

# Constants
HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3
Z_HEIGHT = 8

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

PARAMS_MIN = np.array([5.0, 0.5, 2.0, 0.05, 2, 6, 0.0, 0.0, 1, 0.1], dtype=np.float64)
PARAMS_MAX = np.array(
    [10.0, 1.5, 4.0, 0.20, 3.5, 16, np.pi / 2, 2 * np.pi, 30, 30],
    dtype=np.float64,
)


def denormalize_params(
    params_norm: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters back to their original physical units."""
    return params_norm * (PARAMS_MAX - PARAMS_MIN) + PARAMS_MIN


def condition_from_params(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> MorseScatteringCondition:
    """Convert a tensor of parameters into a ScatteringCondition."""
    depth, height, offset, beta, unit_cell, z_height, theta, phi, energy, mass = (
        denormalize_params(
            params,
        )
    )

    morse_params = operator.build.CorrugatedMorseParameters(
        depth=depth * electron_volt * 10**-3,
        height=height * angstrom_si,
        offset=offset * angstrom_si,
        beta=beta,
    )

    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([unit_cell * angstrom_si, 0, 0]),
            np.array([0, unit_cell * angstrom_si, 0]),
            np.array([0, 0, z_height * angstrom_si]),
        ),
        (15, 15, 200),
    )

    return MorseScatteringCondition(
        mass=mass * atomic_mass,
        morse_parameters=morse_params,
        metadata=metadata,
        incident_k=momentum_from_angles(
            theta=theta,
            phi=phi,
            energy=energy * electron_volt * 10**-3,
            mass=mass * atomic_mass,
        ),
    )


def normalize_params(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters to a [0, 1] range."""
    return (params - PARAMS_MIN) / (PARAMS_MAX - PARAMS_MIN)


def params_from_condition(
    condition: MorseScatteringCondition,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Extract the parameters from a ScatteringCondition."""
    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)
    unit_cell = metadata_x01.children[0].delta / angstrom_si
    z_height = metadata_z.delta / angstrom_si

    morse_parameters = condition.morse_parameters

    depth = morse_parameters.depth / (electron_volt * 10**-3)
    height = morse_parameters.height / angstrom_si
    offset = morse_parameters.offset / angstrom_si
    beta = morse_parameters.beta

    theta = condition.theta
    phi = condition.phi
    energy = condition.incident_energy / (electron_volt * 10**-3)
    mass = condition.mass / atomic_mass

    return normalize_params(
        np.array(
            [
                depth,
                height,
                offset,
                beta,
                unit_cell,
                z_height,
                theta,
                phi,
                energy,
                mass,
            ],
        ),
    )


def simulate_state(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Wrap your physics code into a single callable function."""
    condition = condition_from_params(params)

    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=80)
    state = get_scattering_state(condition, config)

    data = state.with_basis(
        close_coupling_basis(condition.metadata),
    ).raw_data.reshape(15, 15, 200)
    return np.array([data.real, data.imag])


def generate_dataset_hdf5(filepath: Path, num_samples: int = 1000) -> None:
    """Generate parameters and S-matrices, saving them directly to disk."""
    if filepath.exists():
        print(f"Dataset already exists at {filepath}. Skipping generation.")
        return

    print(
        f"Generating {num_samples} samples straight to disk. This may take a while...",
    )
    rng = np.random.default_rng()

    # Open an HDF5 file in write mode
    with h5py.File(filepath, "w") as f:
        # Create empty datasets on disk
        input_data = f.create_dataset("X", shape=(num_samples, 10), dtype=np.float64)  # type: ignore[hd5]
        output_data = f.create_dataset(  # type: ignore[hd5]
            "Y",
            shape=(num_samples, 2, 15, 15, 200),
            dtype=np.float64,
        )

        for i in range(num_samples):
            params = rng.uniform(size=10)

            input_data[i] = params
            output_data[i] = simulate_state(params)


def generate() -> None:
    for i in range(50):
        data_path = Path(f"data/15/state_data.{i}.hdf5")
        generate_dataset_hdf5(data_path, num_samples=1000)
