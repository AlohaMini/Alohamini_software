import pytest
from alohamini_mujoco.model import ModelProfile, build_model


@pytest.fixture(scope="session")
def robot():
    return build_model(profile=ModelProfile(initial_positions={"vertical_move": 0.2}))
