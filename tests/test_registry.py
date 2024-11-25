# SPDX-License-Identifier: Apache-2.0

# First Party
from src.instructlab.sdg.registry import BlockRegistry


def test_block_registry():
    @BlockRegistry.register("TestFooClass")
    class TestFooClass:
        pass

    registry = BlockRegistry.get_registry()
    assert registry is not None
    assert registry["TestFooClass"] is TestFooClass
