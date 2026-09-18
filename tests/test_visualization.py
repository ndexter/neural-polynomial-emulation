import pytest
import torch
from torch import nn

from neural_polynomial_emulation import (
    MultivariateBasisEmulator,
    interactive_network_figure,
    multi_index_set,
    to_feedforward,
)
from neural_polynomial_emulation.visualization import _whole_network_layout


@pytest.mark.parametrize("product", ["relu", "tanh", "repu2"])
def test_whole_network_layout_contains_every_affine_layer_once(product):
    source = MultivariateBasisEmulator(
        multi_index_set(2, 3, "total_degree"),
        product=product,
        num_layers=3,
        tanh_tolerance=1e-5,
    )
    model = to_feedforward(source, dtype=torch.float64)
    layout = _whole_network_layout(model)

    displayed = [
        item.name
        for block in layout.blocks
        for item in block.items
        if isinstance(item.module, nn.Linear)
    ]
    expected = [
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]

    assert len(displayed) == len(set(displayed))
    assert set(displayed) == set(expected)
    assert len(layout.outputs) == len(model.multi_index)
    assert {block.kind for block in layout.blocks} >= {
        "root",
        "univariate_multiplier",
        "normalization",
        "cross_multiplier",
    }


def test_whole_network_layout_requires_materialized_affine_layers():
    model = MultivariateBasisEmulator(
        multi_index_set(2, 2, "total_degree"), product="repu2"
    )

    with pytest.raises(ValueError, match="to_feedforward"):
        _whole_network_layout(model)


def test_interactive_network_figure_contains_all_architectural_trace_types():
    pytest.importorskip("plotly")
    model = to_feedforward(
        MultivariateBasisEmulator(
            multi_index_set(2, 2, "total_degree"),
            product="repu2",
        ),
        dtype=torch.float64,
    )

    figure = interactive_network_figure(model)
    trace_names = {trace.name for trace in figure.data}

    assert trace_names >= {
        "root shifts and constants",
        "univariate multiplier layers",
        "polynomial normalizations",
        "cross-coordinate multiplier layers",
        "elementwise activations",
        "computational wiring",
        "coordinate inputs",
        "basis outputs",
    }
    assert figure.layout.scene.bgcolor == "#071018"
    assert figure.layout.scene.xaxis.visible is False
    assert figure.layout.scene.yaxis.visible is False
    assert figure.layout.scene.zaxis.visible is False
    assert figure.layout.scene.xaxis.showbackground is False
    assert figure.layout.scene.yaxis.showbackground is False
    assert figure.layout.scene.zaxis.showbackground is False
