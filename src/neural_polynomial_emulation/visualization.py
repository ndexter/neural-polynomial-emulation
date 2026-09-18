"""Interactive visualizations of materialized polynomial-emulator networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import nn

from .feedforward import FeedforwardProductNet
from .multivariate import MultivariateBasisEmulator


@dataclass
class _Item:
    name: str
    module: nn.Module


@dataclass
class _Block:
    key: str
    label: str
    kind: str
    items: list[_Item]
    dependencies: tuple[str, ...]
    lane: int
    positions: tuple[float, ...]

    @property
    def start(self) -> float:
        return self.positions[0]

    @property
    def end(self) -> float:
        return self.positions[-1]


@dataclass
class _Output:
    feature: int
    alpha: tuple[int, ...]
    dependency: str | None


@dataclass
class _NetworkLayout:
    blocks: list[_Block]
    input_dependencies: dict[int, tuple[str, ...]]
    outputs: list[_Output]


def _product_items(module: FeedforwardProductNet, prefix: str) -> list[_Item]:
    return [
        _Item(f"{prefix}.network.{index}", child)
        for index, child in enumerate(module.network)
    ]


def _whole_network_layout(model: MultivariateBasisEmulator) -> _NetworkLayout:
    """Construct a topological block layout for a converted multivariate model."""
    if not isinstance(model, MultivariateBasisEmulator):
        raise TypeError("model must be a MultivariateBasisEmulator")

    blocks: list[_Block] = []
    by_key: dict[str, _Block] = {}
    input_dependencies: dict[int, list[str]] = {
        coordinate: [] for coordinate in range(model.dimension)
    }
    polynomial_outputs: dict[tuple[int, int], str] = {}
    item_spacing = 0.82
    block_gap = 1.15

    def add_block(
        key: str,
        label: str,
        kind: str,
        items: list[_Item],
        dependencies: tuple[str, ...],
        *,
        minimum_depth: float = 0.0,
    ) -> str:
        if not items:
            raise ValueError(f"block {key!r} has no modules")
        dependencies = tuple(dict.fromkeys(dependencies))
        dependency_ends = [
            0.0 if dependency.startswith("input.") else by_key[dependency].end
            for dependency in dependencies
        ]
        start = max([minimum_depth, *dependency_ends]) + block_gap
        positions = tuple(start + item_spacing * index for index in range(len(items)))
        block = _Block(
            key=key,
            label=label,
            kind=kind,
            items=items,
            dependencies=dependencies,
            lane=len(blocks),
            positions=positions,
        )
        blocks.append(block)
        by_key[key] = block
        return key

    for coordinate, basis in enumerate(model.coordinate_emulators):
        input_key = f"input.{coordinate}"
        for degree, emulator in enumerate(basis.emulators):
            prefix = f"coordinate_emulators.{coordinate}.emulators.{degree}"
            label_prefix = f"x{coordinate}: psi_{degree}"
            if degree == 0:
                if not hasattr(emulator, "feedforward_constant_layer"):
                    raise ValueError(
                        "model must first be converted with to_feedforward"
                    )
                key = f"u.{coordinate}.{degree}.constant"
                output = add_block(
                    key,
                    f"{label_prefix} constant",
                    "root",
                    [
                        _Item(
                            f"{prefix}.feedforward_constant_layer",
                            emulator.feedforward_constant_layer,
                        )
                    ],
                    (input_key,),
                )
                input_dependencies[coordinate].append(key)
                polynomial_outputs[(coordinate, degree)] = output
                continue

            if not hasattr(emulator, "feedforward_factor_layer") or not hasattr(
                emulator, "feedforward_output_layer"
            ):
                raise ValueError("model must first be converted with to_feedforward")

            factor_key = add_block(
                f"u.{coordinate}.{degree}.factors",
                f"{label_prefix} roots",
                "root",
                [
                    _Item(
                        f"{prefix}.feedforward_factor_layer",
                        emulator.feedforward_factor_layer,
                    )
                ],
                (input_key,),
            )
            input_dependencies[coordinate].append(factor_key)

            if degree == 1:
                product_output = factor_key
            elif emulator.product == "tanh":
                if emulator._tanh_root is None:
                    raise RuntimeError("tanh product tree is missing")
                visited: dict[int, str] = {}

                def add_tanh_node(
                    node,
                    *,
                    factor_key=factor_key,
                    visited=visited,
                    emulator=emulator,
                    coordinate=coordinate,
                    degree=degree,
                    label_prefix=label_prefix,
                    prefix=prefix,
                ) -> str:
                    if node.left is None or node.right is None:
                        return factor_key
                    assert node.module_index is not None
                    if node.module_index in visited:
                        return visited[node.module_index]
                    left = add_tanh_node(node.left)
                    right = add_tanh_node(node.right)
                    module_index = node.module_index
                    product = emulator.product_nets[module_index]
                    key = add_block(
                        f"u.{coordinate}.{degree}.product.{module_index}",
                        f"{label_prefix} multiply {module_index}",
                        "univariate_multiplier",
                        _product_items(
                            product, f"{prefix}.product_nets.{module_index}"
                        ),
                        (left, right),
                    )
                    visited[module_index] = key
                    return key

                product_output = add_tanh_node(emulator._tanh_root)
            else:
                previous = factor_key
                for product_index, product in enumerate(emulator.product_nets):
                    previous = add_block(
                        f"u.{coordinate}.{degree}.product.{product_index}",
                        f"{label_prefix} multiply {product_index}",
                        "univariate_multiplier",
                        _product_items(
                            product, f"{prefix}.product_nets.{product_index}"
                        ),
                        (previous, factor_key),
                    )
                product_output = previous

            output = add_block(
                f"u.{coordinate}.{degree}.normalization",
                f"{label_prefix} normalization",
                "normalization",
                [
                    _Item(
                        f"{prefix}.feedforward_output_layer",
                        emulator.feedforward_output_layer,
                    )
                ],
                (product_output,),
            )
            polynomial_outputs[(coordinate, degree)] = output

    univariate_end = max(block.end for block in blocks)
    cross_minimum_depth = univariate_end + 1.5
    outputs: list[_Output] = []
    indices = model.multi_index.detach().cpu().numpy()
    for feature, (alpha_array, active, modules) in enumerate(
        zip(indices, model.active_coordinates, model.cross_product_nets, strict=True)
    ):
        alpha = tuple(int(value) for value in alpha_array)
        if not active:
            outputs.append(_Output(feature, alpha, None))
            continue

        previous = polynomial_outputs[(active[0], alpha[active[0]])]
        for product_index, (product, coordinate) in enumerate(
            zip(modules, active[1:], strict=True)
        ):
            coordinate_input = polynomial_outputs[(coordinate, alpha[coordinate])]
            prefix = f"cross_product_nets.{feature}.{product_index}"
            previous = add_block(
                f"m.{feature}.product.{product_index}",
                f"alpha={alpha} cross {product_index}",
                "cross_multiplier",
                _product_items(product, prefix),
                (previous, coordinate_input),
                minimum_depth=cross_minimum_depth,
            )
        outputs.append(_Output(feature, alpha, previous))

    layout = _NetworkLayout(
        blocks=blocks,
        input_dependencies={
            coordinate: tuple(dependencies)
            for coordinate, dependencies in input_dependencies.items()
        },
        outputs=outputs,
    )

    displayed_linears = {
        item.name
        for block in blocks
        for item in block.items
        if isinstance(item.module, nn.Linear)
    }
    model_linears = {
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    if displayed_linears != model_linears:
        missing = sorted(model_linears - displayed_linears)
        extra = sorted(displayed_linears - model_linears)
        raise RuntimeError(
            "whole-network layout does not match the model's affine layers: "
            f"missing={missing}, extra={extra}"
        )
    return layout


_INSPECTION_COLORSCALE = [
    [0.00, "#33206f"],
    [0.18, "#5558c9"],
    [0.36, "#38b8dc"],
    [0.50, "#100c2e"],
    [0.64, "#62d7a1"],
    [0.82, "#c7df55"],
    [1.00, "#ee9732"],
]

_KIND_LABELS = {
    "root": "root shifts and constants",
    "univariate_multiplier": "univariate multiplier layers",
    "normalization": "polynomial normalizations",
    "cross_multiplier": "cross-coordinate multiplier layers",
}


def _signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log10(1.0 + np.abs(values))


def interactive_network_figure(
    model: MultivariateBasisEmulator,
    *,
    title: str | None = None,
    show_wiring: bool = True,
    show_axes: bool = False,
    show_exact_zeros: bool = False,
    max_coefficients: int = 300_000,
) -> Any:
    """Return an interactive Plotly X-ray of a complete converted emulator.

    Every materialized affine layer is drawn exactly once. Matrix entries are
    placed on translucent layer planes; the final matrix column contains the
    bias. The graph also includes elementwise activations, coordinate inputs,
    product-tree wiring, cross-coordinate products, and all basis outputs.

    Parameters
    ----------
    model:
        A multivariate basis emulator returned by ``to_feedforward``.
    title:
        Optional figure title.
    show_wiring:
        Display dependency edges between computational blocks.
    show_axes:
        Display the Plotly scene axes, grid, and bounding panes. The default
        leaves only the network geometry visible.
    show_exact_zeros:
        Display zero-valued matrix entries. They can also be toggled from the
        legend after construction.
    max_coefficients:
        Refuse larger figures before constructing their browser payload.
    """
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "interactive_network_figure requires Plotly; install the notebook "
            "extras with `python -m pip install -e '.[notebook]'`"
        ) from error

    max_coefficients = int(max_coefficients)
    if max_coefficients <= 0:
        raise ValueError("max_coefficients must be positive")
    layout = _whole_network_layout(model)
    affine_items = [
        (block, position, item)
        for block in layout.blocks
        for position, item in zip(block.positions, block.items, strict=True)
        if isinstance(item.module, nn.Linear)
    ]
    coefficient_count = sum(
        item.module.weight.numel() + item.module.bias.numel()
        for _, _, item in affine_items
    )
    if coefficient_count > max_coefficients:
        raise ValueError(
            f"the figure contains {coefficient_count:,} affine coefficients, "
            f"exceeding max_coefficients={max_coefficients:,}"
        )

    max_columns = max(item.module.in_features + 1 for _, _, item in affine_items)
    max_rows = max(item.module.out_features for _, _, item in affine_items)
    cell_spacing = 0.52
    lane_spacing = (max_columns + 3.0) * cell_spacing
    lane_centers = {block.key: block.lane * lane_spacing for block in layout.blocks}
    start_by_key = {block.key: block.start for block in layout.blocks}
    end_by_key = {block.key: block.end for block in layout.blocks}
    output_depth = max(end_by_key.values()) + 2.2

    all_values = np.concatenate(
        [
            np.column_stack(
                (
                    item.module.weight.detach().cpu().numpy(),
                    item.module.bias.detach().cpu().numpy(),
                )
            ).ravel()
            for _, _, item in affine_items
        ]
    )
    transformed_values = _signed_log(all_values)
    color_limit = max(float(np.max(np.abs(transformed_values))), 1e-15)

    points: dict[str, dict[str, list]] = {
        kind: {"x": [], "y": [], "z": [], "color": [], "size": [], "text": []}
        for kind in _KIND_LABELS
    }
    zeros = {"x": [], "y": [], "z": []}
    border_x: list[float | None] = []
    border_y: list[float | None] = []
    border_z: list[float | None] = []
    bias_x: list[float | None] = []
    bias_y: list[float | None] = []
    bias_z: list[float | None] = []

    for block, depth, item in affine_items:
        layer = item.module
        matrix = np.column_stack(
            (layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy())
        )
        transformed = _signed_log(matrix)
        rows, columns = matrix.shape
        center = lane_centers[block.key]
        column_centers = (
            center + (np.arange(columns) - 0.5 * (columns - 1)) * cell_spacing
        )
        row_centers = (np.arange(rows) - 0.5 * (rows - 1)) * cell_spacing
        for row in range(rows):
            for column in range(columns):
                value = float(matrix[row, column])
                is_bias = column == layer.in_features
                location = "bias" if is_bias else f"weight column {column}"
                text = (
                    f"<b>{item.name}</b><br>"
                    f"{block.label}<br>"
                    f"shape: {layer.out_features} x {layer.in_features} + bias<br>"
                    f"row {row}, {location}<br>value: {value:.8g}<br>"
                    f"signed log: {float(transformed[row, column]):.6g}"
                )
                if value == 0.0:
                    target = zeros
                else:
                    target = points[block.kind]
                    target["color"].append(float(transformed[row, column]))
                    relative = abs(float(transformed[row, column])) / color_limit
                    target["size"].append(3.2 + 3.8 * relative)
                target["x"].append(depth)
                target["y"].append(float(column_centers[column]))
                target["z"].append(float(row_centers[row]))
                if value != 0.0:
                    target["text"].append(text)

        y_min = center - 0.5 * columns * cell_spacing
        y_max = center + 0.5 * columns * cell_spacing
        z_min = -0.5 * rows * cell_spacing
        z_max = 0.5 * rows * cell_spacing
        border_x.extend([depth] * 5 + [None])
        border_y.extend([y_min, y_max, y_max, y_min, y_min, None])
        border_z.extend([z_min, z_min, z_max, z_max, z_min, None])
        boundary = (
            center + (layer.in_features - 0.5 - 0.5 * (columns - 1)) * cell_spacing
        )
        bias_x.extend([depth, depth, None])
        bias_y.extend([boundary, boundary, None])
        bias_z.extend([z_min, z_max, None])

    figure = go.Figure()
    colorbar_shown = False
    for kind, label in _KIND_LABELS.items():
        data = points[kind]
        if not data["x"]:
            continue
        figure.add_trace(
            go.Scatter3d(
                x=data["x"],
                y=data["y"],
                z=data["z"],
                mode="markers",
                name=label,
                legendgroup=kind,
                marker={
                    "symbol": "square",
                    "size": data["size"],
                    "color": data["color"],
                    "colorscale": _INSPECTION_COLORSCALE,
                    "cmin": -color_limit,
                    "cmax": color_limit,
                    "opacity": 0.88,
                    "showscale": not colorbar_shown,
                    "colorbar": {
                        "title": "sign(theta)<br>log10(1+|theta|)",
                        "thickness": 15,
                        "len": 0.38,
                        "x": 0.84,
                        "y": 0.23,
                    },
                },
                text=data["text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        colorbar_shown = True

    figure.add_trace(
        go.Scatter3d(
            x=zeros["x"],
            y=zeros["y"],
            z=zeros["z"],
            mode="markers",
            name="exactly zero coefficients",
            legendgroup="zeros",
            visible=True if show_exact_zeros else "legendonly",
            marker={
                "symbol": "square",
                "size": 2.2,
                "color": "#24394a",
                "opacity": 0.32,
            },
            hovertemplate="exactly zero<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=border_x,
            y=border_y,
            z=border_z,
            mode="lines",
            name="affine layer boundaries",
            legendgroup="boundaries",
            line={"color": "#72a9bf", "width": 2},
            opacity=0.5,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=bias_x,
            y=bias_y,
            z=bias_z,
            mode="lines",
            name="bias boundaries",
            legendgroup="boundaries",
            showlegend=False,
            line={"color": "#f1f7f9", "width": 3},
            opacity=0.7,
            hoverinfo="skip",
        )
    )

    activation_x: list[float] = []
    activation_y: list[float] = []
    activation_z: list[float] = []
    activation_text: list[str] = []
    for block in layout.blocks:
        for depth, item in zip(block.positions, block.items, strict=True):
            if isinstance(item.module, nn.Linear):
                continue
            activation_x.append(depth)
            activation_y.append(lane_centers[block.key])
            activation_z.append(0.0)
            activation_text.append(
                f"<b>{item.name}</b><br>{block.label}<br>"
                f"operator: {type(item.module).__name__}"
            )
    figure.add_trace(
        go.Scatter3d(
            x=activation_x,
            y=activation_y,
            z=activation_z,
            mode="markers",
            name="elementwise activations",
            legendgroup="activations",
            marker={
                "symbol": "diamond",
                "size": 5,
                "color": "#c7df55",
                "line": {"color": "#071018", "width": 1},
            },
            text=activation_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    dependency_x: list[float | None] = []
    dependency_y: list[float | None] = []
    dependency_z: list[float | None] = []
    for block in layout.blocks:
        center = lane_centers[block.key]
        for left, right in zip(block.positions[:-1], block.positions[1:]):
            dependency_x.extend([left, right, None])
            dependency_y.extend([center, center, None])
            dependency_z.extend([0.0, 0.0, None])
        for dependency in block.dependencies:
            if dependency.startswith("input."):
                continue
            dependency_x.extend([end_by_key[dependency], block.start, None])
            dependency_y.extend(
                [lane_centers[dependency], lane_centers[block.key], None]
            )
            dependency_z.extend([0.0, 0.0, None])

    input_x: list[float] = []
    input_y: list[float] = []
    input_text: list[str] = []
    for coordinate, dependencies in layout.input_dependencies.items():
        centers = [lane_centers[key] for key in dependencies]
        center = float(np.mean(centers)) if centers else 0.0
        input_x.append(0.0)
        input_y.append(center)
        input_text.append(f"input coordinate x_{coordinate}")
        for dependency in dependencies:
            dependency_x.extend([0.0, start_by_key[dependency], None])
            dependency_y.extend([center, lane_centers[dependency], None])
            dependency_z.extend([0.0, 0.0, None])

    output_x: list[float] = []
    output_y: list[float] = []
    output_text: list[str] = []
    if layout.outputs:
        y_min = min(lane_centers.values())
        y_max = max(lane_centers.values())
        target_centers = np.linspace(y_min, y_max, len(layout.outputs))
    else:
        target_centers = np.empty(0)
    for output, center in zip(layout.outputs, target_centers, strict=True):
        output_x.append(output_depth)
        output_y.append(float(center))
        output_text.append(
            f"basis output {output.feature}<br>multi-index alpha={output.alpha}"
        )
        if output.dependency is not None:
            dependency_x.extend([end_by_key[output.dependency], output_depth, None])
            dependency_y.extend([lane_centers[output.dependency], float(center), None])
            dependency_z.extend([0.0, 0.0, None])

    figure.add_trace(
        go.Scatter3d(
            x=dependency_x,
            y=dependency_y,
            z=dependency_z,
            mode="lines",
            name="computational wiring",
            legendgroup="wiring",
            visible=True if show_wiring else "legendonly",
            line={"color": "#38b8dc", "width": 2},
            opacity=0.22,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=input_x,
            y=input_y,
            z=[0.0] * len(input_x),
            mode="markers",
            name="coordinate inputs",
            legendgroup="io",
            marker={"symbol": "circle", "size": 7, "color": "#38b8dc"},
            text=input_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=output_x,
            y=output_y,
            z=[0.0] * len(output_x),
            mode="markers",
            name="basis outputs",
            legendgroup="io",
            marker={"symbol": "circle", "size": 7, "color": "#ee9732"},
            text=output_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    tick_stride = max(1, len(layout.blocks) // 55)
    tick_blocks = layout.blocks[::tick_stride]
    maximum_depth = max(output_depth, 1.0)
    figure.update_layout(
        title={
            "text": title
            or (
                f"{model.product}: complete {model.family} feedforward emulator "
                f"(d={model.dimension}, |Lambda|={model.n_features})"
            ),
            "x": 0.5,
        },
        width=1250,
        height=max(760, min(1400, 610 + 5 * len(layout.blocks))),
        paper_bgcolor="#071018",
        plot_bgcolor="#071018",
        font={"color": "#dbe9f4"},
        dragmode="orbit",
        hovermode="closest",
        uirevision="whole-network-xray",
        legend={
            "bgcolor": "rgba(7,16,24,0.72)",
            "bordercolor": "#527487",
            "borderwidth": 1,
            "groupclick": "toggleitem",
            "x": 0.80,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
        margin={"l": 20, "r": 20, "b": 20, "t": 80},
        scene={
            "bgcolor": "#071018",
            "domain": {"x": [0.0, 0.76], "y": [0.0, 1.0]},
            "xaxis": {
                "title": "computational depth" if show_axes else "",
                "visible": show_axes,
                "showbackground": False,
                "showgrid": False,
                "zeroline": False,
                "showline": False,
                "showspikes": False,
            },
            "yaxis": {
                "title": "computational branch" if show_axes else "",
                "visible": show_axes,
                "showbackground": False,
                "showgrid": False,
                "zeroline": False,
                "showline": False,
                "showspikes": False,
                "tickmode": "array",
                "tickvals": [lane_centers[block.key] for block in tick_blocks],
                "ticktext": [block.label for block in tick_blocks],
                "tickfont": {"size": 8},
            },
            "zaxis": {
                "title": "affine output row" if show_axes else "",
                "visible": show_axes,
                "showbackground": False,
                "showgrid": False,
                "zeroline": False,
                "showline": False,
                "showspikes": False,
            },
            "camera": {"eye": {"x": 1.55, "y": -1.85, "z": 1.15}},
            "aspectmode": "manual",
            "aspectratio": {
                "x": min(4.8, max(1.5, maximum_depth / 15.0)),
                "y": min(5.5, max(1.7, len(layout.blocks) / 12.0)),
                "z": min(2.0, max(0.8, max_rows / 9.0)),
            },
        },
        annotations=[
            {
                "text": (
                    f"{len(affine_items):,} affine layers; "
                    f"{coefficient_count:,} weights and biases. "
                    "Drag to rotate; scroll to zoom; click legend entries to filter."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 1.02,
                "showarrow": False,
                "font": {"size": 11, "color": "#9dc8dc"},
            }
        ],
    )
    return figure
