import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphUNet
from torch_geometric.nn import SAGEConv
import inspect
from typing import Any, Dict, Optional, Sequence


PREDICT_TYPE_STATE = "state"
PREDICT_TYPE_DELTA = "delta"
PREDICT_TYPE_RATE = "rate"
PREDICT_TYPES = {PREDICT_TYPE_STATE, PREDICT_TYPE_DELTA, PREDICT_TYPE_RATE}


def _normalize_predict_type(predict_type: Optional[str], default: str = PREDICT_TYPE_STATE) -> str:
    key = str(default if predict_type is None else predict_type).strip().lower().replace("-", "_")
    aliases = {
        "absolute": PREDICT_TYPE_STATE,
        "next": PREDICT_TYPE_STATE,
        "next_state": PREDICT_TYPE_STATE,
        "state": PREDICT_TYPE_STATE,
        "delta": PREDICT_TYPE_DELTA,
        "residual": PREDICT_TYPE_DELTA,
        "rate": PREDICT_TYPE_RATE,
        "derivative": PREDICT_TYPE_RATE,
    }
    key = aliases.get(key, key)
    if key not in PREDICT_TYPES:
        raise ValueError(f"predict_type must be one of {sorted(PREDICT_TYPES)}, got {predict_type!r}.")
    return key


class FeatureNet(nn.Module):
    """
    GraphSAGE encoder with a regression head that predicts features at dynamic cell centers.
    Optionally includes a refine-score head (for diagnostics), but geometry CE is *not* trained.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
        make_score_head: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        dims = [in_channels] + [hidden] * (layers - 1)
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.feat_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels)
        )
        self.score_head = None
        if make_score_head:
            self.score_head = nn.Sequential(
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden//2, 1)  # refine score (logit)
            )

    def forward(self, X, edge_index):
        h = X
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        y_feat = self.feat_head(h)
        y_score = self.score_head(h) if self.score_head is not None else None
        return y_feat, y_score, h


def _make_activation(
    name: str,
    *,
    negative_slope: float = 0.01,
    elu_alpha: float = 1.0,
) -> nn.Module:
    key = str(name).strip().lower()
    if key in {"relu"}:
        return nn.ReLU()
    if key in {"leaky_relu", "leakyrelu", "lrelu", "leaky"}:
        return nn.LeakyReLU(negative_slope=float(negative_slope))
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key in {"gelu"}:
        return nn.GELU()
    if key in {"elu"}:
        return nn.ELU(alpha=float(elu_alpha))
    if key in {"none", "identity", "linear"}:
        return nn.Identity()
    raise ValueError(
        f"Unsupported activation '{name}'. "
        "Use one of: relu, leaky_relu, silu, gelu, elu, identity."
    )


def _activation(name: str) -> nn.Module:
    """Back-compat wrapper for older local code paths."""
    return _make_activation(name)


class MeshGraphMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        hidden_layers: int = 1,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        dropout: float = 0.0,
        layer_norm: bool = False,
        layernorm_eps: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = int(out_dim if hidden_dim is None else hidden_dim)
        hidden_layers = max(1, int(hidden_layers))

        layers = []
        prev = int(in_dim)
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(
                _make_activation(
                    activation,
                    negative_slope=float(activation_negative_slope),
                    elu_alpha=float(activation_elu_alpha),
                )
            )
            layers.append(nn.Dropout(p=float(dropout)))
            prev = hidden_dim
        layers.append(nn.Linear(prev, int(out_dim)))
        if bool(layer_norm):
            layers.append(nn.LayerNorm(int(out_dim), eps=float(layernorm_eps)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeshGraphNetBlock(nn.Module):
    """
    Residual MeshGraphNet processor block with edge update, message aggregation,
    and node update. Uses pure torch index_add for portability.
    """

    def __init__(
        self,
        hidden: int,
        *,
        mlp_hidden_layers: int = 1,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        dropout: float = 0.0,
        aggregation: str = "sum",
    ):
        super().__init__()
        hidden = int(hidden)
        self.edge_mlp = MeshGraphMLP(
            3 * hidden,
            hidden,
            hidden_dim=hidden,
            hidden_layers=mlp_hidden_layers,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            dropout=dropout,
        )
        self.node_mlp = MeshGraphMLP(
            2 * hidden,
            hidden,
            hidden_dim=hidden,
            hidden_layers=mlp_hidden_layers,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            dropout=dropout,
        )
        agg = str(aggregation).strip().lower()
        if agg != "sum":
            raise ValueError("MeshGraphNet aggregation must be 'sum' to match the 1D/Burgers implementation.")
        self.aggregation = agg
        if bool(use_layernorm):
            self.edge_norm = nn.LayerNorm(hidden, eps=float(layernorm_eps))
            self.node_norm = nn.LayerNorm(hidden, eps=float(layernorm_eps))
        else:
            self.edge_norm = nn.Identity()
            self.node_norm = nn.Identity()

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[0].long()
        dst = edge_index[1].long()

        edge_in = torch.cat([node_h[src], node_h[dst], edge_h], dim=-1)
        edge_h = edge_h + self.edge_mlp(edge_in)
        edge_h = self.edge_norm(edge_h)

        agg = torch.zeros_like(node_h)
        agg.index_add_(0, dst, edge_h)

        node_in = torch.cat([node_h, agg], dim=-1)
        node_h = node_h + self.node_mlp(node_in)
        node_h = self.node_norm(node_h)
        return node_h, edge_h


class MeshGraphNet(nn.Module):
    """
    Encoder-processor-decoder MeshGraphNet.

    Node inputs are the same node feature tensor used by FeatureNet. Edge inputs
    are built on the fly from relative mesh displacement [dx, dy, distance] by
    passing pos=... to forward.
    """

    uses_edge_geometry = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        hidden: int = 128,
        processor_steps: Optional[int] = None,
        layers: Optional[int] = None,
        edge_attr_channels: Optional[int] = None,
        edge_in_channels: Optional[int] = None,
        edge_pos_dim: int = 2,
        mlp_hidden_layers: int = 1,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        layer_norm: Optional[bool] = None,
        decoder_layer_norm: bool = False,
        dropout: float = 0.0,
        aggregation: str = "sum",
        use_skip: bool = False,
        make_score_head: bool = False,
    ):
        super().__init__()
        hidden = int(hidden)
        del use_skip
        if layers is None:
            layers = processor_steps if processor_steps is not None else 3
        layers = max(1, int(layers))
        edge_pos_dim = max(1, int(edge_pos_dim))
        if edge_attr_channels is None:
            edge_attr_channels = edge_in_channels
        if edge_attr_channels is None:
            edge_attr_channels = edge_pos_dim + 1
        if layer_norm is not None:
            use_layernorm = bool(layer_norm)

        if int(in_channels) <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if int(out_channels) <= 0:
            raise ValueError(f"out_channels must be > 0, got {out_channels}.")
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if int(edge_attr_channels) <= 0:
            raise ValueError(f"edge_attr_channels must be > 0, got {edge_attr_channels}.")
        self.hidden = hidden
        self.layers = layers
        self.edge_pos_dim = edge_pos_dim
        self.edge_attr_channels = int(edge_attr_channels)
        self.edge_in_channels = int(edge_attr_channels)
        self.dropout = float(dropout)
        self.block_activation = _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        )

        self.node_encoder = MeshGraphMLP(
            int(in_channels),
            hidden,
            hidden_dim=hidden,
            hidden_layers=mlp_hidden_layers,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            dropout=dropout,
        )
        self.edge_encoder = MeshGraphMLP(
            self.edge_in_channels,
            hidden,
            hidden_dim=hidden,
            hidden_layers=mlp_hidden_layers,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            dropout=dropout,
        )
        self.processor = nn.ModuleList(
            [
                MeshGraphNetBlock(
                    hidden,
                    mlp_hidden_layers=mlp_hidden_layers,
                    activation=activation,
                    activation_negative_slope=activation_negative_slope,
                    activation_elu_alpha=activation_elu_alpha,
                    use_layernorm=bool(use_layernorm),
                    layernorm_eps=float(layernorm_eps),
                    dropout=dropout,
                    aggregation=aggregation,
                )
                for _ in range(layers)
            ]
        )
        self.decoder = MeshGraphMLP(
            hidden,
            int(out_channels),
            hidden_dim=hidden,
            hidden_layers=mlp_hidden_layers,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            dropout=dropout,
            layer_norm=decoder_layer_norm,
            layernorm_eps=layernorm_eps,
        )
        self.score_head = None
        if bool(make_score_head):
            self.score_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )

    def _build_edge_features(
        self,
        edge_index: torch.Tensor,
        pos: Optional[torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if pos is None:
            raise ValueError("MeshGraphNet.forward requires pos=... so edge-relative features can be built.")
        if pos.ndim != 2 or int(pos.size(1)) < self.edge_pos_dim:
            raise ValueError(
                f"MeshGraphNet expected pos shape [N,>={self.edge_pos_dim}], got {tuple(pos.shape)}"
            )
        src = edge_index[0].long()
        dst = edge_index[1].long()
        p = pos[:, : self.edge_pos_dim].to(device=device, dtype=dtype)
        rel = p[dst] - p[src]
        dist = torch.linalg.norm(rel, dim=1, keepdim=True)
        edge_feat = torch.cat([rel, dist], dim=-1)
        if edge_feat.size(1) < self.edge_in_channels:
            pad = torch.zeros(
                (edge_feat.size(0), self.edge_in_channels - edge_feat.size(1)),
                device=device,
                dtype=dtype,
            )
            edge_feat = torch.cat([edge_feat, pad], dim=-1)
        elif edge_feat.size(1) > self.edge_in_channels:
            edge_feat = edge_feat[:, : self.edge_in_channels]
        return edge_feat

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ):
        edge_index = edge_index.to(device=X.device, dtype=torch.long)
        edge_feat = self._build_edge_features(
            edge_index,
            pos,
            dtype=X.dtype,
            device=X.device,
        )
        node_h = self.node_encoder(X)
        edge_h = self.edge_encoder(edge_feat)
        for block in self.processor:
            node_h, edge_h = block(node_h, edge_h, edge_index)
            node_h = self.block_activation(node_h)
            node_h = F.dropout(node_h, p=self.dropout, training=self.training)
        y_feat = self.decoder(node_h)
        y_score = self.score_head(node_h) if self.score_head is not None else None
        return y_feat, y_score, node_h


class FluxGraphNet(nn.Module):
    """
    FluxGraphNet-style model for Karman point-graph velocity prediction.

    The processor is MeshGraphNet-like, but the decoder predicts one flux per
    unique undirected graph edge. That flux is applied with opposite signs to
    the two edge endpoints, so interior/cylinder nodes receive conservative
    pairwise updates. Exterior domain-boundary nodes can additionally receive
    non-conservative open-boundary source terms.
    """

    uses_edge_geometry = True
    uses_dual_volume = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        hidden: int = 128,
        layers: Optional[int] = None,
        processor_steps: Optional[int] = None,
        state_channel: int = 0,
        predict_type: Optional[str] = None,
        edge_attr_channels: Optional[int] = None,
        edge_in_channels: Optional[int] = None,
        edge_pos_dim: int = 2,
        mlp_hidden_layers: int = 1,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        layer_norm: Optional[bool] = None,
        dropout: float = 0.0,
        aggregation: str = "sum",
        flux_scale: float = 1.0,
        use_dual_volume: bool = True,
        volume_floor: float = 1e-12,
        use_open_boundary_source: bool = True,
        open_boundary_mode: str = "learned_source",
        open_boundary_modes_by_side: Optional[Dict[str, str]] = None,
        open_boundary_source_channels: Optional[Sequence[int]] = None,
        open_boundary_flux_sides: Optional[Sequence[str] | str] = None,
        open_boundary_flux_scale: float = 0.05,
        open_boundary_flux_outflow_only: bool = True,
        boundary_width: float = 0.02,
        domain_bbox: Optional[Sequence[float] | str] = None,
        cylinder_center_xy: Optional[Sequence[float]] = None,
        cylinder_radius: float = 0.5,
        cylinder_boundary_width: float = 0.08,
        velocity_channels: Optional[Sequence[int]] = None,
        make_score_head: bool = False,
    ):
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if int(out_channels) <= 0:
            raise ValueError(f"out_channels must be > 0, got {out_channels}.")
        if int(state_channel) < 0 or int(state_channel) + int(out_channels) > int(in_channels):
            raise ValueError(
                "state_channel/out_channels must select a valid state slice from the model input; "
                f"got state_channel={state_channel}, out_channels={out_channels}, in_channels={in_channels}."
            )
        hidden = int(hidden)
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers is None:
            layers = processor_steps if processor_steps is not None else 3
        layers = max(1, int(layers))
        edge_pos_dim = max(1, int(edge_pos_dim))
        if edge_attr_channels is None:
            edge_attr_channels = edge_in_channels
        if edge_attr_channels is None:
            edge_attr_channels = (2 * edge_pos_dim) + 1
        if int(edge_attr_channels) <= 0:
            raise ValueError(f"edge_attr_channels must be > 0, got {edge_attr_channels}.")
        if layer_norm is not None:
            use_layernorm = bool(layer_norm)
        agg = str(aggregation).strip().lower()
        if agg != "sum":
            raise ValueError("FluxGraphNet aggregation must be 'sum' so edge fluxes remain additive.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden = hidden
        self.layers = layers
        self.state_channel = int(state_channel)
        self.predict_type = _normalize_predict_type(predict_type, default=PREDICT_TYPE_STATE)
        self.edge_pos_dim = edge_pos_dim
        self.edge_attr_channels = int(edge_attr_channels)
        self.edge_in_channels = int(edge_attr_channels)
        self.dropout = float(dropout)
        self.flux_scale = float(flux_scale)
        self.use_dual_volume = bool(use_dual_volume)
        self.volume_floor = float(volume_floor)
        self.use_open_boundary_source = bool(use_open_boundary_source)
        self.open_boundary_mode = self._normalize_open_boundary_mode(open_boundary_mode)
        self.open_boundary_flux_scale = float(open_boundary_flux_scale)
        if self.open_boundary_flux_scale < 0.0:
            raise ValueError("open_boundary_flux_scale must be >= 0.")
        self.open_boundary_flux_outflow_only = bool(open_boundary_flux_outflow_only)
        self.boundary_width = float(boundary_width)
        self.cylinder_radius = float(cylinder_radius)
        self.cylinder_boundary_width = float(cylinder_boundary_width)

        if domain_bbox is None or str(domain_bbox).strip().lower() == "auto":
            self.domain_bbox = None
        else:
            if not isinstance(domain_bbox, (list, tuple)) or len(domain_bbox) != 4:
                raise ValueError("FluxGraphNet domain_bbox must be null/'auto' or [xmin, xmax, ymin, ymax].")
            self.domain_bbox = tuple(float(v) for v in domain_bbox)

        if cylinder_center_xy is None:
            cylinder_center_xy = (0.0, 0.0)
        if len(cylinder_center_xy) != 2:
            raise ValueError("cylinder_center_xy must contain two values [cx, cy].")
        self.register_buffer(
            "cylinder_center_xy",
            torch.tensor([float(cylinder_center_xy[0]), float(cylinder_center_xy[1])], dtype=torch.float32),
            persistent=False,
        )

        learned_side_mask, flux_mode_side_mask = self._make_open_boundary_mode_side_masks(
            self.open_boundary_mode,
            open_boundary_modes_by_side,
        )
        self.register_buffer("open_boundary_learned_side_mask", learned_side_mask, persistent=False)
        self.register_buffer("open_boundary_flux_mode_side_mask", flux_mode_side_mask, persistent=False)
        self.register_buffer(
            "open_boundary_flux_side_mask",
            self._make_boundary_side_mask(open_boundary_flux_sides),
            persistent=False,
        )
        self.register_buffer(
            "open_boundary_source_mask",
            self._make_state_channel_mask(open_boundary_source_channels, self.out_channels),
            persistent=False,
        )

        if velocity_channels is None:
            velocity_channels = (0, 1) if self.out_channels >= 2 else (0, 0)
        if len(velocity_channels) < 2:
            raise ValueError("velocity_channels must provide two state-relative channel indices.")
        vel0 = int(velocity_channels[0])
        vel1 = int(velocity_channels[1])
        if vel0 < 0 or vel0 >= self.out_channels or vel1 < 0 or vel1 >= self.out_channels:
            raise ValueError(
                f"velocity_channels must index the {self.out_channels} predicted channels, got {velocity_channels}."
            )
        self.velocity_channels = (vel0, vel1)

        self.block_activation = _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        )
        self.node_encoder = MeshGraphMLP(
            self.in_channels,
            hidden,
            hidden_dim=hidden,
            hidden_layers=mlp_hidden_layers,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            dropout=dropout,
        )
        self.edge_encoder = MeshGraphMLP(
            self.edge_in_channels,
            hidden,
            hidden_dim=hidden,
            hidden_layers=mlp_hidden_layers,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            dropout=dropout,
        )
        self.processor = nn.ModuleList(
            [
                MeshGraphNetBlock(
                    hidden,
                    mlp_hidden_layers=mlp_hidden_layers,
                    activation=activation,
                    activation_negative_slope=activation_negative_slope,
                    activation_elu_alpha=activation_elu_alpha,
                    use_layernorm=bool(use_layernorm),
                    layernorm_eps=float(layernorm_eps),
                    dropout=dropout,
                    aggregation=aggregation,
                )
                for _ in range(layers)
            ]
        )
        self.flux_head = MeshGraphMLP(
            (2 * hidden) + self.edge_in_channels,
            self.out_channels,
            hidden_dim=hidden,
            hidden_layers=mlp_hidden_layers,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            dropout=dropout,
        )
        self.open_boundary_head = (
            MeshGraphMLP(
                hidden,
                self.out_channels,
                hidden_dim=hidden,
                hidden_layers=mlp_hidden_layers,
                activation=activation,
                activation_negative_slope=activation_negative_slope,
                activation_elu_alpha=activation_elu_alpha,
                dropout=dropout,
            )
            if (
                self.use_open_boundary_source
                and bool(torch.any(self.open_boundary_learned_side_mask > 0).item())
            )
            else None
        )
        self.score_head = None
        if bool(make_score_head):
            self.score_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )

    @staticmethod
    def _normalize_open_boundary_mode(mode: Optional[str]) -> str:
        key = str("learned_source" if mode is None else mode).strip().lower().replace("-", "_")
        aliases = {
            "learned": "learned_source",
            "source": "learned_source",
            "learned_source": "learned_source",
            "boundary_source": "learned_source",
            "mlp": "learned_source",
            "flux": "boundary_flux",
            "boundary_flux": "boundary_flux",
            "outflow": "boundary_flux",
            "outflow_flux": "boundary_flux",
            "open_flux": "boundary_flux",
            "none": "none",
            "off": "none",
            "disabled": "none",
            "false": "none",
        }
        if key not in aliases:
            raise ValueError(
                "open_boundary_mode must be one of {learned_source, boundary_flux, none}; "
                f"got {mode!r}."
            )
        return aliases[key]

    @staticmethod
    def _boundary_side_index(side: str) -> int:
        aliases = {
            "left": 0,
            "l": 0,
            "xmin": 0,
            "x_min": 0,
            "right": 1,
            "r": 1,
            "xmax": 1,
            "x_max": 1,
            "bottom": 2,
            "b": 2,
            "ymin": 2,
            "y_min": 2,
            "top": 3,
            "t": 3,
            "ymax": 3,
            "y_max": 3,
        }
        key = str(side).strip().lower().replace("-", "_")
        if key not in aliases:
            raise ValueError(
                "Boundary side entries must be drawn from {left, right, bottom, top}; "
                f"got {side!r}."
            )
        return aliases[key]

    @classmethod
    def _make_open_boundary_mode_side_masks(
        cls,
        global_mode: str,
        modes_by_side: Optional[Dict[str, str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        side_modes = [cls._normalize_open_boundary_mode(global_mode)] * 4
        if modes_by_side is not None:
            if not isinstance(modes_by_side, dict):
                raise ValueError(
                    "open_boundary_modes_by_side must map {left,right,bottom,top} "
                    "to {learned_source,boundary_flux,none}."
                )
            for side, mode in modes_by_side.items():
                mode_key = cls._normalize_open_boundary_mode(mode)
                side_key = str(side).strip().lower().replace("-", "_")
                if side_key in {"all", "*"}:
                    side_modes = [mode_key] * 4
                else:
                    side_modes[cls._boundary_side_index(side_key)] = mode_key

        learned = torch.zeros(4, dtype=torch.float32)
        flux = torch.zeros(4, dtype=torch.float32)
        for i, mode in enumerate(side_modes):
            if mode == "learned_source":
                learned[i] = 1.0
            elif mode == "boundary_flux":
                flux[i] = 1.0
        return learned, flux

    @classmethod
    def _make_boundary_side_mask(cls, sides: Optional[Sequence[str] | str]) -> torch.Tensor:
        mask = torch.ones(4, dtype=torch.float32)
        if sides is None:
            return mask
        if isinstance(sides, str):
            items = [s.strip() for s in sides.replace(";", ",").split(",") if s.strip()]
        else:
            items = [str(s).strip() for s in sides]
        if not items or any(str(s).strip().lower() in {"all", "*"} for s in items):
            return mask
        mask.zero_()
        for side in items:
            mask[cls._boundary_side_index(side)] = 1.0
        return mask

    @staticmethod
    def _make_state_channel_mask(channels: Optional[Sequence[int]], out_channels: int) -> torch.Tensor:
        mask = torch.ones(int(out_channels), dtype=torch.float32)
        if channels is None:
            return mask
        mask.zero_()
        for ch in channels:
            ci = int(ch)
            if ci < 0 or ci >= int(out_channels):
                raise ValueError(f"Boundary source channel index {ci} is outside [0,{int(out_channels) - 1}].")
            mask[ci] = 1.0
        return mask

    def _edge_features_from_pairs(
        self,
        lo: torch.Tensor,
        hi: torch.Tensor,
        pos: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        p = pos[:, : self.edge_pos_dim].to(device=device, dtype=dtype)
        rel = p[hi.long()] - p[lo.long()]
        dist = torch.linalg.norm(rel, dim=1, keepdim=True).clamp_min(1e-12)
        unit = rel / dist
        edge_feat = torch.cat([rel, dist, unit], dim=-1)
        if edge_feat.size(1) < self.edge_in_channels:
            pad = torch.zeros(
                (edge_feat.size(0), self.edge_in_channels - edge_feat.size(1)),
                device=device,
                dtype=dtype,
            )
            edge_feat = torch.cat([edge_feat, pad], dim=-1)
        elif edge_feat.size(1) > self.edge_in_channels:
            edge_feat = edge_feat[:, : self.edge_in_channels]
        return edge_feat

    def _build_edge_features(
        self,
        edge_index: torch.Tensor,
        pos: Optional[torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if pos is None:
            raise ValueError("FluxGraphNet.forward requires pos=... for unstructured edge-flux geometry.")
        if pos.ndim != 2 or int(pos.size(1)) < self.edge_pos_dim:
            raise ValueError(
                f"FluxGraphNet expected pos shape [N,>={self.edge_pos_dim}], got {tuple(pos.shape)}"
            )
        return self._edge_features_from_pairs(edge_index[0], edge_index[1], pos, dtype=dtype, device=device)

    def _unique_undirected_pairs(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes) & (src != dst)
        if not bool(valid.any()):
            empty = src.new_empty((0,))
            return empty, empty
        src = src[valid]
        dst = dst[valid]
        lo_all = torch.minimum(src, dst)
        hi_all = torch.maximum(src, dst)
        pair_key = lo_all * int(num_nodes) + hi_all
        unique_key = torch.unique(pair_key, sorted=True)
        lo = torch.div(unique_key, int(num_nodes), rounding_mode="floor")
        hi = unique_key - (lo * int(num_nodes))
        return lo.long(), hi.long()

    def _volume_factor(
        self,
        dual_volume: Optional[torch.Tensor],
        *,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.use_dual_volume or dual_volume is None:
            return None
        vol = dual_volume.to(device=device, dtype=dtype)
        if vol.ndim == 2:
            vol = vol[:, 0]
        elif vol.ndim != 1:
            raise ValueError(f"dual_volume must be [N] or [N,1], got shape={tuple(vol.shape)}.")
        if int(vol.numel()) != int(num_nodes):
            raise ValueError(f"dual_volume node count mismatch: got {int(vol.numel())}, expected {num_nodes}.")
        return vol.clamp_min(self.volume_floor).view(-1, 1)

    def _compute_flux_update(
        self,
        *,
        node_latent: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        dual_volume: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_nodes = int(node_latent.size(0))
        lo, hi = self._unique_undirected_pairs(edge_index, num_nodes)
        update = torch.zeros((num_nodes, self.out_channels), device=node_latent.device, dtype=node_latent.dtype)
        if lo.numel() == 0:
            return update

        h_lo = node_latent[lo]
        h_hi = node_latent[hi]
        pair_mean = 0.5 * (h_lo + h_hi)
        pair_absdiff = torch.abs(h_hi - h_lo)
        edge_feat = self._edge_features_from_pairs(
            lo,
            hi,
            pos,
            dtype=node_latent.dtype,
            device=node_latent.device,
        )
        flux_in = torch.cat([pair_mean, pair_absdiff, edge_feat], dim=-1)
        flux = self.flux_head(flux_in) * self.flux_scale
        update.index_add_(0, lo, -flux)
        update.index_add_(0, hi, flux)

        vol = self._volume_factor(
            dual_volume,
            num_nodes=num_nodes,
            device=node_latent.device,
            dtype=node_latent.dtype,
        )
        if vol is not None:
            update = update / vol
        return update

    def _domain_bbox_for_pos(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pxy = pos[:, :2]
        if self.domain_bbox is None:
            xmin = pxy[:, 0].min()
            xmax = pxy[:, 0].max()
            ymin = pxy[:, 1].min()
            ymax = pxy[:, 1].max()
        else:
            xmin, xmax, ymin, ymax = [pxy.new_tensor(v) for v in self.domain_bbox]
        return xmin, xmax, ymin, ymax

    def _linear_gate_from_distance(self, dist: torch.Tensor, width: float) -> torch.Tensor:
        width_t = max(float(width), 1e-12)
        return (1.0 - (dist / width_t)).clamp(min=0.0, max=1.0)

    def _boundary_side_gates_and_cylinder(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pxy = pos[:, :2]
        xmin, xmax, ymin, ymax = self._domain_bbox_for_pos(pxy)
        x = pxy[:, 0]
        y = pxy[:, 1]
        x_span = (xmax - xmin).abs().clamp_min(1e-12)
        y_span = (ymax - ymin).abs().clamp_min(1e-12)
        side_dist = torch.stack(
            [
                (x - xmin).clamp_min(0.0) / x_span,
                (xmax - x).clamp_min(0.0) / x_span,
                (y - ymin).clamp_min(0.0) / y_span,
                (ymax - y).clamp_min(0.0) / y_span,
            ],
            dim=1,
        )
        side_gates = self._linear_gate_from_distance(side_dist, self.boundary_width)

        cxy = self.cylinder_center_xy.to(device=pos.device, dtype=pos.dtype).view(1, 2)
        radial = torch.linalg.norm(pxy - cxy, dim=1, keepdim=True)
        cylinder_dist = (radial - float(self.cylinder_radius)).abs()
        cylinder_gate = self._linear_gate_from_distance(
            cylinder_dist,
            max(float(self.cylinder_boundary_width), 1e-12),
        )
        return side_gates, cylinder_gate

    def _open_boundary_flux_source(
        self,
        *,
        state: torch.Tensor,
        side_gates: torch.Tensor,
        cylinder_gate: torch.Tensor,
    ) -> torch.Tensor:
        source = torch.zeros_like(state)
        if (
            not self.use_open_boundary_source
            or self.open_boundary_flux_scale <= 0.0
            or side_gates.numel() == 0
        ):
            return source

        side_mask = self.open_boundary_flux_side_mask.to(device=state.device, dtype=state.dtype).view(1, 4)
        mode_mask = self.open_boundary_flux_mode_side_mask.to(device=state.device, dtype=state.dtype).view(1, 4)
        gates = (side_gates.to(dtype=state.dtype) * side_mask * mode_mask * (1.0 - cylinder_gate)).clamp(0.0, 1.0)
        if not bool(torch.any(gates > 0).item()):
            return source

        u = state[:, self.velocity_channels[0]]
        v = state[:, self.velocity_channels[1]]
        normals = state.new_tensor(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
            ]
        )
        channel_mask = self.open_boundary_source_mask.to(device=state.device, dtype=state.dtype).view(1, -1)
        for side in range(4):
            gate = gates[:, side : side + 1]
            if not bool(torch.any(gate > 0).item()):
                continue
            un = (u * normals[side, 0]) + (v * normals[side, 1])
            speed = un.clamp_min(0.0) if self.open_boundary_flux_outflow_only else un
            source = source - float(self.open_boundary_flux_scale) * gate * speed.view(-1, 1) * state * channel_mask
        return source

    def _boundary_source(
        self,
        *,
        state: torch.Tensor,
        node_latent: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        source = torch.zeros_like(state)
        if not self.use_open_boundary_source:
            return source
        side_gates, cylinder_gate = self._boundary_side_gates_and_cylinder(pos)
        side_gates = side_gates.to(device=state.device, dtype=state.dtype)
        cylinder_gate = cylinder_gate.to(device=state.device, dtype=state.dtype)

        learned_mask = self.open_boundary_learned_side_mask.to(device=state.device, dtype=state.dtype).view(1, 4)
        learned_side_gates = (side_gates * learned_mask * (1.0 - cylinder_gate)).clamp(0.0, 1.0)
        learned_gate = learned_side_gates.max(dim=1, keepdim=True).values
        if self.open_boundary_head is not None and bool(torch.any(learned_gate > 0).item()):
            channel_mask = self.open_boundary_source_mask.to(device=state.device, dtype=state.dtype).view(1, -1)
            source = source + learned_gate * self.open_boundary_head(node_latent) * channel_mask

        source = source + self._open_boundary_flux_source(
            state=state,
            side_gates=side_gates,
            cylinder_gate=cylinder_gate,
        )
        return source

    def _state_slice(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.state_channel : self.state_channel + self.out_channels]

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        dual_volume: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ):
        del dt
        if pos is None:
            raise ValueError("FluxGraphNet.forward requires pos=... for graph-edge flux geometry.")
        edge_index = edge_index.to(device=X.device, dtype=torch.long)
        pos = pos.to(device=X.device, dtype=X.dtype)
        edge_feat = self._build_edge_features(
            edge_index,
            pos,
            dtype=X.dtype,
            device=X.device,
        )

        node_h = self.node_encoder(X)
        edge_h = self.edge_encoder(edge_feat)
        for block in self.processor:
            node_h, edge_h = block(node_h, edge_h, edge_index)
            node_h = self.block_activation(node_h)
            node_h = F.dropout(node_h, p=self.dropout, training=self.training)

        state = self._state_slice(X)
        update = self._compute_flux_update(
            node_latent=node_h,
            edge_index=edge_index,
            pos=pos,
            dual_volume=dual_volume,
        )
        update = update + self._boundary_source(state=state, node_latent=node_h, pos=pos)

        if self.predict_type == PREDICT_TYPE_STATE:
            y_feat = state + update
        elif self.predict_type in {PREDICT_TYPE_DELTA, PREDICT_TYPE_RATE}:
            y_feat = update
        else:
            raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")

        y_score = self.score_head(node_h) if self.score_head is not None else None
        return y_feat, y_score, node_h

    def predict_state(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        pos: Optional[torch.Tensor] = None,
        dual_volume: Optional[torch.Tensor] = None,
        dt: float = 1.0,
        state_override: Optional[torch.Tensor] = None,
        state_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if state_override is not None and state_residual is not None:
            raise ValueError("Pass only one of state_override or state_residual.")
        state_in = self._state_slice(X)
        state_ref = state_override if state_override is not None else state_residual
        if state_ref is None:
            state_base = state_in
        else:
            if state_ref.ndim != 2 or state_ref.shape != state_in.shape:
                raise ValueError(
                    "state_override/state_residual must match the state slice shape "
                    f"{tuple(state_in.shape)}, got {tuple(state_ref.shape)}."
                )
            state_base = state_ref.to(device=X.device, dtype=X.dtype)

        y, _score, _h = self.forward(X, edge_index, pos=pos, dual_volume=dual_volume, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            if state_ref is None:
                return y
            return y + (state_base - state_in)
        if self.predict_type == PREDICT_TYPE_DELTA:
            return state_base + y
        if self.predict_type == PREDICT_TYPE_RATE:
            return state_base + float(dt) * y
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")


class FeatureExtractorGNN(nn.Module):
    """
    GraphUNet-based feature extractor for each node with attention.
    """
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=128, 
                 depth=3, pool_ratios=0.5, heads=4, concat=True, dropout=0.6):
        super(FeatureExtractorGNN, self).__init__()
        self.unet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            depth=depth,
            pool_ratios=pool_ratios,
            act=F.relu
        )
        self.attention1 = GATConv(out_channels, out_channels, heads=heads, 
                                  concat=concat, dropout=dropout)
        self.attention2 = GATConv(out_channels * heads if concat else out_channels, 
                                  out_channels, heads=1, concat=False, dropout=dropout)
        # This linear layer for the residual connection needs to match the output of attention2
        self.residual_proj = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        # The original input is passed to the UNet
        unet_out = self.unet(x, edge_index)
        
        # The output of the UNet is used for the residual connection and the attention layers
        residual = self.residual_proj(unet_out)
        
        x = F.elu(self.attention1(unet_out, edge_index))
        x = self.attention2(x, edge_index)
        
        # Add the projected residual
        x += residual
        return x
    
class MPSFeatureExtractor(nn.Module):
    """
    MPS-safe alternative to FeatureExtractorGNN.
    Two GAT layers + residual; no GraphUNet, no CSR sparse ops.
    """
    def __init__(self, in_channels=2, hidden_channels=32, out_channels=32,
                 heads=2, concat=True, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, out_channels,
                            heads=heads, concat=concat, dropout=dropout)
        h1 = out_channels * heads if concat else out_channels
        self.gat2 = GATConv(h1, out_channels, heads=1, concat=False, dropout=dropout)
        self.residual_proj = nn.Linear(in_channels, out_channels)  # for skip from input

    def forward(self, x, edge_index):
        res = self.residual_proj(x)
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x + res
    
class DerivativeGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=3,
                 num_layers=3, heads=4, concat=True, dropout=0.2, use_residual=True):
        super(DerivativeGNN, self).__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.out_channels = out_channels

        if self.use_residual:
            self.residual_proj = nn.Linear(in_channels, out_channels)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            current_in = in_channels if i == 0 else hidden_channels * (heads if concat else 1)
            current_out = hidden_channels if i < num_layers - 1 else out_channels
            is_last_layer = (i == num_layers - 1)
            
            self.layers.append(nn.LayerNorm(current_in, eps=1e-6))# adding in , eps=1e-6
            self.layers.append(
                GATConv(current_in, current_out, 
                        heads=1 if is_last_layer else heads,
                        concat=False if is_last_layer else concat, 
                        dropout=dropout)
            )

    def forward(self, x, edge_index):
        residual = self.residual_proj(x) if self.use_residual else None

        for i in range(self.num_layers):
            ln = self.layers[2*i]
            gnn = self.layers[2*i + 1]
            
            x_res = x
            x = ln(x)
            x = gnn(x, edge_index)

            if i < self.num_layers - 1:
                x = F.gelu(x)
                if x.shape == x_res.shape: # Add skip connections between layers
                     x = x + x_res
        
        if self.use_residual and residual is not None:
            x = x + residual
            
        return x
    

class IntegralGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=3, 
                 num_layers=3, heads=4, concat=True, dropout=0.2, use_residual=True):
        super(IntegralGNN, self).__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.out_channels = out_channels

        if use_residual:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            current_in = in_channels if i == 0 else hidden_channels * (heads if concat else 1)
            current_out = hidden_channels if i < num_layers - 1 else out_channels
            is_last_layer = (i == num_layers - 1)

            self.layers.append(nn.LayerNorm(current_in, eps=1e-6))#, eps=1e-6
            self.layers.append(
                GATConv(current_in, current_out, 
                        heads=1 if is_last_layer else heads,
                        concat=False if is_last_layer else concat, 
                        dropout=dropout)
            )

    def forward(self, x, edge_index):
        residual = self.residual_proj(x) if self.use_residual else None
        
        for i in range(self.num_layers):
            ln = self.layers[2*i]
            gnn = self.layers[2*i + 1]
            
            x_res = x
            x = ln(x)
            x = gnn(x, edge_index)

            if i < self.num_layers - 1:
                x = F.gelu(x)
                if x.shape == x_res.shape: # Add skip connections between layers
                    x = x + x_res

        if self.use_residual and residual is not None:
            x = x + residual
            
        return x

# ---------------------------------------------------------------------
# GPARC-style wrapper keeping your modules unchanged
# ---------------------------------------------------------------------
class GPARCCompat(nn.Module):
    """
    Wraps FeatureExtractorGNN, DerivativeGNN, IntegralGNN into a single nn.Module.

    forward inputs:
      x_static : [N, S] (pos, level, etc.)
      x_dyn    : [N, D] (rho, px, E at time t on M_pred(t+1))
      edge_idx : [2, E]
      g        : [G] or [B,G] (optional global conditioning, currently ignored)

    forward outputs:
      x_pred   : [N, D] (features at t+1 on M_pred(t+1))
    """
    def __init__(
        self,
        in_static: int,
        in_dynamic: int,
        feature_out: int = 128,
        feat_hidden: int = 64,
        feat_depth: int = 2,
        feat_pool: float = 0.1,
        feat_heads: int = 4,
        feat_dropout: float = 0.2,
        deriv_hidden: int = 128,
        deriv_layers: int = 4,
        deriv_heads: int = 8,
        deriv_dropout: float = 0.3,
        deriv_residual: bool = True,
        integ_hidden: int = 128,
        integ_layers: int = 4,
        integ_heads: int = 8,
        integ_dropout: float = 0.3,
        integ_residual: bool = True,
        use_delta: bool = True,
        global_embed_dim: int = 0,  # set >0 if you later add a global embedding
    ):
        super().__init__()
        self.use_delta = bool(use_delta)
        self.in_dynamic = int(in_dynamic)
        self.global_embed_dim = int(global_embed_dim)

        # 1) per-node static encoder (GraphUNet + attention)
        self.feature_extractor = MPSFeatureExtractor(
            in_channels=in_static,
            hidden_channels=feat_hidden,
            out_channels=feature_out,
            heads=feat_heads,
            concat=True,
            dropout=feat_dropout,
        )

        # 2) time-derivative GNN (takes concat[feat_emb, x_dyn, (g_emb)])
        deriv_in = feature_out + in_dynamic + self.global_embed_dim
        self.derivative = DerivativeGNN(
            in_channels=deriv_in,
            hidden_channels=deriv_hidden,
            out_channels=in_dynamic,
            num_layers=deriv_layers,
            heads=deriv_heads,
            concat=True,
            dropout=deriv_dropout,
            use_residual=deriv_residual,
        )

        # 3) integrator GNN (maps derivative to a delta or absolute)
        self.integrator = IntegralGNN(
            in_channels=in_dynamic,
            hidden_channels=integ_hidden,
            out_channels=in_dynamic,
            num_layers=integ_layers,
            heads=integ_heads,
            concat=True,
            dropout=integ_dropout,
            use_residual=integ_residual,
        )

    def forward(
        self,
        x_static: torch.Tensor,
        x_dyn: torch.Tensor,
        edge_index: torch.Tensor,
        g: torch.Tensor | None = None,
    ):
        # 1) Feature embedding from static geometry
        z = self.feature_extractor(x_static, edge_index)  # [N, feature_out]

        # 2) Concatenate dynamic and optional global context
        if self.global_embed_dim > 0 and g is not None:
            if g.dim() == 1:
                g = g[None, :]                 # [1, G]
            g = g.expand(z.size(0), -1)        # naive broadcast
            h_in = torch.cat([z, x_dyn, g], dim=-1)
        else:
            h_in = torch.cat([z, x_dyn], dim=-1)

        # 3) Derivative -> Integral stacks
        dstate = self.derivative(h_in, edge_index)        # [N, D]
        delta  = self.integrator(dstate, edge_index)      # [N, D]

        # 4) Output contract:
        #    - use_delta=True  -> return Δ (your training/rollout expect this)
        #    - use_delta=False -> return absolute (x_dyn + Δ)
        return delta if self.use_delta else (x_dyn + delta)

def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def build_model(cfg: Dict[str, Any], in_dim: int, out_dim: int):
    """
    Build a model from cfg. Expected cfg structure:
      cfg["model"] = {
        "type": "sageconv" or "meshgraphnet",
        # ... hyperparameters specific to that class ...
      }

    We will pass (in_dim, out_dim, **filtered_kwargs) if the constructor
    accepts them; otherwise we pass only what the constructor supports.
    """
    if cfg is None:
        raise ValueError("cfg is None; need cfg['model']['type'] or cfg['model']['name'] at minimum.")
    model_cfg = cfg.get("model", cfg)  # allow cfg itself to be the model dict
    name = model_cfg.get("name", None)
    model_type = model_cfg.get("type", None)
    selector = model_type if model_type is not None else name
    if not selector:
        raise ValueError("cfg['model']['type'] or cfg['model']['name'] is required, e.g. 'sageconv'.")

    # Resolve the class by name from this module
    aliases = {
        "feature_net": "FeatureNet",
        "featurenet": "FeatureNet",
        "graphsage": "FeatureNet",
        "graph_sage": "FeatureNet",
        "sage": "FeatureNet",
        "sageconv": "FeatureNet",
        "meshgraphnet": "MeshGraphNet",
        "mesh_graph_net": "MeshGraphNet",
        "mgn": "MeshGraphNet",
        "fluxgraphnet": "FluxGraphNet",
        "flux_graph_net": "FluxGraphNet",
        "fluxgnn": "FluxGraphNet",
        "flux": "FluxGraphNet",
    }
    selector_key = str(selector).strip().lower().replace("-", "_")
    class_name = aliases.get(selector_key, str(selector).strip())
    cls = globals().get(class_name, None)
    if cls is None:
        # Also allow lowercase alias
        for k, v in globals().items():
            if k.lower() == class_name.lower() and inspect.isclass(v):
                cls = v
                break
    if cls is None or not inspect.isclass(cls):
        raise ValueError(f"Model class or type '{selector}' not found in models.py.")

    # Prepare kwargs: include in_dim/out_dim if the ctor supports them
    ctor = cls.__init__
    kwargs = dict(model_cfg)  # copy
    kwargs.pop("name", None)
    kwargs.pop("type", None)
    kwargs = {k: v for k, v in kwargs.items() if not str(k).startswith("_")}
    # Some configs put dims under different keys; keep a few aliases:
    if "input_dim" not in kwargs:  kwargs["input_dim"]  = in_dim
    if "in_dim" not in kwargs:     kwargs["in_dim"]     = in_dim
    if "in_channels" not in kwargs: kwargs["in_channels"] = in_dim
    if "out_dim" not in kwargs:    kwargs["out_dim"]    = out_dim
    if "output_dim" not in kwargs: kwargs["output_dim"] = out_dim
    if "out_channels" not in kwargs: kwargs["out_channels"] = out_dim

    kwargs = _filter_kwargs(ctor, kwargs)

    # Instantiate
    model = cls(**kwargs)

    # If the class exposes a 'reset_parameters' helper, call it
    if hasattr(model, "reset_parameters") and callable(getattr(model, "reset_parameters")):
        try:
            model.reset_parameters()
        except Exception:
            pass

    return model

# Back-compat alias some training scripts use
make_model = build_model


class ParcFeatureAdapter(nn.Module):
    """
    Applies:
      (1) pre-clip on PARC features
      (2) running normalization (train updates; eval uses frozen stats)
      (3) learnable gates (per-channel by default) initialized near 0 influence
      (4) optional post-clip
    """
    def __init__(
        self,
        dim_adv: int,
        dim_diff: int,
        *,
        use_norm: bool = True,
        clip_pre: float = 50.0,
        clip_post: float = 10.0,
        momentum: float = 0.02,
        eps: float = 1e-6,
        var_floor: float = 1e-6,
        per_channel_gates: bool = True,
        gate_init: float = -3.0,   # sigmoid(-5) ~ 0.0067 (starts near OFF)
    ):
        super().__init__()
        self.dim_adv = int(dim_adv)
        self.dim_diff = int(dim_diff)
        self.dim = self.dim_adv + self.dim_diff

        self.use_norm = bool(use_norm)
        self.clip_pre = float(clip_pre) if clip_pre is not None else None
        self.clip_post = float(clip_post) if clip_post is not None else None
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.var_floor = float(var_floor)

        self.per_channel_gates = bool(per_channel_gates)

        # Running stats in FP32 (stable across AMP / MPS / CUDA)
        self.register_buffer("running_mean", torch.zeros(self.dim, dtype=torch.float32))
        self.register_buffer("running_var",  torch.ones(self.dim,  dtype=torch.float32))
        self.register_buffer("num_updates",  torch.tensor(0, dtype=torch.long))

        # Gates
        if self.per_channel_gates:
            self.gate_logits = nn.Parameter(torch.full((self.dim,), float(gate_init), dtype=torch.float32))
        else:
            # one scalar for adv block, one for diff block
            self.gate_adv_logit  = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
            self.gate_diff_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def _gate_vector(self, device, dtype):
        if self.dim == 0:
            return None
        if self.per_channel_gates:
            g = torch.sigmoid(self.gate_logits)  # [dim]
        else:
            g_adv  = torch.sigmoid(self.gate_adv_logit)
            g_diff = torch.sigmoid(self.gate_diff_logit)
            parts = []
            if self.dim_adv  > 0: parts.append(g_adv.expand(self.dim_adv))
            if self.dim_diff > 0: parts.append(g_diff.expand(self.dim_diff))
            g = torch.cat(parts, dim=0) if len(parts) else torch.zeros((0,), dtype=torch.float32)
        return g.to(device=device, dtype=dtype).view(1, -1)  # [1,dim]

    def forward(
        self,
        parc_extra: torch.Tensor,
        *,
        update_adv_stats: bool = True,
        update_diff_stats: bool = True,
    ) -> torch.Tensor:
        if parc_extra is None or parc_extra.numel() == 0:
            return parc_extra
        if parc_extra.ndim != 2 or parc_extra.size(1) != self.dim:
            raise RuntimeError(f"ParcFeatureAdapter expected [N,{self.dim}], got {tuple(parc_extra.shape)}")

        out_dtype = parc_extra.dtype
        x = parc_extra.to(dtype=torch.float32)

        # 1) pre-clip
        if (self.clip_pre is not None) and (self.clip_pre > 0):
            x = x.clamp(-self.clip_pre, self.clip_pre)

        # 2) normalization
        if self.use_norm:
            if self.training:
                # ---- batch stats for normalization ----
                m = x.mean(dim=0)
                v = (x - m).pow(2).mean(dim=0)  # population var

                denom = torch.sqrt(v.clamp_min(self.var_floor) + self.eps)
                x = (x - m) / denom

                # ---- selective EMA update for eval-time stability ----
                did_update = False
                if self.dim_adv > 0 and update_adv_stats:
                    sl = slice(0, self.dim_adv)
                    self.running_mean[sl].mul_(1.0 - self.momentum).add_(self.momentum * m[sl].detach())
                    self.running_var[sl].mul_(1.0 - self.momentum).add_(self.momentum * v[sl].detach())
                    did_update = True

                if self.dim_diff > 0 and update_diff_stats:
                    sl = slice(self.dim_adv, self.dim_adv + self.dim_diff)
                    self.running_mean[sl].mul_(1.0 - self.momentum).add_(self.momentum * m[sl].detach())
                    self.running_var[sl].mul_(1.0 - self.momentum).add_(self.momentum * v[sl].detach())
                    did_update = True

                if did_update:
                    self.num_updates.add_(1)

            else:
                # ---- eval: use running stats ----
                denom = torch.sqrt(self.running_var.clamp_min(self.var_floor) + self.eps)
                x = (x - self.running_mean) / denom

        # 3) gates
        g = self._gate_vector(device=x.device, dtype=x.dtype)  # [1,dim]
        if g is not None:
            x = x * g

        # 4) post-clip
        if (self.clip_post is not None) and (self.clip_post > 0):
            x = x.clamp(-self.clip_post, self.clip_post)

        return x.to(dtype=out_dtype)

    @torch.no_grad()
    def gate_values(self):
        """Convenience for debug prints."""
        if self.dim == 0:
            return {"adv": None, "diff": None}
        if self.per_channel_gates:
            g = torch.sigmoid(self.gate_logits).detach().cpu()
            ga = g[:self.dim_adv] if self.dim_adv > 0 else None
            gd = g[self.dim_adv:] if self.dim_diff > 0 else None
            return {"adv": ga, "diff": gd}
        else:
            return {
                "adv":  float(torch.sigmoid(self.gate_adv_logit).detach().cpu()),
                "diff": float(torch.sigmoid(self.gate_diff_logit).detach().cpu()),
            }
