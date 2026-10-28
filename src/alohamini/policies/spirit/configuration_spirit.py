"""Native physical-field contract around the Spirit v1.5 network."""

from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig

NETWORK_DEFAULTS = dict(
    dit_hidden_size=1536,
    dit_num_heads=32,
    dit_num_layers=16,
    dit_interleave_self_attention=True,
    dit_cross_attention_dim=2560,
    max_state_dim=32,
    max_action_dim=32,
    chunk_size=60,
    num_steps=10,
    attention_implementation="eager",
    dit_dropout=0.0,
)


@dataclass
class SpiritConfig(PolicyConfig):
    network: dict = field(default_factory=dict)
    backbone_path: str = ""
    n_action_steps: int = 60
    camera_mapping: dict = field(default_factory=dict)
    robot_type: str = "aloha"
    action_representation: str = "eef_delta"
    kinematics_path: str = ""
    kinematic_fields: dict = field(default_factory=dict)
    gradient_checkpointing: bool = True
    image_augmentation: bool = True
    optimizer_lr: float = 2.5e-5
    optimizer_weight_decay: float = 1e-10
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "warmup_cosine"
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 50000
    scheduler_decay_lr: float = 2.5e-6

    @property
    def chunk_size(self):
        return self.network["chunk_size"]

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.network, dict):
            raise ValueError("Spirit network settings must be a dictionary")
        if set(self.network) - set(NETWORK_DEFAULTS):
            raise ValueError(
                "Unknown Spirit network settings: "
                + str(sorted(set(self.network) - set(NETWORK_DEFAULTS)))
            )
        self.network = {**NETWORK_DEFAULTS, **self.network}
        if self.robot_state_feature is None:
            raise ValueError("Spirit requires an explicitly selected numeric observation state")
        if (
            not self.backbone_path
            or not isinstance(self.robot_type, str)
            or not self.robot_type.strip()
        ):
            raise ValueError("Spirit requires local Qwen assets and an explicit robot type")
        if self.action_representation not in ("recorded", "eef_delta"):
            raise ValueError("action_representation must be recorded or eef_delta")
        if self.action_representation == "eef_delta" and not self.kinematic_fields:
            raise ValueError("eef_delta requires an explicit kinematic-fields descriptor")
        self.encoded_state_dim = self.robot_state_feature.shape[0]
        self.encoded_action_dim = self.action_feature.shape[0]
        if self.action_representation == "eef_delta":
            from alohamini.datasets.kinematic_fields import KinematicFields

            geometry = KinematicFields(self.kinematic_fields)
            if any(
                width != len(geometry.names)
                for width in (self.encoded_state_dim, self.encoded_action_dim)
            ):
                raise ValueError("Kinematic source width must match original state/action features")
            self.encoded_state_dim = self.encoded_action_dim = geometry.width
        if any(type(v) is not bool for v in (self.gradient_checkpointing, self.image_augmentation)):
            raise ValueError("gradient_checkpointing and image_augmentation must be boolean")
        for name in (
            "chunk_size",
            "num_steps",
            "dit_hidden_size",
            "dit_num_heads",
            "dit_num_layers",
            "max_state_dim",
            "max_action_dim",
        ):
            if type(self.network[name]) is not int or self.network[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.n_action_steps) is not int
            or not 1 <= self.n_action_steps <= self.network["chunk_size"]
        ):
            raise ValueError("n_action_steps must fit the generated chunk as a positive integer")
        if self.network["chunk_size"] > 511:
            raise ValueError("Spirit supports at most 511 action tokens plus one state token")
        if self.network["dit_hidden_size"] % self.network["dit_num_heads"]:
            raise ValueError("DiT hidden size must be divisible by attention heads")
        if (
            self.encoded_action_dim > self.network["max_action_dim"]
            or self.encoded_state_dim > self.network["max_state_dim"]
        ):
            raise ValueError("Encoded feature width exceeds the Spirit padded dimensions")
        cameras = {k.removeprefix("observation.images.") for k in self.image_features}
        if not self.camera_mapping:
            self.camera_mapping = {
                slot: source
                for slot, source in (
                    ("cam_high", "forward"),
                    ("cam_left_wrist", "wrist_left"),
                    ("cam_right_wrist", "wrist_right"),
                )
                if source in cameras
            }
        if (
            not self.camera_mapping
            or set(self.camera_mapping) - {"cam_high", "cam_left_wrist", "cam_right_wrist"}
            or set(self.camera_mapping.values()) != cameras
            or len(set(self.camera_mapping.values())) != len(self.camera_mapping)
        ):
            raise ValueError(
                "camera_mapping must explicitly map each selected camera to a Spirit slot"
            )
