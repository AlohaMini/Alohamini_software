WAN_WRAPPED_MODULE = "alohamini.policies.openwam._vendor.model.video_backbone.wan.shared.core.vram.layers.AutoWrappedModule"
WAN_WRAPPED_LINEAR = "alohamini.policies.openwam._vendor.model.video_backbone.wan.shared.core.vram.layers.AutoWrappedLinear"
WAN_WRAPPED_NON_RECURSE = "alohamini.policies.openwam._vendor.model.video_backbone.wan.shared.core.vram.layers.AutoWrappedNonRecurseModule"

VRAM_MANAGEMENT_MODULE_MAPS = {
    "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.dit.WanModel": {
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.dit.MLP": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.dit.DiTBlock": WAN_WRAPPED_NON_RECURSE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.dit.Head": WAN_WRAPPED_MODULE,
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv3d": WAN_WRAPPED_MODULE,
        "torch.nn.LayerNorm": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.dit.RMSNorm": WAN_WRAPPED_MODULE,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
    },
    "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.image_encoder.WanImageEncoder": {
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.image_encoder.VisionTransformer": WAN_WRAPPED_MODULE,
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
        "torch.nn.LayerNorm": WAN_WRAPPED_MODULE,
    },
    "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.text_encoder.WanTextEncoder": {
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Embedding": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.text_encoder.T5RelativeEmbedding": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.text_encoder.T5LayerNorm": WAN_WRAPPED_MODULE,
    },
    "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.vace.VaceWanModel": {
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.dit.DiTBlock": WAN_WRAPPED_MODULE,
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv3d": WAN_WRAPPED_MODULE,
        "torch.nn.LayerNorm": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.dit.RMSNorm": WAN_WRAPPED_MODULE,
    },
    "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.vae.WanVideoVAE": {
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.vae.RMS_norm": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.vae.CausalConv3d": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.vae.Upsample": WAN_WRAPPED_MODULE,
        "torch.nn.SiLU": WAN_WRAPPED_MODULE,
        "torch.nn.Dropout": WAN_WRAPPED_MODULE,
    },
    "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.vae.WanVideoVAE38": {
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.vae.RMS_norm": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.vae.CausalConv3d": WAN_WRAPPED_MODULE,
        "alohamini.policies.openwam._vendor.model.video_backbone.wan.models.vae.Upsample": WAN_WRAPPED_MODULE,
        "torch.nn.SiLU": WAN_WRAPPED_MODULE,
        "torch.nn.Dropout": WAN_WRAPPED_MODULE,
    },
}
