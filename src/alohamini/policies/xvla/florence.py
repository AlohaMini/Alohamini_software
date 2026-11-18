#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from typing import Any


def _translate_vision_config(vision_config: dict[str, Any]) -> dict[str, Any]:
    """Translate a vision config from the original Microsoft remote-code Florence-2 format
    (used by existing XVLA checkpoints) to the native ``transformers`` format.

    Configs already in the native format pass through unchanged.
    """
    vision = dict(vision_config)
    model_type = vision.pop("model_type", None)
    if model_type not in (None, "davit", "florence_vision"):
        raise ValueError(f"Unsupported Florence-2 vision backbone: {model_type!r}")
    vision.pop("enable_checkpoint", None)

    image_pos_embed = vision.pop("image_pos_embed", None)
    if image_pos_embed is not None:
        if image_pos_embed.get("type") != "learned_abs_2d":
            raise ValueError(f"Unsupported image_pos_embed type: {image_pos_embed.get('type')!r}")
        vision["max_position_embeddings"] = image_pos_embed["max_pos_embeddings"]

    visual_temporal_embedding = vision.pop("visual_temporal_embedding", None)
    if visual_temporal_embedding is not None:
        if visual_temporal_embedding.get("type") != "COSINE":
            raise ValueError(
                "Unsupported visual_temporal_embedding type: "
                f"{visual_temporal_embedding.get('type')!r}"
            )
        vision["max_temporal_embeddings"] = visual_temporal_embedding["max_temporal_embeddings"]

    image_feature_source = vision.pop("image_feature_source", None)
    if image_feature_source is not None and list(image_feature_source) != [
        "spatial_avg_pool",
        "temporal_avg_pool",
    ]:
        # the native Florence2MultiModalProjector hardcodes this feature combination
        raise ValueError(f"Unsupported image_feature_source: {image_feature_source!r}")

    if "dim_embed" in vision:
        vision["embed_dim"] = vision.pop("dim_embed")
    return vision
