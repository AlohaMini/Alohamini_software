"""Language supervision metadata uses the same explicit FK coordinates as the model."""

import torch
from test_data_fields import source_dataset
from test_fineart_vla import tiny_options
from test_language_data import add_annotations
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.data_fields import prepare_fields
from alohamini.learning.loading import make_loader
from alohamini.learning.validation import loss_denominators
from alohamini.policies.fineart_vla.adapter import FineARTVLAAlgorithm


def test_transformed_language_counter_matches_training_state(recording, tmp_path, monkeypatch):
    recipe = FineARTVLAAlgorithm()
    options = tiny_options(tmp_path / "assets", monkeypatch)
    source, settings, _ = source_dataset(recording, tmp_path)
    add_annotations(source.root)
    source = AlohaMiniDataset(
        source.root,
        episodes=[0],
        state="recorded",
        allow_unannotated=True,
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    data = prepare_fields(source, settings, recipe, options=options)
    assert source[0]["observation.state"].shape == (3,)
    assert data[0]["observation.state"].shape == (7,)
    torch.testing.assert_close(
        data.action_metadata(0)["observation.state"], data[0]["observation.state"]
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    model = recipe.build(options)
    stats = recipe.statistics(data)
    counter = recipe.make_loss_counter(model.config, stats)
    processor = recipe.processor(model, stats, "cpu")
    raw = next(iter(make_loader(data, batch_size=len(data))))
    assert loss_denominators(data, counter) == counter(raw)
    loss, _ = model(processor(raw))
    loss.backward()
    assert torch.isfinite(loss)
