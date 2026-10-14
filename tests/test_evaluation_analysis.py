"""Analysis must preserve recorded clocks, coordinate names and lazy media semantics."""

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("numpy", "pyarrow", "PIL"))


@unittest.skipUnless(DEPENDENCIES, "analysis dependencies unavailable")
class EvaluationAnalysisTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image

        from alohamini.analysis import metrics, source

        self.metrics, self.source = metrics, source
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        (self.root / "meta/episodes/chunk-000").mkdir(parents=True)
        (self.root / "data/chunk-000").mkdir(parents=True)
        self.info = {
            "codebase_version": "v3.0",
            "fps": 30,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "action": {"names": ["arm_left_joint.pos", "x.vel"], "shape": [2]},
                "observation.state": {"names": ["x.vel", "arm_left_joint.pos"], "shape": [2]},
                "observation.images.wrist_right": {"dtype": "video"},
                "observation.images.wrist_left": {"dtype": "image"},
            },
        }
        self.write_info()
        buffer = io.BytesIO()
        Image.new("RGB", (3, 2), "red").save(buffer, format="PNG")
        self.png = buffer.getvalue()
        rows = []
        for episode in (0, 1):
            for frame in range(4):
                rows.append(
                    dict(
                        episode_index=episode,
                        frame_index=frame,
                        timestamp=frame / 30,
                        action=[frame * 2.0, 0.15],
                        **{
                            "observation.state": [0.0, 100 + frame],
                            "observation.images.wrist_left": {"bytes": self.png, "path": None},
                        },
                    )
                )
        pq.write_table(
            pa.Table.from_pylist(rows),
            self.root / "data/chunk-000/file-000.parquet",
            row_group_size=3,
        )
        metadata = []
        for episode in (0, 1):
            metadata.append(
                {
                    "episode_index": episode,
                    "length": 4,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "videos/observation.images.wrist_right/chunk_index": 0,
                    "videos/observation.images.wrist_right/file_index": 0,
                    "videos/observation.images.wrist_right/from_timestamp": episode * 2.0,
                }
            )
        pq.write_table(
            pa.Table.from_pylist(metadata), self.root / "meta/episodes/chunk-000/file-000.parquet"
        )

    def write_info(self):
        (self.root / "meta/info.json").write_text(json.dumps(self.info))

    def sidecars(self, times=(100, 100.1, 100.2, 100.4), sequences=(10, 11, 13, 14)):
        records = []
        for i, (time, seq) in enumerate(zip(times, sequences, strict=True)):
            records.append(
                dict(
                    episode_index=1,
                    frame_index=i,
                    host_timing={"state_sample_monotonic_s": time, "clock_id": "host-a"},
                    issued_command={
                        "sequence": seq,
                        "client_id": "c",
                        "host_session_id": "h",
                        "control_epoch": 2,
                    },
                    client_timing={
                        "action_sample_started_monotonic_s": 9000 + i,
                        "action_sample_finished_monotonic_s": 9000 + i + 0.02,
                    },
                    safety={
                        "watchdog_events": 12,
                        "host_session_id": "h",
                        "accepted_targets": {"arm_left_joint.pos": -10},
                    },
                )
            )
        path = self.root / "meta/safety/episode_000001.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(json.dumps(s) for s in records))
        return records, path

    def load(self):
        return self.source.load_episode(self.root, 1)

    def test_v3_shared_shard_offsets_and_no_sidecar(self):
        data = self.load()
        self.assertEqual(list(data.media), ["wrist_left", "wrist_right"])
        self.assertEqual([r.index for r in data.media["wrist_right"]], [60, 61, 62, 63])
        self.assertEqual(
            [(r.group, r.index) for r in data.media["wrist_left"]], [(1, 1), (1, 2), (2, 0), (2, 1)]
        )
        result = self.metrics.analyze(data)
        self.assertEqual(result.summary["time_basis"], "dataset_timestamp")
        self.assertIsNone(result.summary["command_hz_estimate"])
        self.assertEqual(result.summary["actions"]["x.vel"]["unit"], "recorded units (unverified)")
        self.assertEqual(
            result.series["actions/arm_left_joint.pos"]["observed_before_request"].tolist(),
            [100, 101, 102, 103],
        )

    def test_host_clock_command_gaps_and_counter_baseline(self):
        self.sidecars()
        result = self.metrics.analyze(self.load(), assumed_nas=3, first_command=10, no_resets=True)
        self.assertAlmostEqual(result.summary["duration_s"], 0.4)
        self.assertAlmostEqual(result.summary["recorded_frame_hz"], 7.5)
        self.assertAlmostEqual(result.summary["command_hz_estimate"], 10)
        self.assertAlmostEqual(result.summary["select_action_ms"]["median"], 20)
        self.assertFalse(any(e["kind"] == "counter_increment" for e in result.events))
        chunk = [e for e in result.events if e["kind"] == "inferred_chunk_transition"]
        self.assertEqual([e["frame_index"] for e in chunk], [2])
        self.assertEqual(chunk[0]["evidence"], "user_assumption")
        self.assertEqual(sum(e["kind"] == "recording_gap" for e in result.events), 1)
        with self.assertRaises(ValueError):
            self.metrics.analyze(self.load(), assumed_nas=3)

    def test_clock_reset_segments_and_single_frame(self):
        self.sidecars(times=(100, 101, 4, 5))
        result = self.metrics.analyze(self.load(), segment=1)
        self.assertEqual(result.indices.tolist(), [2, 3])
        self.assertEqual(result.times.tolist(), [0, 1])
        self.assertEqual(result.summary["segment_count"], 2)
        result = self.metrics.analyze(self.load(), segment=1, time_range=(0, 0.5))
        self.assertEqual(result.summary["selected_rows"], 1)
        self.assertIsNone(result.summary["recorded_frame_hz"])

    def test_missing_host_time_does_not_interpolate_different_clocks(self):
        records, path = self.sidecars()
        records[1].pop("host_timing")
        path.write_text("\n".join(json.dumps(s) for s in records))
        result = self.metrics.analyze(self.load())
        self.assertEqual(result.summary["time_basis"], "dataset_timestamp")
        self.assertIsNone(result.summary["command_hz_estimate"])

    def test_session_change_prevents_command_estimate(self):
        records, path = self.sidecars()
        records[2]["issued_command"]["control_epoch"] = 5
        path.write_text("\n".join(json.dumps(s) for s in records))
        self.assertIsNone(self.metrics.analyze(self.load()).summary["command_hz_estimate"])
        with self.assertRaises(ValueError):
            self.metrics.analyze(self.load(), assumed_nas=3, first_command=10, no_resets=True)

    def test_lazy_image_cache_and_source_mutation_detection(self):
        from alohamini.analysis.media import MediaReader

        data = self.load()
        reader = MediaReader(self.base / "cache")
        self.addCleanup(reader.close)
        ref = data.media["wrist_left"][0]
        self.assertEqual(reader.encoded(ref, "wrist_left"), self.png)
        first = reader.keyframe(ref, "wrist_left")
        with patch.object(reader, "encoded", side_effect=AssertionError("must use cache")):
            self.assertEqual(reader.keyframe(ref, "wrist_left"), first)
        self.assertEqual(reader.cache_hits, 1)
        self.assertEqual(reader.video_frame_requests, 0)
        self.info["fps"] = 25
        self.write_info()
        with self.assertRaises(RuntimeError):
            data.unchanged()

    def test_numeric_cli_never_reads_media_and_preserves_source(self):
        from alohamini.analysis.cli import main
        from alohamini.analysis.media import MediaReader

        def hashes():
            return {
                p: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob("*")
                if p.is_file()
            }

        before = hashes()
        output = self.base / "report"
        with (
            patch.object(MediaReader, "encoded", side_effect=AssertionError("no media reads")),
            patch.object(MediaReader, "keyframe", side_effect=AssertionError("no decoding")),
        ):
            main([str(self.root), "--episode", "1", "--viewer", "none", "--output", str(output)])
        self.assertEqual(before, hashes())
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["performance"]["video_keyframe_reads"], 0)
        self.assertTrue((output / "events.csv").is_file())
        with self.assertRaises(ValueError):
            main([str(self.root), "--viewer", "none", "--output", str(self.root / "report")])

    def test_native_missing_camera_frame(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.info["format"] = "alohamini-episodes"
        self.info["cameras"] = ["wrist_left", "wrist_right"]
        self.info["features"].pop("observation.images.wrist_left")
        self.info["features"].pop("observation.images.wrist_right")
        self.write_info()
        directory = self.root / "episodes/episode_000001"
        directory.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    dict(episode_index=1, frame_index=i, timestamp=i / 30, action=[0.0, 0.0])
                    for i in range(3)
                ]
            ),
            directory / "frames.parquet",
        )
        data = self.load()
        self.assertEqual(data.media["wrist_left"], [None, None, None])
        self.assertEqual(len(data.rows), 3)

    def test_repeated_observation_timestamp_is_not_a_reset(self):
        self.sidecars(times=(100, 100.1, 100.1, 100.2))
        result = self.metrics.analyze(self.load())
        self.assertEqual(result.summary["segment_count"], 1)
        self.assertEqual(result.summary["selected_rows"], 4)

    def test_legacy_host_jpeg_color_is_corrected_and_cached(self):
        from PIL import Image

        from alohamini.analysis.media import MediaReader
        from alohamini.analysis.source import Media

        directory = self.base / "episode"
        path = directory / "images/wrist_left/frame_000000.jpg"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (4, 4), (0, 0, 255)).save(path)
        ref = Media("native_host_image", directory, reference="images/wrist_left/frame_000000.jpg")
        reader = MediaReader(self.base / "cache")
        self.addCleanup(reader.close)
        with Image.open(reader.keyframe(ref, "wrist_left")) as image:
            red, green, blue = image.getpixel((0, 0))
            self.assertGreater(red, 250)
            self.assertLess(blue, 5)

    @unittest.skipUnless(importlib.util.find_spec("rerun"), "Rerun unavailable")
    def test_rerun_embedded_image_recording(self):
        import rerun as rr

        from alohamini.analysis.media import MediaReader
        from alohamini.analysis.viewer import show

        reader = MediaReader(self.base / "cache")
        self.addCleanup(reader.close)
        rr.set_strict_mode(True)
        self.addCleanup(rr.set_strict_mode, False)
        output = self.base / "test.rrd"
        show(
            self.metrics.analyze(self.load()),
            reader,
            spawn=False,
            rrd=output,
            cameras=["wrist_left"],
        )
        self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
