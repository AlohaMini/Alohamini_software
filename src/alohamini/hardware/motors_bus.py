"""Calibrated motor coordinates shared by robot adapters and device transports.

Mapping never performs I/O. Normalized requests become physical targets before
controller supervision; only the supervised physical targets may be written.
"""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from alohamini.calibration.encoder import EncoderCalibration, HostPositionUnits
from alohamini.hardware.feedback import FeedbackBatch


class MotorsBus:
    """Own installed position calibration and explicitly declared action units.

    Physical-only/calibration access may omit position_units. Normalized access
    then fails instead of inferring a scale from joint names or vector sizes.
    Rebuild the bus after changing installed calibration or action coordinates.
    """

    def __init__(
        self,
        *,
        position_calibrations: Mapping[str, EncoderCalibration],
        position_units: Mapping[str, HostPositionUnits] | None = None,
    ) -> None:
        self._positions = dict(position_calibrations)
        if any(not isinstance(c, EncoderCalibration) for c in self._positions.values()):
            raise TypeError("Expected EncoderCalibration")
        self._position_units = dict(position_units or {})
        if position_units is not None and self._position_units.keys() != self._positions.keys():
            raise ValueError("Position units must cover every calibrated position motor")
        for name, units in self._position_units.items():
            if not isinstance(units, HostPositionUnits):
                raise TypeError("Expected HostPositionUnits")
            calibration = self._positions[name]
            # Preserve the Host's startup compatibility check in the owning bus.
            for tick in (units.range_min, units.range_max):
                calibration.position_to_tick(calibration.position_from_tick(tick))

    @property
    def position_calibrations(self) -> Mapping[str, EncoderCalibration]:
        return MappingProxyType(self._positions)

    @property
    def position_units(self) -> Mapping[str, HostPositionUnits]:
        return MappingProxyType(self._position_units)

    def prepare_position_targets(self, targets: Mapping[str, float]) -> dict[str, float]:
        """Saturate/quantize declared action values into physical joint targets.

        Preserve deployed truncation before converting to radians. Degree mode
        remains strict. No target is staged and no hardware is written here.
        """
        if not targets.keys() <= self._position_units.keys():
            raise ValueError("Unknown motor or missing declared position units")
        return {
            name: self._positions[name].position_from_tick(
                self._position_units[name].to_tick(value)
            )
            for name, value in targets.items()
        }

    def encode_positions(self, positions_rad: Mapping[str, float]) -> dict[str, int]:
        """Encode supervised physical targets; physical limits remain strict."""
        if not positions_rad.keys() <= self._positions.keys():
            raise ValueError("Unknown position-controlled motor")
        return {
            name: self._positions[name].position_to_tick(value)
            for name, value in positions_rad.items()
        }

    def normalize_positions(
        self,
        ticks: Mapping[str, int],
        *,
        normalizations: Mapping[str, str] | None = None,
    ) -> dict[str, float]:
        """Decode ticks into declared units, optionally selecting leader read units.

        A leader may read in the follower's declared normalization. This read-only
        selection does not change the bus's installed calibration or write units.
        """
        if not ticks.keys() <= self._position_units.keys():
            raise ValueError("Unknown motor or missing declared position units")
        if normalizations is not None and normalizations.keys() != ticks.keys():
            raise ValueError("Read normalization must match the selected motors")
        result = {}
        for name, tick in ticks.items():
            units = self._position_units[name]
            if normalizations is not None and normalizations[name] != units.normalization:
                units = replace(units, normalization=normalizations[name])
            result[name] = units.from_tick(tick)
        return result

    def position_targets_to_units(self, positions_rad: Mapping[str, float]) -> dict[str, float]:
        """Report commanded physical targets in declared units, not measured state."""
        return self.normalize_positions(self.encode_positions(positions_rad))

    def feedback_positions(self, batch: FeedbackBatch) -> dict[str, float]:
        """Map available position registers; missing/out-of-turn feedback stays absent.

        Session, freshness, packet errors and current protection are checked by
        Host supervision. Raw registers remain in the original feedback batch.
        """
        ticks = {}
        for name in self._position_units:
            sample = batch.samples.get(name)
            if sample is not None and "position_raw" in sample.registers:
                tick = sample.registers["position_raw"]
                if 0 <= tick < self._positions[name].ticks_per_revolution:
                    ticks[name] = tick
        return self.normalize_positions(ticks)
