"""Per-bundle metrics collection for benchmark scenarios."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dagzoo.math import (
    coerce_optional_finite_float as _coerce_optional_finite_float,
)
from dagzoo.types import DatasetBundle

_NOISE_FAMILY_MIXTURE = "mixture"
_NOISE_MIXTURE_COMPONENTS = {"gaussian", "laplace", "student_t"}


def _matrix_cell_count(matrix: Any) -> int:
    """Return cell count for a rank-2 matrix-like payload."""

    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) < 2:
        return 0
    try:
        n_rows = max(0, int(shape[0]))
        n_cols = max(0, int(shape[1]))
    except (TypeError, ValueError):
        return 0
    return n_rows * n_cols


def _mean_or_none(*, total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return float(total / float(count))


@dataclass(slots=True)
class _BundleMetricsCollector:
    """Collect benchmark scenario metrics from emitted bundles."""

    expected_noise_family_requested: str
    datasets_seen: int = 0
    attempts_total: int = 0
    retry_dataset_count: int = 0
    filter_attempts_total: int = 0
    filter_rejections_total: int = 0
    filter_retry_dataset_count: int = 0
    missingness_bundles_with_metadata: int = 0
    missing_cells: int = 0
    total_cells: int = 0
    shift_bundles_with_metadata: int = 0
    shift_enabled_true: int = 0
    graph_edge_density_sum: float = 0.0
    graph_edge_density_count: int = 0
    edge_odds_multiplier_sum: float = 0.0
    edge_odds_multiplier_count: int = 0
    mechanism_nonlinear_mass_sum: float = 0.0
    mechanism_nonlinear_mass_count: int = 0
    noise_variance_multiplier_sum: float = 0.0
    noise_variance_multiplier_count: int = 0
    noise_bundles_with_metadata: int = 0
    noise_bundles_with_valid_metadata: int = 0
    noise_sampled_family_counts: dict[str, int] = field(default_factory=dict)
    noise_invalid_reason_counts: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _coerce_non_negative_int(value: Any, *, default: int) -> int:
        if isinstance(value, bool):
            return int(default)
        if isinstance(value, int):
            return int(max(0, value))
        if isinstance(value, float):
            if not math.isfinite(value):
                return int(default)
            return int(max(0, int(value)))
        if isinstance(value, str):
            normalized = value.strip()
            signless = normalized[1:] if normalized.startswith(("+", "-")) else normalized
            if not signless.isdigit():
                return int(default)
            return int(max(0, int(normalized)))
        return int(default)

    def update(self, bundle: DatasetBundle) -> None:
        """Collect metrics for one emitted bundle."""

        self.datasets_seen += 1
        self._update_pressure(bundle)
        self._update_missingness(bundle)
        self._update_shift(bundle)
        self._update_noise(bundle)

    def _update_pressure(self, bundle: DatasetBundle) -> None:
        metadata = bundle.metadata
        attempts_payload = metadata.get("generation_attempts")

        total_attempts = 1
        filter_attempts = 0
        filter_rejections = 0

        if isinstance(attempts_payload, dict):
            total_attempts = self._coerce_non_negative_int(
                attempts_payload.get("total_attempts"),
                default=1,
            )
            filter_attempts = self._coerce_non_negative_int(
                attempts_payload.get("filter_attempts"),
                default=0,
            )
            filter_rejections = self._coerce_non_negative_int(
                attempts_payload.get("filter_rejections"),
                default=0,
            )
        else:
            attempt_used = self._coerce_non_negative_int(metadata.get("attempt_used"), default=0)
            total_attempts = max(1, attempt_used + 1)
            filter_payload = metadata.get("filter")
            if isinstance(filter_payload, dict) and bool(filter_payload.get("enabled")):
                filter_attempts = 1
                if not bool(filter_payload.get("accepted", False)):
                    filter_rejections = 1

        total_attempts = max(1, int(total_attempts))
        filter_attempts = max(0, int(filter_attempts))
        filter_rejections = max(0, min(int(filter_rejections), filter_attempts))

        self.attempts_total += total_attempts
        if total_attempts > 1:
            self.retry_dataset_count += 1

        self.filter_attempts_total += filter_attempts
        self.filter_rejections_total += filter_rejections
        if filter_rejections > 0:
            self.filter_retry_dataset_count += 1

    def _update_missingness(self, bundle: DatasetBundle) -> None:
        payload = bundle.metadata.get("missingness")
        if not isinstance(payload, dict):
            return

        total_cells = _matrix_cell_count(bundle.X_train) + _matrix_cell_count(bundle.X_test)
        if total_cells <= 0:
            return

        missing_count_raw = payload.get("missing_count_overall")
        if isinstance(missing_count_raw, bool) or not isinstance(missing_count_raw, (int, float)):
            return
        missing_count = int(max(0, min(total_cells, int(missing_count_raw))))

        self.missingness_bundles_with_metadata += 1
        self.total_cells += total_cells
        self.missing_cells += missing_count

    def _update_shift(self, bundle: DatasetBundle) -> None:
        shift_payload = bundle.metadata.get("shift")
        if not isinstance(shift_payload, dict):
            return

        self.shift_bundles_with_metadata += 1
        if shift_payload.get("enabled") is True:
            self.shift_enabled_true += 1

        graph_edge_density = _coerce_optional_finite_float(
            bundle.metadata.get("graph_edge_density")
        )
        if graph_edge_density is not None:
            self.graph_edge_density_sum += float(graph_edge_density)
            self.graph_edge_density_count += 1

        edge_odds_multiplier = _coerce_optional_finite_float(
            shift_payload.get("edge_odds_multiplier")
        )
        if edge_odds_multiplier is not None:
            self.edge_odds_multiplier_sum += float(edge_odds_multiplier)
            self.edge_odds_multiplier_count += 1

        mechanism_nonlinear_mass = _coerce_optional_finite_float(
            shift_payload.get("mechanism_nonlinear_mass")
        )
        if mechanism_nonlinear_mass is not None:
            self.mechanism_nonlinear_mass_sum += float(mechanism_nonlinear_mass)
            self.mechanism_nonlinear_mass_count += 1

        noise_variance_multiplier = _coerce_optional_finite_float(
            shift_payload.get("noise_variance_multiplier")
        )
        if noise_variance_multiplier is not None:
            self.noise_variance_multiplier_sum += float(noise_variance_multiplier)
            self.noise_variance_multiplier_count += 1

    def _update_noise(self, bundle: DatasetBundle) -> None:
        payload = bundle.metadata.get("noise_distribution")
        if not isinstance(payload, dict):
            return

        self.noise_bundles_with_metadata += 1
        valid, sampled_family, reason = self._validate_noise_payload(payload)
        if not valid:
            if reason is not None:
                self.noise_invalid_reason_counts[reason] = (
                    int(self.noise_invalid_reason_counts.get(reason, 0)) + 1
                )
            return

        self.noise_bundles_with_valid_metadata += 1
        self.noise_sampled_family_counts[sampled_family] = (
            int(self.noise_sampled_family_counts.get(sampled_family, 0)) + 1
        )

    def _validate_noise_payload(self, payload: dict[str, Any]) -> tuple[bool, str, str | None]:
        expected = str(self.expected_noise_family_requested).strip().lower()
        family_requested_raw = payload.get("family_requested")
        family_sampled_raw = payload.get("family_sampled")
        sampling_strategy_raw = payload.get("sampling_strategy")
        scale_raw = payload.get("base_scale")
        student_t_df_raw = payload.get("student_t_df")
        mixture_weights_raw = payload.get("mixture_weights")

        if not isinstance(family_requested_raw, str):
            return False, "", "family_requested_type"
        family_requested = family_requested_raw.strip().lower()
        if family_requested != expected:
            return False, "", "family_requested_mismatch"

        if not isinstance(family_sampled_raw, str):
            return False, "", "family_sampled_type"
        family_sampled = family_sampled_raw.strip().lower()

        if not isinstance(sampling_strategy_raw, str):
            return False, "", "sampling_strategy_type"
        if sampling_strategy_raw.strip().lower() != "dataset_level":
            return False, "", "sampling_strategy_value"

        scale = _coerce_optional_finite_float(scale_raw)
        student_t_df = _coerce_optional_finite_float(student_t_df_raw)
        if scale is None or scale <= 0.0:
            return False, "", "base_scale_value"
        if student_t_df is None or student_t_df <= 2.0:
            return False, "", "student_t_df_value"

        if family_requested != _NOISE_FAMILY_MIXTURE:
            if family_sampled != family_requested:
                return False, "", "family_sampled_mismatch"
            if mixture_weights_raw is not None:
                return False, "", "mixture_weights_unexpected"
            return True, family_sampled, None

        if family_sampled not in _NOISE_MIXTURE_COMPONENTS:
            return False, "", "family_sampled_invalid_for_mixture"
        if not isinstance(mixture_weights_raw, dict):
            return False, "", "mixture_weights_type"

        total_weight = 0.0
        for key_raw, value_raw in mixture_weights_raw.items():
            if not isinstance(key_raw, str):
                return False, "", "mixture_weights_key_type"
            key = key_raw.strip().lower()
            if key not in _NOISE_MIXTURE_COMPONENTS:
                return False, "", "mixture_weights_key_value"
            value = _coerce_optional_finite_float(value_raw)
            if value is None or value < 0.0:
                return False, "", "mixture_weights_value"
            total_weight += float(value)
        if total_weight <= 0.0:
            return False, "", "mixture_weights_total_nonpositive"
        if not math.isclose(total_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            return False, "", "mixture_weights_total_not_one"
        return True, family_sampled, None


def build_pressure_metrics(collector: _BundleMetricsCollector) -> dict[str, Any]:
    datasets = int(collector.datasets_seen)
    attempts = int(collector.attempts_total)
    filter_attempts = int(collector.filter_attempts_total)
    filter_rejections = int(collector.filter_rejections_total)
    retry_datasets = int(collector.retry_dataset_count)
    filter_retry_datasets = int(collector.filter_retry_dataset_count)

    attempts_per_dataset = float(attempts) / float(datasets) if datasets > 0 else 0.0
    retry_dataset_rate = float(retry_datasets) / float(datasets) if datasets > 0 else None
    filter_retry_dataset_rate = (
        float(filter_retry_datasets) / float(datasets)
        if datasets > 0 and filter_attempts > 0
        else None
    )
    filter_rejection_rate_attempt_level = (
        float(filter_rejections) / float(filter_attempts) if filter_attempts > 0 else None
    )
    return {
        "datasets_seen": datasets,
        "attempts_total": attempts,
        "attempts_per_dataset_mean": float(attempts_per_dataset),
        "retry_dataset_count": retry_datasets,
        "retry_dataset_rate": retry_dataset_rate,
        "filter_attempts_total": filter_attempts,
        "filter_rejections_total": filter_rejections,
        "filter_rejection_rate_attempt_level": filter_rejection_rate_attempt_level,
        "filter_retry_dataset_count": filter_retry_datasets,
        "filter_retry_dataset_rate": filter_retry_dataset_rate,
    }


def build_missingness_metrics(
    collector: _BundleMetricsCollector,
    *,
    target_rate: float,
) -> dict[str, Any]:
    coverage_rate = (
        float(collector.missingness_bundles_with_metadata) / float(collector.datasets_seen)
        if collector.datasets_seen > 0
        else 0.0
    )
    realized_rate = (
        float(collector.missing_cells) / float(collector.total_cells)
        if collector.total_cells > 0
        else 0.0
    )
    return {
        "metadata_coverage_rate": float(coverage_rate),
        "realized_rate_overall": float(realized_rate),
        "rate_abs_error": float(abs(realized_rate - float(target_rate))),
        "target_rate": float(target_rate),
    }


def build_shift_metrics(collector: _BundleMetricsCollector) -> dict[str, Any]:
    metadata_coverage_rate = (
        float(collector.shift_bundles_with_metadata) / float(collector.datasets_seen)
        if collector.datasets_seen > 0
        else 0.0
    )
    shift_enabled_coverage_rate = (
        float(collector.shift_enabled_true) / float(collector.datasets_seen)
        if collector.datasets_seen > 0
        else 0.0
    )
    return {
        "metadata_coverage_rate": float(metadata_coverage_rate),
        "shift_enabled_coverage_rate": float(shift_enabled_coverage_rate),
        "mean_graph_edge_density": _mean_or_none(
            total=collector.graph_edge_density_sum,
            count=collector.graph_edge_density_count,
        ),
        "mean_edge_odds_multiplier": _mean_or_none(
            total=collector.edge_odds_multiplier_sum,
            count=collector.edge_odds_multiplier_count,
        ),
        "mean_mechanism_nonlinear_mass": _mean_or_none(
            total=collector.mechanism_nonlinear_mass_sum,
            count=collector.mechanism_nonlinear_mass_count,
        ),
        "mean_noise_variance_multiplier": _mean_or_none(
            total=collector.noise_variance_multiplier_sum,
            count=collector.noise_variance_multiplier_count,
        ),
    }


def build_noise_metrics(collector: _BundleMetricsCollector) -> dict[str, Any]:
    metadata_coverage_rate = (
        float(collector.noise_bundles_with_metadata) / float(collector.datasets_seen)
        if collector.datasets_seen > 0
        else 0.0
    )
    metadata_valid_rate = (
        float(collector.noise_bundles_with_valid_metadata) / float(collector.datasets_seen)
        if collector.datasets_seen > 0
        else 0.0
    )
    return {
        "metadata_coverage_rate": float(metadata_coverage_rate),
        "metadata_valid_rate": float(metadata_valid_rate),
        "valid_metadata_count": int(collector.noise_bundles_with_valid_metadata),
        "sampled_family_counts": {
            key: int(collector.noise_sampled_family_counts[key])
            for key in sorted(collector.noise_sampled_family_counts)
        },
        "invalid_reason_counts": {
            key: int(collector.noise_invalid_reason_counts[key])
            for key in sorted(collector.noise_invalid_reason_counts)
        },
    }


def build_filtering_metrics(
    collector: _BundleMetricsCollector,
    *,
    filter_stage_measurement: Any,
) -> dict[str, Any]:
    pressure = build_pressure_metrics(collector)
    datasets_measured = int(getattr(filter_stage_measurement, "filter_accepted_datasets", 0)) + int(
        getattr(filter_stage_measurement, "filter_rejected_datasets", 0)
    )
    acceptance_rate = (
        float(getattr(filter_stage_measurement, "filter_accepted_datasets", 0))
        / float(datasets_measured)
        if datasets_measured > 0
        else None
    )
    rejection_rate = (
        float(getattr(filter_stage_measurement, "filter_rejected_datasets", 0))
        / float(datasets_measured)
        if datasets_measured > 0
        else None
    )
    accepted_dpm = (
        float(getattr(filter_stage_measurement, "datasets_per_minute", 0.0))
        * float(acceptance_rate)
        if acceptance_rate is not None
        else None
    )
    return {
        **pressure,
        "datasets_per_minute": float(getattr(filter_stage_measurement, "datasets_per_minute", 0.0)),
        "accepted_datasets_per_minute": accepted_dpm,
        "filter_attempts_total": int(getattr(filter_stage_measurement, "filter_attempts_total", 0)),
        "filter_rejections_total": int(
            getattr(filter_stage_measurement, "filter_rejections_total", 0)
        ),
        "accepted_datasets": int(getattr(filter_stage_measurement, "filter_accepted_datasets", 0)),
        "rejected_datasets": int(getattr(filter_stage_measurement, "filter_rejected_datasets", 0)),
        "acceptance_rate_dataset_level": acceptance_rate,
        "rejection_rate_dataset_level": rejection_rate,
        "elapsed_seconds": float(getattr(filter_stage_measurement, "elapsed_seconds", 0.0)),
        "cpu_time_seconds": float(getattr(filter_stage_measurement, "cpu_time_seconds", 0.0)),
    }


def _compose_bundle_callback(
    *,
    diagnostics_aggregator: Any,
    collector: _BundleMetricsCollector | None,
) -> Callable[[DatasetBundle], None] | None:
    """Compose optional per-bundle collectors into one callback."""

    if diagnostics_aggregator is None and collector is None:
        return None

    def _on_bundle(bundle: DatasetBundle) -> None:
        if diagnostics_aggregator is not None:
            diagnostics_aggregator.update_bundle(bundle)
        if collector is not None:
            collector.update(bundle)

    return _on_bundle
