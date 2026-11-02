"""Build component configs from config_so101.yaml, the single source of settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .learning.offline_train import OfflineConfig
from .learning.online_builder import OnlineConfig
from .learning.promotion import PromotionConfig
from .learning.replay import CreditLedger, ReplayConfig
from .planning.joint_cem import CemConfig


@dataclass(frozen=True)
class Settings:
    online: OnlineConfig
    offline: OfflineConfig
    promotion: PromotionConfig
    replay: ReplayConfig
    cem: CemConfig
    raw: dict

    def credits(self) -> CreditLedger:
        online = self.raw["online"]
        return CreditLedger(
            windows_per_credit=online["new_unique_windows_per_update_credit"],
            minimum_windows=online["minimum_valid_windows"],
            max_backlog=online["maximum_credit_backlog"],
            max_updates_per_s=online["maximum_updates_per_second"],
            max_duty=online["maximum_operational_duty_fraction"],
        )


def load_settings(path: str | os.PathLike = Path(__file__).resolve().parents[1] / "config_so101.yaml") -> Settings:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    online, visual = raw["online"], raw["offline"]["visual"]
    promotion, replay, planning = raw["promotion"], raw["replay"], raw["planning"]
    online_config = OnlineConfig(
        lora_lr=online["lora_lr"],
        residual_lr=online["residual_lr"],
        betas=tuple(online["betas"]),
        eps=online["eps"],
        weight_decay=online["weight_decay"],
        grad_clip_norm=online["grad_clip_norm"],
        warmup_updates=online["warmup_updates"],
        horizon=online["initial_rollout_horizon"],
        weight_ar=online["visual_loss_weights"]["autonomous_rollout"],
        weight_anchor=online["visual_loss_weights"]["compatible_functional_anchor"],
        weight_prox=online["visual_loss_weights"]["proximal"],
    )
    offline_config = OfflineConfig(
        sequence_frames=visual["sequence_frames"],
        effective_batch=visual["effective_batch"],
        microbatch=visual["microbatch_initial"],
        max_updates=visual["max_updates"],
        warmup_updates=visual["warmup_updates"],
        curriculum=tuple((c["updates"], c["horizon"]) for c in visual["rollout_curriculum"]),
        gamma=visual["rollout_discount"],
        lora_lr=visual["lora_lr"],
        residual_lr=visual["residual_lr"],
        final_lr_fraction=visual["final_lr_fraction"],
        betas=tuple(visual["betas"]),
        eps=visual["eps"],
        weight_decay=visual["weight_decay"],
        grad_clip_norm=visual["grad_clip_norm"],
        weight_realized=visual["loss_weights"]["realized_motion"],
        weight_cmd_tf=visual["loss_weights"]["command_tf"],
        weight_rollout=visual["loss_weights"]["autonomous_rollout"],
        validation_interval=raw["offline"]["compact"]["validation_interval_updates"],
        patience_evaluations=raw["offline"]["compact"]["early_stop_patience_evaluations"],
        seed=raw["seed"],
    )
    promotion_config = PromotionConfig(
        minimum_snapshot_interval_s=promotion["minimum_snapshot_interval_s"],
        future_macro_transitions=promotion["future_validation_macro_transitions"],
        temporal_block_s=promotion["temporal_block_seconds"],
        macro_hz=raw["timing"]["macro_hz"],
        anchor_windows=promotion["anchor_windows"],
        minimum_relative_improvement=promotion["minimum_relative_visual_improvement"],
        bootstrap_confidence=promotion["bootstrap_confidence"],
        bootstrap_replicates=promotion["bootstrap_replicates"],
        maximum_joint_regression=promotion["maximum_joint_regression_fraction"],
        maximum_anchor_regression=promotion["maximum_anchor_group_regression_fraction"],
        maximum_task_regression=promotion["maximum_task_probe_regression_fraction"],
        rollback_regression=promotion["rollback_prequential_regression_fraction"],
        rollback_window_s=promotion["rollback_window_seconds"],
        retain_validated=promotion["retain_validated_versions"],
    )
    replay_config = ReplayConfig(
        recent_windows=replay["recent_windows"],
        reservoir_windows=replay["per_regime_reservoir_windows"],
        anchor_windows=replay["immutable_anchor_windows_total"],
        quarantine_windows=replay["quarantine_windows"],
        recent_fraction=replay["recent_fraction"],
        regime_fraction=replay["same_regime_fraction"],
        anchor_fraction=replay["compatible_anchor_fraction"],
    )
    cem_config = CemConfig(
        horizon=planning["horizon_macro_steps"],
        candidates=planning["candidates"],
        iterations=planning["iterations"],
        elites=planning["elites"],
        initial_std=planning["initial_std_normalized"],
        std_floor=planning["std_floor_normalized"],
        mean_momentum=planning["mean_old_momentum"],
        std_momentum=planning["std_old_momentum"],
        movement_weight=planning["movement_cost_weight"],
        smoothness_weight=planning["smoothness_cost_weight"],
        view_weights=tuple(planning["view_goal_weights"]),
        microbatch=planning["microbatch_initial"],
    )
    return Settings(online_config, offline_config, promotion_config, replay_config, cem_config, raw)


def compact_config(settings: Settings, joint_sensor_std=None, residual_bound=None, view_ids=None):
    """CompactConfig from the YAML `belief` section, cross-checked against its own dimensions.

    joint_sensor_std and residual_bound are measured hardware values. They come from
    the arguments, else from robot.joint_sensor_std_normalized and
    robot.residual_joint_bound_normalized, and are refused when absent rather than
    defaulted (spec 6.4, 6.5).
    """
    from .belief.compact import CompactConfig
    from .belief.imm import JitterPolicy

    raw = settings.raw
    belief, robot, sketch = raw["belief"], raw["robot"], raw["representation"]["sketch"]

    def six(value, name):
        if value is None:
            raise ValueError(f"{name} is a measured value; pass it or set it in config_so101.yaml")
        values = (float(value),) if isinstance(value, (int, float)) else tuple(float(v) for v in value)
        if len(values) == 1:
            values = values * 6
        if len(values) != 6:
            raise ValueError(f"{name} needs one or six values")
        return values

    views = tuple(view_ids) if view_ids is not None else tuple(raw["representation"].get("checkpoint_view_ids") or (6, 1))
    variance = belief["process_variance_per_macro"]
    bounds = belief["process_variance_bounds"]
    config = CompactConfig(
        joint_sensor_std=six(joint_sensor_std if joint_sensor_std is not None else robot.get("joint_sensor_std_normalized"), "joint_sensor_std"),
        residual_bound=six(residual_bound if residual_bound is not None else robot.get("residual_joint_bound_normalized"), "residual_bound"),
        view_ids=views,
        visual_per_view=sketch["pca_dims_per_view"],
        fast_dims=belief["fast_dims"],
        slow_dims=belief["slow_dims"],
        recurrent_hidden=belief["recurrent_hidden"],
        command_hidden=belief["command_hidden"],
        regime_dim=belief["regime_embedding_dim"],
        max_regimes=belief["max_regimes"],
        transition_hidden=tuple(belief["mean_transition_hidden"]),
        observation_hidden=tuple(belief["observation_hidden"]),
        nominal_macro_dt_s=raw["timing"]["nominal_macro_dt_s"],
        fast_decay_s=belief["fast_initial_decay_seconds"],
        fast_coefficient=belief["fast_residual_coefficient_per_macro"],
        observation_matrix_bound=belief["observation_matrix_initial_bound"],
        process_variance_init=(variance["joint"], variance["fast"], variance["slow"]),
        joint_fast_variance_bounds=tuple(bounds["joint_fast"]),
        slow_variance_bounds=tuple(bounds["slow"]),
        visual_std_init=belief["visual_initial_observation_std"],
        initial_latent_variance=belief["initial_fast_variance"],
        mode_residence_s=belief["mode_residence_prior_seconds"],
        correction_rank=belief["online_transition_correction"]["rank"],
        jitter=JitterPolicy(belief["numerical_jitter_initial"], belief["numerical_jitter_max"]),
    )
    checks = {
        "latent_state_dim": config.state_dim,
        "joint_dims": 6,
        "command_input_dim": 8,
    }
    if len(views) == 2:
        checks["observation_dim_two_views"] = config.obs_dim
        checks["recurrent_input_dim"] = config.gru_input_dim
    for key, value in checks.items():
        if belief[key] != value:
            raise ValueError(f"config_so101.yaml belief.{key}={belief[key]} disagrees with the derived {value}")
    if belief["initial_fast_variance"] != belief["initial_slow_variance"]:
        raise ValueError("this implementation uses one initial latent variance; make fast and slow equal")
    return config
