# RoboJEPA + persistent test-time learning and recurrent belief for SO-101

Prepared 2026-11-02. Status: proposed research/engineering specification; not a trained or robot-validated system.

- [Full mathematical and implementation specification](SPECIFICATION.md)
- [Concrete starting configuration](config_so101.yaml)
- [Source audit and distinctions from the papers](SOURCE_AUDIT.md)
- [Numerical checks](verify_math.py) and [their results](verification_results.json)
- [Build status and training guide](TRAINING_GUIDE.md) for the TTT layers in `robojepa_ttt/`

The core recommendation is a frozen RoboJEPA representation, a command-to-motion model for SO-101, a recurrent probabilistic state estimator, and persistent replay-based adaptation of small predictor modules. State inference and parameter learning have separate clocks and lifetimes. Robot I/O, command expiry, limits, and stopping remain local even when inference and training use cloud GPUs.

Read the specification before treating the YAML as an implementation configuration. Unmeasured hardware limits are deliberately unset. All proposed numerical defaults require validation on the actual arm, cameras, network, and cloud instance. The numerical checks verify selected mathematical invariants, not robot performance.
