# RoboJEPA + persistent test-time learning and recurrent belief for SO-101

Prepared 2026-11-02. Status: proposed research/engineering specification; not a trained or robot-validated system.

- [Full mathematical and implementation specification](SPECIFICATION.md)
- [Concrete starting configuration](config_so101.yaml)
- [Source audit and distinctions from the papers](SOURCE_AUDIT.md)
- [Build status and training guide](TRAINING_GUIDE.md) for the TTT layers in [`robojepa_ttt/`](robojepa_ttt/)

The core recommendation is a frozen RoboJEPA representation, a command-to-motion model for SO-101, a recurrent probabilistic state estimator, and persistent replay-based adaptation of small predictor modules. State inference and parameter learning have separate clocks and lifetimes. Robot I/O, command expiry, limits, and stopping remain local even when inference and training use cloud GPUs.

Read the specification before treating the YAML as an implementation configuration. Unmeasured hardware limits are deliberately unset. All proposed numerical defaults require validation on the actual arm, cameras, network, and cloud instance. Numerical checks of the specification's invariants are described in the specification but are not yet included in this repository.

## Licensing and vendored code

The code in [`robojepa_ttt/`](robojepa_ttt/), [`scripts/`](scripts/) and [`tests/`](tests/) is under the MIT License in [`LICENSE`](LICENSE).
[`third_party/robo_jepa/`](third_party/robo_jepa/) is a copy of [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa) at commit `f93106f5`,
under its own [CC BY-NC-SA 4.0 license](third_party/robo_jepa/LICENSE), which does not allow commercial use.
The Python files and paper copies in [`sources/`](sources/) are taken from upstream and from the papers, and keep their original licenses and terms.
