"""Astro P2 V3 deployment asset and mjlab control configuration."""

import json
from pathlib import Path

import mujoco

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_XML = _ASSETS / "astro_p2_deploy_reduced.xml"
_CONTRACT = json.loads((_ASSETS / "p2_deploy_contract.json").read_text())
P2_JOINT_NAMES = tuple(_CONTRACT["joint_names"][:-1])
P2_FOOT_GEOMS = r"^(left|right)_foot_capsule_collision[1-8]$"

_DEFAULT_ANGLES = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
}


def load_p2_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(_XML))


def _p2_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML))


def make_p2_robot():
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    # Torque limits from the V3 source MJCF motor force ranges.
    efforts = (130, 130, 130, 130, 60, 45) * 2 + (130, 60, 50) + (30, 30, 30, 30, 30, 10, 10) * 2
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name,), stiffness=kp, damping=kd, effort_limit=effort,
        )
        for name, kp, kd, effort in zip(
            P2_JOINT_NAMES, _CONTRACT["kp"][:-1], _CONTRACT["kd"][:-1], efforts, strict=True,
        )
    )
    cfg = EntityCfg(
        spec_fn=_p2_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.645), joint_pos={name: _DEFAULT_ANGLES.get(name, 0.0) for name in P2_JOINT_NAMES},
        ),
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.9),
    )
    # Same normalized-action convention as SuperMimic's G1.
    scales = {a.target_names_expr[0]: 0.25 * a.effort_limit / a.stiffness for a in actuators}
    return cfg, scales
