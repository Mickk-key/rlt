"""Optional human teleop intervention for online-RL rollout (RLT Sec. V).

HG-DAgger-style takeover that reuses the SFT SpaceMouse path (deoxys
``input2action`` / ``SpaceMouse``). It is DISABLED unless explicitly enabled in
config AND the device actually opens, so importing/using it never changes
behaviour on a machine without a SpaceMouse.

Engagement is implicit: while the operator pushes the puck past a deadzone the
teleop action overrides the executed policy/reference action for that step;
releasing the puck hands control back automatically. The gripper/latch and the
``s``/``f``/``q`` reward-and-termination keys are untouched.

The teleop action is returned in the SAME physical EE-delta space (meters,
radians) as the VLA reference and the RL actor output, so it flows through the
identical ``DeoxysEnv.step`` path — the Phase-1 hard safety clamp still applies —
and is stored in the replay buffer consistently. Per the paper, on intervened
steps the human action also replaces the stored reference action (done by the
caller in ``actor_loop``).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SpaceMouseIntervention:
    """SpaceMouse-driven human takeover, producing physical EE-delta actions.

    ``input2action`` returns an action in *controller* units (what deoxys'
    OSC controller multiplies by its ``action_scale`` to realise a physical
    delta). We convert to physical units — ``phys[:3] = raw[:3] * ctrl_trans``,
    ``phys[3:6] = raw[3:6] * ctrl_rot`` — the exact inverse of the conversion in
    ``DeoxysEnv.step`` (``action_is_physical``), so the executed teleop delta and
    the value stored in replay match the actor/reference action space.
    """

    def __init__(
        self,
        *,
        ctrl_trans: float,
        ctrl_rot: float,
        controller_type: str = "OSC_POSE",
        vendor_id: int = 9583,
        product_id: int = 50746,
        trans_deadzone: float = 0.15,
        rot_deadzone: float = 0.15,
        action_dim: int = 7,
    ) -> None:
        self._ctrl_trans = float(ctrl_trans)
        self._ctrl_rot = float(ctrl_rot)
        self._controller_type = str(controller_type)
        self._vendor_id = int(vendor_id)
        self._product_id = int(product_id)
        self._trans_deadzone = float(trans_deadzone)
        self._rot_deadzone = float(rot_deadzone)
        self._action_dim = int(action_dim)
        self._device = None
        self._input2action = None
        self._acknowledge_reset = None

    @property
    def active(self) -> bool:
        return self._device is not None

    def start(self) -> bool:
        """Open the SpaceMouse. Returns True on success, False (disabled) otherwise."""
        try:
            from deoxys.utils.input_utils import input2action
            from deoxys.utils.io_devices import SpaceMouse

            from rlt.teleop.spacemouse_control import acknowledge_spacemouse_reset
        except Exception as exc:  # deoxys / HID not available
            logger.warning("teleop intervention disabled — SpaceMouse import failed: %s", exc)
            return False
        try:
            device = SpaceMouse(vendor_id=self._vendor_id, product_id=self._product_id)
            device.start_control()
        except Exception as exc:  # device not connected / no permission
            logger.warning("teleop intervention disabled — SpaceMouse open failed: %s", exc)
            return False
        self._device = device
        self._input2action = input2action
        self._acknowledge_reset = acknowledge_spacemouse_reset
        logger.info(
            "teleop intervention ENABLED (SpaceMouse vendor=%d product=%d; "
            "deadzone trans=%.3f rot=%.3f, controller=%s)",
            self._vendor_id,
            self._product_id,
            self._trans_deadzone,
            self._rot_deadzone,
            self._controller_type,
        )
        return True

    def poll(self) -> tuple[np.ndarray | None, bool]:
        """Return ``(physical_action[action_dim], engaged)``.

        ``engaged`` is False (and action None) when the operator is not pushing
        the puck past the deadzone, so the caller keeps the policy/reference
        action. Never raises: any device error degrades to no-intervention.
        """
        if self._device is None:
            return None, False
        try:
            action_raw, _grasp = self._input2action(
                device=self._device, controller_type=self._controller_type
            )
        except Exception as exc:
            logger.error("teleop poll failed — treating as no intervention: %s", exc)
            return None, False
        if action_raw is None:  # SpaceMouse RIGHT button = reset; re-enable, no takeover
            if self._acknowledge_reset is not None:
                try:
                    self._acknowledge_reset(self._device)
                except Exception:
                    pass
            return None, False

        raw = np.asarray(action_raw, dtype=np.float32).reshape(-1)
        trans_mag = float(np.linalg.norm(raw[:3])) if raw.shape[0] >= 3 else 0.0
        rot_mag = float(np.linalg.norm(raw[3:6])) if raw.shape[0] >= 6 else 0.0
        if trans_mag <= self._trans_deadzone and rot_mag <= self._rot_deadzone:
            return None, False

        phys = raw.copy()
        if phys.shape[0] >= 3:
            phys[:3] *= self._ctrl_trans
        if phys.shape[0] >= 6:
            phys[3:6] *= self._ctrl_rot
        if phys.shape[0] < self._action_dim:
            phys = np.concatenate(
                [phys, np.zeros(self._action_dim - phys.shape[0], dtype=np.float32)]
            )
        return phys[: self._action_dim].astype(np.float32), True

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None


def build_intervention(
    raw_cfg: dict,
    *,
    ctrl_trans: float,
    ctrl_rot: float,
    controller_type: str = "OSC_POSE",
    action_dim: int = 7,
) -> SpaceMouseIntervention | None:
    """Factory from config. Returns a *started* controller or None if disabled.

    Reads ``online_rl.teleop_intervention``; falls back to the SFT
    ``data_collection`` SpaceMouse ids. Off by default.
    """
    ti = (raw_cfg.get("online_rl", {}) or {}).get("teleop_intervention", {}) or {}
    if not bool(ti.get("enabled", False)):
        return None
    dc = raw_cfg.get("data_collection", {}) or {}
    interv = SpaceMouseIntervention(
        ctrl_trans=ctrl_trans,
        ctrl_rot=ctrl_rot,
        controller_type=controller_type,
        vendor_id=int(ti.get("spacemouse_vendor_id", dc.get("spacemouse_vendor_id", 9583))),
        product_id=int(ti.get("spacemouse_product_id", dc.get("spacemouse_product_id", 50746))),
        trans_deadzone=float(ti.get("trans_deadzone", 0.15)),
        rot_deadzone=float(ti.get("rot_deadzone", 0.15)),
        action_dim=int(action_dim),
    )
    if not interv.start():
        return None
    return interv
