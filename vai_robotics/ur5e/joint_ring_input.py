"""Route mouse joint drags and panel targets through the arm's existing clutch and limits."""

from .teleop_types import InputSample


class JointRingInput:
    def __init__(self):
        self.selection = None
        self._context = None
        self._epoch = 0
        self._ready = False
        self._active = None
        self._numeric = False
        self._physical_held = False

    def poll(self, view, arm, *, reset=False):
        context = (arm.command_mode, arm.control_law, arm.dry_run) if arm.joint_space else None
        if reset or context != self._context:
            self._context = context
            self._epoch += 1
            self._ready = self._numeric = False
            self._active = None
            self._physical_held = False
            if view is not None:
                view.set_joint_ring_epoch(self._epoch if context else 0)
        self.selection = view.joint_drag() if context is not None and view is not None else None

    @property
    def dragging(self):
        return self.selection is not None and self.selection.joint is not None

    def real_clutch(self, view, arm, physical):
        held = physical is not None and physical.enabled
        if held and not self._physical_held:
            # An idle report and even a press may already be in flight from the
            # viewer. A new epoch requires a mouse release observed after the clutch.
            self.poll(view, arm, reset=True)
        self._physical_held = held

    def arm_sample(self, physical, *, require_clutch, numeric_request):
        """The REAL mouse gesture must start after a confirmed held physical clutch."""
        clutch = physical is not None and physical.enabled
        permitted = not require_clutch or clutch
        drag = self.selection
        if not self.dragging:
            self._active = None
            self._ready = drag is not None and permitted
        elif not permitted:
            self._ready = False
            self._active = None
        elif self._ready:
            self._active = (drag.serial, drag.joint)
            self._ready = False

        if self.dragging:
            self._numeric = False
            if self._active == (drag.serial, drag.joint):
                return InputSample.stationary(enabled=True, joint_drag=drag)
        else:
            # Apply Target is a separate, clutched input in the Qt window.
            self._numeric = clutch and (numeric_request or self._numeric)
            if self._numeric:
                return InputSample.stationary(enabled=True)

        if drag is None and physical is None:
            return None
        return InputSample.stationary()
