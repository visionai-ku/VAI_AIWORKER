"""Mirror a connected UR5e and plan a parallel DRY RUN from each real clutch."""

from .controllers import TeleopController


class ConnectedPreview:
    def __init__(self):
        self.payload = None
        self.comparison_status = None
        self._planner = None
        self._q = None
        self._speed_feedback = None

    def read_feedback(self, arm):
        """Capture feedback before this tick sends its motion command."""
        if arm.dry_run:
            self._planner = None
        self.payload = arm.backend.payload()
        self._speed_feedback = None if arm.dry_run else arm.backend.tcp_speed_feedback()
        # In joint mode the controller already reads q when it reads the TCP.
        self._q = None if arm.joint_space else arm.backend.joints()

    def plan_comparison(self, arm, sample, now, *, scale, values):
        if arm.dry_run:
            return
        plan = self._planner
        if plan is None or (plan.command_mode, plan.control_law) != (arm.command_mode, arm.control_law):
            self._planner = plan = TeleopController(None, dry_run=True, command_mode=arm.command_mode,
                                                   control_law=arm.control_law)
        # Replaying the robot's target here would inherit its measured-position lead
        # limit. Give the DRY controller the original sample and the same tick time.
        plan.plan(sample, now, measured=arm.measured, measured_joints=arm.measured_joints,
                  scale=scale, values=values)

    def render(self, view, arm, width):
        q = arm.measured_joints if arm.joint_space else self._q
        plan = self._planner if not arm.dry_run and arm.active else None
        gains = None
        if arm.active and arm.control_law == "impedance":
            gains = arm.backend.get_impedance_gains("joint" if arm.joint_space else "eef")
        stamp, twist = self._speed_feedback or (None, None)
        view.update(q=q, measured_tcp=arm.measured, dry_run=arm.dry_run,
                    target=arm.target if arm.dry_run else None,
                    joint_target=arm.joint_target if arm.dry_run else None,
                    gripper_width_mm=width, preview_gains=gains if arm.dry_run else None,
                    preview_delta=arm.dry_run and gains is not None and arm.target_from_device,
                    payload=self.payload, actual_tcp_speed=twist, feedback_timestamp=stamp,
                    comparison_target=plan.target if plan else None,
                    comparison_joints=plan.joint_target if plan else None,
                    comparison_gains=gains if plan else None,
                    comparison_delta=plan.target_from_device if plan else False)
        self.comparison_status = view.speed_comparison()

    def panel_status(self):
        payload = None
        if self.payload is not None:
            payload = {"mass_kg": self.payload.mass_kg, "cog_m": self.payload.cog_m}
        return {"tool_payload": payload, "speed_comparison": self.comparison_status}
