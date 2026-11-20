# UR5e MuJoCo model

`scene.xml`, the arm description in `ur5e.xml`, and the 20 OBJ meshes in
`assets/` come from MuJoCo Menagerie. The wrist now includes an RG2 visual
model. [UR5e source](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/universal_robots_ur5e)
· [BSD 3-Clause license](LICENSE)

Load `scene.xml` to use the arm and floor together. The six joint-position
actuators accept radians. The `home` keyframe provides matching initial joint
positions and control targets, and `attachment_site` marks the tool mounting
frame.

## OnRobot RG2

The four STL meshes in `assets/rg2/` come from the Osaka University Harada
Laboratory RG2 description at commit `46fae3557dd3aff678adf6b3320ec1338493d84a`.
`rg2_visual.xml` adapts its joint transforms to MJCF at metre scale. The base
is flush with `attachment_site`, with the fingers extending along tool +Z.
The standalone model uses massless visual links with contact disabled. In a
robot-connected session, MuJoCo assigns the active RTDE payload mass to
`rg2_base` at the reported flange-local CoG. This point-mass approximation
includes its moment arm and omits the tool's intrinsic rotational inertia.
The flange-relative visual mount places all RG2 meshes and pad sites together.
With zero offsets the gripper base is flush with the flange, the fingers extend
along flange +Z, and the jaws open along flange Y. The panel accepts XYZ in
millimetres and roll, pitch and yaw about fixed flange axes in degrees.
The TCP and the load CoG use their independently configured frames.
[RG2 meshes and Xacro](https://github.com/UOsaka-Harada-Laboratory/onrobot/tree/46fae3557dd3aff678adf6b3320ec1338493d84a/onrobot_rg_description)
· [MIT license](assets/rg2/LICENSE)

The visible aperture is measured between the inside faces of the supplied
finger pads. The upstream linkage reaches about 104 mm between those faces
at its open stop. The viewer displays this fully open pose for the nominal
110 mm command. Intermediate widths below 104 mm match the measured gap.
OnRobot likewise distinguishes the compensated device width (0–110 mm) from
the contact range of a particular pair of fingertips.
[RG2 standard device profile](https://github.com/OnRobot/onrobot-ros2/blob/1b24536e4ae58b127025a4336fd1f06109c54801/onrobot_gripper_description/config/finger_profiles/rg2_standard.yaml)
· [Stock inward finger contact profile](https://github.com/OnRobot/onrobot-ros2/blob/1b24536e4ae58b127025a4336fd1f06109c54801/onrobot_gripper_description/config/planning_profiles/rg2.yaml)
