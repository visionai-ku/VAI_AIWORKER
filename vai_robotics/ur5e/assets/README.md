# UR5e MuJoCo model

`scene.xml`, `ur5e.xml`, and the 20 OBJ meshes in `assets/` are unmodified
files from the [UR5e model in MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/universal_robots_ur5e).
The model and meshes are distributed under the included [BSD 3-Clause license](LICENSE).

Load `scene.xml` to use the arm and floor together. The six joint-position
actuators accept radians. The `home` keyframe provides matching initial joint
positions and control targets, and `attachment_site` marks the tool mounting
frame.
