# FFW SG2 VR Teleoperation

MetaQuest3 wrist tracking to FFW SG2 arm IK.

This folder intentionally leaves hand/finger retargeting empty. SG2 uses grippers, so the first pass only solves arm IK from the left/right Quest wrist poses. Gripper joints are not published by the real teleop script.

## Files

- `package/utils.py`: SG2 wrist frame helpers, Quest packet parsing, attach state, IK target helpers, and real robot message utilities.
- `xml/ffw_sg2.xml`: Local SG2 MJCF copied from the shared asset tree.
- `xml/floor_simple_white.xml`: Local floor MJCF used by the SG2 scenes.
- `xml/mesh/`: Local SG2 mesh assets required by the MJCF files.
- `xml/ffw_sg2_body_added_vis.xml`: Orange `vis_` SG2 command-visualization model for the render worker.
- `sim_notebook/sim_MetaQuest3_sg2_teleop.ipynb`: MuJoCo preview loop for Quest wrist to SG2 arm IK.
- `real_notebook/real_render_MetaQuest3_sg2.ipynb`: Render window for the real teleop publisher.
- `real_notebook/real_MetaQuest3_sg2_teleop.ipynb`: Real robot command publisher. Robot command is OFF by default in the Qt widget.

## Run

Open and run:

- Sim preview only: `sim_notebook/sim_MetaQuest3_sg2_teleop.ipynb`
- Real render window: `real_notebook/real_render_MetaQuest3_sg2.ipynb`
- Real command publisher: `real_notebook/real_MetaQuest3_sg2_teleop.ipynb`

For real teleoperation, run the render notebook in one kernel/window and the
command publisher in another. The command publisher sends render data on
`tcp://*:5556`; the render notebook subscribes to `tcp://127.0.0.1:5556`.

The notebooks use the local XML and mesh files under `xml/`; no shared asset
folder is required.
