"""Generated fixed-base SG2 scene and a separate command-validation environment."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ri_motion_v5_package.mujoco_sim.mujoco_utils import merge_mjcfs


def build_fixed_quest_scene(robot_xml,floor_xml,output_xml):
    """Merge original assets with a generated robot copy whose base is fixed.

    Only files below ``.cache`` are written. Mesh paths in the copy are absolute
    so moving the generated scene does not change which original assets load.
    """
    robot_xml = Path(robot_xml).expanduser().resolve(strict=True)
    floor_xml = Path(floor_xml).expanduser().resolve(strict=True)
    output_xml = Path(output_xml).expanduser().resolve()
    robot_copy_xml = output_xml.with_name(output_xml.stem + ".robot.xml")
    if ".cache" not in output_xml.parts:
        raise ValueError("Generated fixed Quest scenes must be written below .cache.")
    if {output_xml,robot_copy_xml} & {robot_xml,floor_xml}:
        raise ValueError("Generated files must not overwrite source XML files.")

    tree = ET.parse(robot_xml,parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    base = root.find("./worldbody/body[@name='base_link']")
    if base is None:
        raise ValueError("SG2 source must contain a top-level base_link body.")
    free_joints = list(base.findall("freejoint")) + list(base.findall("joint[@type='free']"))
    if len(free_joints) != 1:
        raise ValueError("Expected exactly one floating base joint in the SG2 source.")
    base.remove(free_joints[0])
    p_base = np.fromstring(base.get("pos","0 0 0"),sep=" ")
    if p_base.shape != (3,) or not np.all(np.isfinite(p_base)):
        raise ValueError("Invalid SG2 base position.")
    p_base[2] = 0.01
    base.set("pos"," ".join(format(value,".12g") for value in p_base))

    compiler = root.find("compiler")
    for element in root.iter():
        filename = element.get("file")
        if filename is None or Path(filename).is_absolute():
            continue
        asset_dir = ""
        if compiler is not None and element.tag in ("mesh","texture","hfield"):
            kind = "texturedir" if element.tag == "texture" else "meshdir"
            asset_dir = compiler.get(kind,compiler.get("assetdir",""))
        element.set("file",str((robot_xml.parent / asset_dir / filename).resolve(strict=True)))
    output_xml.parent.mkdir(parents=True,exist_ok=True)
    tree.write(robot_copy_xml,encoding="utf-8",xml_declaration=True)
    return str(merge_mjcfs(
        mjcf_model_name     = "ffw_sg2_fixed_quest",
        included_mjcf_files = [str(floor_xml),str(robot_copy_xml)],
        output_xml_path     = str(output_xml),
        verbose             = False,
    ))


class SG2CommandGuard:
    """Check joint limits and sampled collision clearance without solving IK.

    ``q_start`` and ``q_end`` use ``joint_names`` order. The caller owns a
    dedicated ``env_check`` and synchronizes its uncommanded body state. The
    guard restores its qpos after each check; it never receives the main env.
    Sampling bounds joint increments, not continuous collision clearance.
    """

    def __init__(self,env_check,joint_names,safe_margin=0.005,sweep_step_rad=np.deg2rad(1.0),
                 escape_min_distance=None):
        self.env_check = env_check
        self.joint_names = list(joint_names)
        self.safe_margin = float(safe_margin)
        # Opt-in escape: a path that STARTS inside the margin (but at least this far apart,
        # i.e. not touching) is accepted while every sample stays at least as far apart as the
        # start (0.2 mm sampling tolerance); samples above the margin are always fine. Without it, a
        # measured pose that drifted a fraction of a millimetre inside the margin rejects
        # every command, including the ones that would move the arms apart.
        self.escape_min_distance = None if escape_min_distance is None else float(escape_min_distance)
        if self.escape_min_distance is not None and not 0 < self.escape_min_distance < self.safe_margin:
            raise ValueError("escape_min_distance must be in (0, safe_margin).")
        self.sweep_step_rad = float(sweep_step_rad)
        if not np.isfinite(self.safe_margin) or self.safe_margin < 0:
            raise ValueError("safe_margin must be finite and nonnegative.")
        if not np.isfinite(self.sweep_step_rad) or self.sweep_step_rad <= 0:
            raise ValueError("sweep_step_rad must be finite and positive.")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be nonempty and unique.")
        model = env_check.model
        self.idxs_joint = np.array([model.joint(name).id for name in self.joint_names],dtype=int)
        if np.any(~np.isin(model.jnt_type[self.idxs_joint],[mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE])):
            raise ValueError("Only scalar hinge and slide joints can be command checked.")
        self.idxs_qpos = model.jnt_qposadr[self.idxs_joint].copy()
        self.joint_limits = model.jnt_range[self.idxs_joint].copy()
        self.joint_limited = model.jnt_limited[self.idxs_joint].astype(bool)
        self.sweep_steps = np.full(len(self.joint_names),self.sweep_step_rad)
        self.sweep_steps[model.jnt_type[self.idxs_joint] == mujoco.mjtJoint.mjJNT_SLIDE] = 0.001
        self.gripper_couplings = []
        for side in ("l","r"):
            driver_name = "gripper_%s_joint1" % side
            if driver_name in self.joint_names:
                for number,sign in ((2,-1.0),(3,-1.0),(4,1.0)):
                    follower_name = "gripper_%s_joint%d" % (side,number)
                    if follower_name not in self.joint_names:
                        joint_id = model.joint(follower_name).id
                        self.gripper_couplings.append((model.jnt_qposadr[joint_id],self.joint_names.index(driver_name),sign))

        candidate_info = env_check.get_root_geom_pair_candidates(
            root_body_name               = "base_link",
            exclude_current_contacts     = False,
            exclude_initial_close_pairs  = False,
            respect_body_excludes        = False,
            forward                      = False,
        )
        self.exclusion_rules = {
            "parser": "Honor XML collision masks/excludes; skip visual, same-body and direct parent-child geoms.",
            "stationary": "Ignore pairs where neither geom belongs to an arm or gripper subtree.",
            "same_gripper": "Ignore internal linkage pairs in each single gripper; retain gripper versus arm/body/floor/other gripper.",
        }
        self.explicit_exclusions = []
        # MuJoCo encodes actual body IDs (without a +1 offset) in these pairs.
        # Decode here because the shared parser's exclusion check adds +1.
        xml_body_pairs = {
            (int(signature) >> 16,int(signature) & 0xffff)
            for signature in model.exclude_signature
        }
        candidate_pairs = []
        for g1,g2 in candidate_info["candidate_geom_pairs"]:
            b1,b2 = (int(model.geom_bodyid[g]) for g in (g1,g2))
            names = (env_check.body_names[b1],env_check.body_names[b2])
            moving = any(name.startswith(("arm_l_link","arm_r_link","gripper_l_","gripper_r_")) for name in names)
            if not moving:
                continue
            if tuple(sorted((b1,b2))) in xml_body_pairs:
                self.explicit_exclusions.append({"geom_pair":(int(g1),int(g2)),"body_pair":names,"reason":"source_xml_contact_exclude"})
                continue
            same_gripper = any(all(name.startswith("gripper_%s_" % side) for name in names) for side in ("l","r"))
            if same_gripper:
                self.explicit_exclusions.append({"geom_pair":(int(g1),int(g2)),"body_pair":names,"reason":"same_gripper_mechanical_linkage"})
                continue
            candidate_pairs.append((int(g1),int(g2)))
        self.candidate_geom_pairs = np.asarray(candidate_pairs,dtype=np.int32).reshape(-1,2)
        if not len(self.candidate_geom_pairs):
            raise ValueError("No arm/gripper collision candidates found.")

    def _set_q(self,q):
        self.env_check.data.qpos[self.idxs_qpos] = q
        for qpos_idx,driver_idx,sign in self.gripper_couplings:
            self.env_check.data.qpos[qpos_idx] = sign*q[driver_idx]
        self.env_check.forward(clip_position=False)

    def _minimum_distance(self):
        """Bound distant geometry and validate every possibly close query.

        Some mesh queries return zero with widely separated nearest points.
        A finite search radius avoids that unbounded-query failure; AABBs only
        discard pairs whose separation is already proven. Ambiguous close
        results are retried at the guard threshold and otherwise rejected.
        """
        env = self.env_check
        model = env.model
        rotations = env.data.geom_xmat.reshape(-1,3,3)
        centers = env.data.geom_xpos + np.einsum("gij,gj->gi",rotations,model.geom_aabb[:,:3])
        half_sizes = np.einsum("gij,gj->gi",np.abs(rotations),model.geom_aabb[:,3:])
        g1,g2 = self.candidate_geom_pairs.T
        gap = np.maximum(np.abs(centers[g1]-centers[g2])-half_sizes[g1]-half_sizes[g2],0.0)
        lower_bounds = np.linalg.norm(gap,axis=1)
        # Overlapping boxes do not bound signed penetration. A plane's visual
        # bounds do not bound its infinite collision surface, either.
        unbounded = (lower_bounds <= 1e-12) | np.any(model.geom_type[self.candidate_geom_pairs] == mujoco.mjtGeom.mjGEOM_PLANE,axis=1)
        lower_bounds[unbounded] = -np.inf
        radius = max(0.05,self.safe_margin+0.001)
        near_mask = lower_bounds <= radius
        near_pairs = self.candidate_geom_pairs[near_mask]
        infos = env.get_geom_pair_signed_distance_infos(
            near_pairs,distmax=radius,topk=None,forward=False,sanity_check=False,
        )
        by_pair = {(int(info["geom1_idx"]),int(info["geom2_idx"])):info for info in infos}
        if len(by_pair) != len(near_pairs):
            raise ValueError("Collision query did not cover every potentially close pair.")

        def valid_distance(info,cutoff,bound):
            distance = float(info.get("signed_dist",info["dist"]))
            if not np.isfinite(distance):
                return False,False
            if info.get("source") == "contact":
                return True,False
            points = np.asarray(info["fromto"],dtype=float)
            if points.shape != (6,) or not np.all(np.isfinite(points)):
                return False,False
            truncated = abs(distance-cutoff) <= 1e-12 and np.all(points == 0)
            # An all-zero endpoint buffer with a zero scalar can also be a
            # failed mesh query. Actual origin contacts arrive via contact_info.
            consistent = np.any(points != 0) and abs(np.linalg.norm(points[3:]-points[:3])-abs(distance)) <= 1e-4
            return truncated or (consistent and distance >= bound-1e-6),truncated

        closest = None
        for pair,bound in zip(self.candidate_geom_pairs,lower_bounds):
            key = tuple(int(value) for value in pair)
            is_bound = bound > radius
            if is_bound:
                info = {"dist":float(bound),"geom1_idx":key[0],"geom2_idx":key[1]}
            else:
                info = by_pair[key]
                valid,is_bound = valid_distance(info,radius,bound)
                if not valid and bound >= self.safe_margin:
                    info = dict(info,dist=float(bound),signed_dist=float(bound))
                    valid,is_bound = True,True
                if not valid:
                    cutoff = self.safe_margin+1e-6
                    info = env.get_geom_distance_info(geom1_idx=key[0],geom2_idx=key[1],distmax=cutoff)
                    valid,is_bound = valid_distance(info,cutoff,bound)
                    if not valid:
                        raise ValueError("Invalid distance for potentially colliding geometry.")
            distance = float(info.get("signed_dist",info["dist"]))
            if closest is None or distance < closest["signed_dist"]:
                closest = {
                    "signed_dist":distance,"is_lower_bound":bool(is_bound),
                    "geom1_idx":key[0],"geom2_idx":key[1],
                    "body1_name":env.body_names[int(model.geom_bodyid[key[0]])],
                    "body2_name":env.body_names[int(model.geom_bodyid[key[1]])],
                }
        return closest

    def check(self,q_start,q_end):
        """Reject malformed, out-of-range or insufficient-clearance commands."""
        check_info = {"accepted":False,"reason":"invalid_command","min_distance_m":float("nan"),"pair":None,"samples_checked":0}
        try:
            if np.asarray(q_start).dtype.kind not in "fiu" or np.asarray(q_end).dtype.kind not in "fiu":
                return check_info
            q_start,q_end = (np.asarray(q,dtype=np.float64) for q in (q_start,q_end))
        except (TypeError,ValueError,OverflowError):
            return check_info
        if any(q.shape != (len(self.joint_names),) or not np.all(np.isfinite(q)) for q in (q_start,q_end)):
            return check_info
        if not np.all(np.isfinite(self.env_check.data.qpos)):
            check_info["reason"] = "invalid_check_environment"
            return check_info
        for label,q in (("start",q_start),("end",q_end)):
            outside = self.joint_limited & ((q < self.joint_limits[:,0]) | (q > self.joint_limits[:,1]))
            if np.any(outside):
                check_info["reason"] = "joint_limit_%s:%s" % (label,self.joint_names[int(np.flatnonzero(outside)[0])])
                return check_info
        with np.errstate(over="ignore",invalid="ignore"):
            n_steps = np.ceil(np.max(np.abs(q_end-q_start)/self.sweep_steps))
        if not np.isfinite(n_steps) or n_steps > 10000:
            check_info["reason"] = "sweep_too_large"
            return check_info
        n_segments = max(1,int(n_steps))
        qpos_saved = self.env_check.data.qpos.copy()
        escape_floor = None
        try:
            for index,alpha in enumerate(np.linspace(0.0,1.0,n_segments+1)):
                self._set_q((1.0-alpha)*q_start+alpha*q_end)
                distance_info = self._minimum_distance()
                check_info["samples_checked"] += 1
                if distance_info is None or not np.isfinite(distance_info["signed_dist"]):
                    check_info["reason"] = "collision_query_invalid"
                    return check_info
                distance = float(distance_info["signed_dist"])
                if not np.isfinite(check_info["min_distance_m"]) or distance < check_info["min_distance_m"]:
                    check_info["min_distance_m"] = distance
                    check_info["distance_is_lower_bound"] = distance_info["is_lower_bound"]
                    check_info["pair"] = {
                        "geom_ids":(int(distance_info["geom1_idx"]),int(distance_info["geom2_idx"])),
                        "body_names":(distance_info["body1_name"],distance_info["body2_name"]),
                    }
                if (index == 0 and self.escape_min_distance is not None
                        and self.escape_min_distance <= distance < self.safe_margin):
                    escape_floor = distance
                    check_info["escape_start_distance_m"] = distance
                    continue
                if escape_floor is not None and escape_floor - 2e-4 <= distance < self.safe_margin:
                    continue   # never closer than the start pose (0.2 mm sampling tolerance)
                if distance < self.safe_margin:
                    check_info["reason"] = "collision_clearance"
                    check_info["sweep_fraction"] = float(alpha)
                    return check_info
            check_info.update(accepted=True,reason="accepted")
            return check_info
        except (ValueError,TypeError,RuntimeError,KeyError,IndexError) as exc:
            check_info["reason"] = "collision_query_error:%s" % type(exc).__name__
            return check_info
        finally:
            self.env_check.data.qpos[:] = qpos_saved
            self.env_check.forward(clip_position=False)


__all__ = ["build_fixed_quest_scene","SG2CommandGuard"]
