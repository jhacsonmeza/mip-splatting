'''
Usage:
    python render_video.py -m ${MODEL_PATH} --frames_per_segment 60 --keyframe_order angular
'''

import json
import argparse
import os
import copy
from tqdm import tqdm
import numpy as np
import imageio.v2 as imageio

import torch
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state


# ---------------------------------------------------------------------------
# === SLERP-based pose interpolation ===
# ---------------------------------------------------------------------------
def slerp(q0, q1, t):
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return ((1 - t) * q0 + t * q1) / np.linalg.norm((1 - t) * q0 + t * q1)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1

def rotmat_to_quat(R):
    from scipy.spatial.transform import Rotation as R_
    return R_.from_matrix(R).as_quat()

def quat_to_rotmat(q):
    from scipy.spatial.transform import Rotation as R_
    return R_.from_quat(q).as_matrix()

def interpolate_pose(pose1, pose2, t):
    R1, t1 = pose1[:, :3], pose1[:, 3]
    R2, t2 = pose2[:, :3], pose2[:, 3]
    q1, q2 = rotmat_to_quat(R1), rotmat_to_quat(R2)
    q_interp = slerp(q1, q2, t)
    R_interp = quat_to_rotmat(q_interp)
    t_interp = (1 - t) * t1 + t * t2
    return np.hstack([R_interp, t_interp[:, None]])




# ---------------------------------------------------------------------------
# === Trajectory + camera helpers ===
# ---------------------------------------------------------------------------
def cameras_to_c2w(cams):
    """Extract 3x4 camera-to-world poses from 3DGS Camera objects.
 
    Camera.world_view_transform is stored transposed (row-vector convention),
    so transpose it back to get the usual 4x4 world-to-camera matrix, then invert.
    """
    poses = []
    for cam in cams:
        w2c = cam.world_view_transform.transpose(0, 1).detach().cpu().numpy().astype(np.float64)
        poses.append(np.linalg.inv(w2c)[:3, :])
    return poses


def build_trajectory(keyframes_poses, frames_per_segment, loop=False):
    """Interpolate through consecutive keyframes_poses.

    Each segment contributes `frames_per_segment` frames (start pose included, end pose
    excluded) so no pose is duplicated at segment boundaries. The final keyframe is
    appended once at the end unless `loop` is set (then the path returns to the first pose).
    """
    keyframes_poses = list(keyframes_poses)
    if loop:
        keyframes_poses.append(keyframes_poses[0])

    traj = []
    for p0, p1 in zip(keyframes_poses[:-1], keyframes_poses[1:]):
        for t in np.linspace(0.0, 1.0, frames_per_segment, endpoint=False):
            traj.append(interpolate_pose(p0, p1, t))
    if not loop:
        traj.append(keyframes_poses[-1])
    return traj

def save_trajectory(poses, output_path):
    """
    Save a list/array of camera poses to a JSON file.

    Args:
        poses: NumPy array of shape (N, 3, 4).
        output_path: Path to the output JSON file.
    """
    poses = np.asarray(poses)

    if poses.ndim != 3 or poses.shape[1:] != (3, 4):
        raise ValueError(
            f"Expected poses with shape (N, 3, 4), got {poses.shape}"
        )

    with open(output_path, "w") as f:
        json.dump(poses.tolist(), f, indent=2)

def make_camera(template, c2w34):
    """Clone a real camera (keeps FoV / resolution) and overwrite its extrinsics."""
    c2w = np.eye(4)
    c2w[:3, :] = c2w34
    w2c = np.linalg.inv(c2w)

    cam = copy.copy(template)
    wvt = torch.tensor(w2c.T, dtype=torch.float32, device="cuda")  # stored transposed
    cam.world_view_transform = wvt
    cam.full_proj_transform = (
        wvt.unsqueeze(0).bmm(template.projection_matrix.unsqueeze(0)).squeeze(0)
    )
    cam.camera_center = torch.tensor(c2w[:3, 3], dtype=torch.float32, device="cuda")
    return cam



# ---------------------------------------------------------------------------
# === Keyframe ordering ===
# ---------------------------------------------------------------------------
def _nearest_neighbor_order(dist, start=0):
    """Greedy chain: repeatedly hop to the closest unvisited point. Fast, but a pure
    greedy pass can strand one outlier for last, forcing one long edge somewhere."""
    n = dist.shape[0]
    unvisited = set(range(n)) - {start}
    order = [start]
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda j: dist[last, j])
        order.append(nxt)
        unvisited.discard(nxt)
    return order


def _two_opt(order, dist, cyclic=False, max_passes=200):
    """Classic 2-opt local search: uncrosses any pair of edges whose swap shortens the
    tour. Cleans up the outlier/crossing artifacts nearest-neighbor leaves behind.
    `cyclic=True` also considers the wrap-around edge (last keyframe -> first), which
    matters when the video will use --loop.
    """
    order = list(order)
    n = len(order)
    if n < 4:
        return order
    j_end = n if cyclic else n - 1  # cyclic tours have one extra (wrap-around) edge
    improved, passes = True, 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            a, b = order[i], order[i + 1]
            for j in range(i + 2, j_end):
                c, d = order[j], order[(j + 1) % n]
                if a == d:  # these two edges already share a vertex in the cycle
                    continue
                if dist[a, c] + dist[b, d] < dist[a, b] + dist[c, d] - 1e-9:
                    order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
                    a, b = order[i], order[i + 1]
                    improved = True
    return order


def order_by_proximity(centers, cyclic=False):
    """Reorder keyframes to (approximately) minimize total travel distance: nearest-
    neighbor construction followed by 2-opt refinement. Makes no assumption about the
    capture pattern, but a straight-line path can still cut through the scene if the
    cameras weren't captured along a single sweep."""
    centers = np.asarray(centers, dtype=np.float64)
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    order = _nearest_neighbor_order(dist, start=0)
    order = _two_opt(order, dist, cyclic=cyclic)
    return order

def order_by_angle(centers):
    """Order keyframes by azimuth around the scene's centroid, in the plane that best
    fits the camera centers (found via SVD). This matches how Mip-NeRF360-style scenes
    are usually captured -- one loop around a central subject -- and gives a smooth
    orbit without solving any combinatorial problem. It assumes a single roughly-planar
    loop; if the capture isn't like that (e.g. a straight walkthrough), use 'nearest'."""
    centers = np.asarray(centers, dtype=np.float64)
    centered = centers - centers.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:2]  # the 2 directions capturing the most spread = the orbit plane
    proj = centered @ basis.T
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    return list(np.argsort(angles))






# === Video render function ===
def render_trajectory_video(dataset, pipeline, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        if args.output:
            out_path = args.output
        else:
            video_path = os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter))
            os.makedirs(video_path, exist_ok=True)
            out_path = os.path.join(video_path, f"video_{scale_factor}.mp4")

        cams = scene.getTestCameras()
        if len(cams) == 0:
            print("[warn] No test cameras found (was the model trained with --eval?). "
                  "Falling back to the training cameras.")
            cams = scene.getTrainCameras()
        cams = sorted(cams, key=lambda c: c.image_name)
        template = cams[0]


        if args.traj_file:
            with open(args.traj_file) as f:
                data = json.load(f)
            traj = [np.array(p) for p in data]
            print("Trajectory loaded from JSON file")
        else:
            if len(cams) < 2:
                raise RuntimeError("Fewer than 2 keyframes remain after --keyframe_stride.")

            centers = np.array([c2w[:, 3] for c2w in cameras_to_c2w(cams)])
            if args.keyframe_order == "filename":
                order = list(range(len(cams)))
            elif args.keyframe_order == "nearest":
                order = order_by_proximity(centers, cyclic=args.loop)
            else:  # "angular"
                order = order_by_angle(centers)
            cams = [cams[i] for i in order][:: args.keyframe_stride]

            keyframes_poses = cameras_to_c2w(cams)
            traj = build_trajectory(keyframes_poses, args.frames_per_segment, loop=args.loop)

            if args.save_traj:
                save_trajectory(traj, os.path.join(os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0] + ".json"))
                print("Trajectory file stored")


        print(f"{len(cams)} keyframes -> {len(traj)} frames "
              f"({len(traj) / args.fps:.1f}s @ {args.fps} fps), "
              f"{template.image_width}x{template.image_height}")

        writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264", quality=4, pixelformat="yuv420p", macro_block_size=1)
        try:
            for pose in tqdm(traj, desc="Rendering"):
                cam = make_camera(template, pose)
                img = render(cam, gaussians, pipeline, background, kernel_size=dataset.kernel_size)["render"]
                frame = (img.clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
                # yuv420p needs even dimensions
                h, w = frame.shape[:2]
                writer.append_data(frame[: h // 2 * 2, : w // 2 * 2])
        finally:
            writer.close()


        print(f"Saved {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render video")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--frames_per_segment", default=30, type=int,
                        help="Interpolated frames between two consecutive keyframe poses")
    parser.add_argument("--keyframe_stride", default=1, type=int,
                        help="Use every N-th test camera as a keyframe (shorter / smoother path)")
    parser.add_argument("--keyframe_order", default="filename", choices=["filename", "nearest", "angular"],
                        help="How to order test cameras into keyframes.")
    parser.add_argument("--loop", action="store_true",
                        help="Close the path by interpolating from the last pose back to the first")
    parser.add_argument("--output", default="", type=str,
                        help="Path of the video file to write, e.g. videos/garden.mp4")
    parser.add_argument("--save_traj", action="store_true",
                        help="Writes trajectory as a JSON file")
    parser.add_argument("--traj_file", default="", type=str,
                        help="Path to the trajectory JSON file")

    args = get_combined_args(parser)


    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_trajectory_video(model.extract(args), pipeline.extract(args), args)
