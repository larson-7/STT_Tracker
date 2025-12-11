import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Tuple
from tqdm import tqdm

# Stone Soup Imports
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.types.state import GaussianState, State
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantAcceleration,
)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator

# --- Configuration ---
OUTPUT_DIR = "data"
TRAIN_SAMPLES = 100
VAL_SAMPLES = 20
DURATION_FRAMES = 60

# Map sensor names to integer IDs
SENSOR_MAP = {
    "Modality_A": 0,
    "Modality_B": 1,
    "Modality_C": 2,
}

# Map Scenario Types to Integer IDs
SCENARIO_MAP = {
    "straight": 0,
    "turn_90": 1,
    "u_turn": 2,
    "break_reacquire": 3,
    "snake_search": 4,
}


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# --- 1. Ownship Generation ---
def generate_ownship_path(
    start_time: datetime, duration: int
) -> Tuple[GroundTruthPath, str]:
    """
    Generates Ownship trajectory with various maneuver scenarios.
    """
    scenario_type = np.random.choice(list(SCENARIO_MAP.keys()))

    # Initial State (Origin)
    x, y, z = 0, 0, 0
    speed = 5.0
    heading = np.radians(90)  # Facing North

    path = GroundTruthPath()

    for k in range(duration):
        time_step = start_time + timedelta(seconds=k)

        # --- Maneuver Logic ---
        if scenario_type == "break_reacquire":
            if 15 <= k < 25:
                heading -= np.radians(18)  # Turn Away (Blind)
            elif 25 <= k < 35:
                pass  # Hold (Blind)
            elif 35 <= k < 45:
                heading += np.radians(18)  # Turn Back (Reacquire)

        elif scenario_type == "snake_search":
            if 10 <= k < 20:
                heading += np.radians(6)  # Left
            elif 20 <= k < 35:
                heading -= np.radians(8)  # Hard Right
            elif 35 <= k < 50:
                heading += np.radians(4)  # Recover

        elif scenario_type == "turn_90" and 20 <= k < 30:
            heading -= np.radians(9)

        elif scenario_type == "u_turn" and 15 <= k < 35:
            heading += np.radians(9)

        # Calculate Velocity components
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        vz = 0

        # Update Position
        x += vx
        y += vy

        state_vec = [x, vx, 0, y, vy, 0, z, vz, 0]
        path.append(GroundTruthState(state_vec, timestamp=time_step))

    return path, scenario_type


# --- 2. Target Scenario Generation ---
def generate_random_scenario(
    start_time: datetime, duration: int
) -> List[GroundTruthPath]:
    transition_model = CombinedLinearGaussianTransitionModel(
        [
            ConstantAcceleration(0.01),
            ConstantAcceleration(0.01),
            ConstantAcceleration(0.01),
        ]
    )

    num_targets = np.random.randint(2, 5)
    ground_truths = []

    for i in range(num_targets):
        x = np.random.uniform(-40, 40)
        y = np.random.uniform(30, 90)
        z = np.random.uniform(0, 50)
        vx = np.random.uniform(-3, 3)
        vy = np.random.uniform(-1, 1)
        vz = np.random.uniform(-0.1, 0.1)

        # Force Crossing for first 2 targets
        if i == 0:
            x, y, z = -30, 60, 20
            vx, vy, vz = 2.0, 0, 0
        elif i == 1:
            x, y, z = 30, 60, 25
            vx, vy, vz = -2.0, 0, 0

        initial_vector = [x, vx, 0, y, vy, 0, z, vz, 0]
        state = GroundTruthState(initial_vector, timestamp=start_time)
        truth = GroundTruthPath([state])

        current_state = state
        for k in range(1, duration):
            new_time = start_time + timedelta(seconds=k)
            new_vector = transition_model.function(
                current_state, noise=True, time_interval=timedelta(seconds=1)
            )
            current_state = GroundTruthState(new_vector, timestamp=new_time)
            truth.append(current_state)

        ground_truths.append(truth)

    return ground_truths


# --- 3. Geometric Probability Calculation ---
def get_detection_probability(
    ownship_state: State, target_state: State, base_prob: float
) -> float:
    """Calculates Pd based on Antenna Train Angle (ATA)."""
    ox, oy = ownship_state.state_vector[0], ownship_state.state_vector[3]
    tx, ty = target_state.state_vector[0], target_state.state_vector[3]
    ovx, ovy = ownship_state.state_vector[1], ownship_state.state_vector[4]

    v_target_x = tx - ox
    v_target_y = ty - oy

    mag_o = np.sqrt(ovx**2 + ovy**2)
    mag_t = np.sqrt(v_target_x**2 + v_target_y**2)

    if mag_o == 0 or mag_t == 0:
        return base_prob

    dot = (ovx * v_target_x) + (ovy * v_target_y)
    angle_rad = np.arccos(np.clip(dot / (mag_o * mag_t), -1.0, 1.0))
    angle_deg = np.degrees(np.abs(angle_rad))

    # 1. Blind Spot
    if angle_deg > 90:
        return 0.0
    # 2. Edge of Envelope
    elif 40 < angle_deg < 55:
        return 0.40
    # 3. Nominal
    else:
        return base_prob


# --- 4. Tracker Components ---
def create_sensor_tracker_components(measurement_model):
    predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel(
            [
                ConstantAcceleration(0.1),
                ConstantAcceleration(0.1),
                ConstantAcceleration(0.1),
            ]
        )
    )
    updater = KalmanUpdater(measurement_model)
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=Mahalanobis(), missed_distance=15
    )
    data_associator = GNNWith2DAssignment(hypothesiser)
    deleter = CovarianceBasedDeleter(covar_trace_thresh=250)

    initiator = MultiMeasurementInitiator(
        prior_state=GaussianState(np.zeros((9, 1)), np.diag([20, 5, 1] * 3)),
        measurement_model=None,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=3,
    )

    return {
        "initiator": initiator,
        "deleter": deleter,
        "data_associator": data_associator,
        "updater": updater,
        "tracks": set(),
    }


# --- 5. Simulation Runner ---
def run_episode(
    episode_id, start_time, duration
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapping = (0, 3, 6)

    sensor_configs = {
        "Modality_A": {"cov": 0.5, "prob": 0.95, "clutter": 1},
        "Modality_B": {"cov": 5.0, "prob": 0.90, "clutter": 2},
        "Modality_C": {"cov": 10.0, "prob": 0.85, "clutter": 5},
    }

    trackers = {}
    models = {}
    for name, cfg in sensor_configs.items():
        mat = np.diag([cfg["cov"]] * 3)
        if name == "Modality_C":
            mat[2, 2] *= 10
        models[name] = LinearGaussian(ndim_state=9, mapping=mapping, noise_covar=mat)
        trackers[name] = create_sensor_tracker_components(models[name])

    ground_truths = generate_random_scenario(start_time, duration)
    gt_path_map = {path: i for i, path in enumerate(ground_truths)}
    ownship_path, scenario_type = generate_ownship_path(start_time, duration)

    scenario_code = SCENARIO_MAP[scenario_type]

    episode_tracks_rows = []
    episode_gt_rows = []
    episode_ownship_rows = []

    for k in range(duration):
        current_time = start_time + timedelta(seconds=k)

        # Access state safely
        ownship_state = ownship_path[k]

        # Ownship Data
        o_vec = ownship_state.state_vector.flatten()
        episode_ownship_rows.append(
            [
                episode_id,
                k,
                o_vec[0],
                o_vec[3],
                o_vec[6],
                o_vec[1],
                o_vec[4],
                o_vec[7],
                scenario_code,
            ]
        )

        # Ground Truth Data
        active_truths = []
        for obj_idx, gt_path in enumerate(ground_truths):
            state = next((s for s in gt_path if s.timestamp == current_time), None)
            if state:
                active_truths.append((gt_path, state))
                vec = state.state_vector.flatten()
                episode_gt_rows.append(
                    [
                        episode_id,
                        k,
                        obj_idx,
                        vec[0],
                        vec[3],
                        vec[6],
                        vec[1],
                        vec[4],
                        vec[7],
                        vec[2],
                        vec[5],
                        vec[8],
                    ]
                )

        # Sensor & Tracker Loop
        for name, config in sensor_configs.items():
            detections = set()
            model = models[name]
            base_prob = config["prob"]

            for gt_path, state in active_truths:
                if isinstance(ownship_state, State):
                    det_prob = get_detection_probability(
                        ownship_state, state, base_prob
                    )
                else:
                    det_prob = 0.9

                if np.random.rand() <= det_prob:
                    meas = model.function(state, noise=True)
                    detections.add(
                        TrueDetection(
                            meas,
                            timestamp=current_time,
                            measurement_model=model,
                            groundtruth_path=gt_path,
                        )
                    )

            for _ in range(np.random.poisson(config["clutter"])):
                detections.add(
                    Clutter(
                        np.random.uniform(-100, 100, (3, 1)),
                        timestamp=current_time,
                        measurement_model=model,
                    )
                )

            tracker = trackers[name]
            tracks = tracker["tracks"]
            tracks = tracker["deleter"].delete_tracks(tracks)
            hypotheses = tracker["data_associator"].associate(
                tracks, detections, current_time
            )

            for track in tracks:
                hyp = hypotheses[track]
                if hyp.measurement:
                    track.append(tracker["updater"].update(hyp))
                elif hyp.prediction:
                    track.append(hyp.prediction)

            new_tracks = tracker["initiator"].initiate(
                detections
                - {h.measurement for h in hypotheses.values() if h.measurement},
                current_time,
            )
            tracks.update(new_tracks)
            tracker["tracks"] = tracks

            # Extract Track Data
            sid = SENSOR_MAP[name]
            for trk in tracks:
                vec = trk.state_vector.flatten()
                cov = np.diag(trk.covar).flatten()

                truth_id = -1
                if hasattr(trk.state, "hypothesis") and trk.state.hypothesis:
                    meas = trk.state.hypothesis.measurement
                    if (
                        isinstance(meas, TrueDetection)
                        and meas.groundtruth_path in gt_path_map
                    ):
                        truth_id = gt_path_map[meas.groundtruth_path]

                episode_tracks_rows.append(
                    [
                        episode_id,
                        k,
                        vec[0],
                        vec[3],
                        vec[6],
                        vec[1],
                        vec[4],
                        vec[7],
                        vec[2],
                        vec[5],
                        vec[8],
                        cov[0],
                        cov[3],
                        cov[6],
                        sid,
                        truth_id,
                    ]
                )

    return (
        np.array(episode_tracks_rows) if episode_tracks_rows else np.empty((0, 16)),
        np.array(episode_gt_rows) if episode_gt_rows else np.empty((0, 12)),
        np.array(episode_ownship_rows) if episode_ownship_rows else np.empty((0, 9)),
    )


def main():
    ensure_dir(OUTPUT_DIR)
    start_time = datetime.now().replace(microsecond=0)

    # --- FORMAT SPECIFICATIONS ---
    # These lists enforce the output types for CSV saving.
    # %d = Integer, %.6f = Float with 6 decimals.

    # Tracks: [episode, frame] (ints) + 12 floats + [sensor, truth] (ints)
    fmt_tracks = ["%d", "%d"] + ["%.6f"] * 12 + ["%d", "%d"]

    # Truth: [episode, frame, object] (ints) + 9 floats
    fmt_gt = ["%d", "%d", "%d"] + ["%.6f"] * 9

    # Ownship: [episode, frame] (ints) + 6 floats + [scenario] (int)
    fmt_own = ["%d", "%d"] + ["%.6f"] * 6 + ["%d"]

    # Headers
    header_tracks = "episode_id,frame_idx,x,y,z,vx,vy,vz,ax,ay,az,var_x,var_y,var_z,sensor_id,truth_id"
    header_gt = "episode_id,frame_idx,object_id,x,y,z,vx,vy,vz,ax,ay,az"
    header_own = "episode_id,frame_idx,x,y,z,vx,vy,vz,scenario_code"

    print(f"Generating {TRAIN_SAMPLES} Training Episodes...")
    t_tracks, t_gt, t_own = [], [], []
    for i in tqdm(range(TRAIN_SAMPLES)):
        tracks, gt, own = run_episode(i, start_time, DURATION_FRAMES)
        t_tracks.append(tracks)
        t_gt.append(gt)
        t_own.append(own)

    # Save Training Data
    np.savetxt(
        f"{OUTPUT_DIR}/train_tracks.csv",
        np.vstack(t_tracks),
        delimiter=",",
        header=header_tracks,
        comments="",
        fmt=fmt_tracks,
    )
    np.savetxt(
        f"{OUTPUT_DIR}/train_truth.csv",
        np.vstack(t_gt),
        delimiter=",",
        header=header_gt,
        comments="",
        fmt=fmt_gt,
    )
    np.savetxt(
        f"{OUTPUT_DIR}/train_ownship.csv",
        np.vstack(t_own),
        delimiter=",",
        header=header_own,
        comments="",
        fmt=fmt_own,
    )

    print(f"Generating {VAL_SAMPLES} Validation Episodes...")
    v_tracks, v_gt, v_own = [], [], []
    for i in tqdm(range(VAL_SAMPLES)):
        tracks, gt, own = run_episode(i + 1000, start_time, DURATION_FRAMES)
        v_tracks.append(tracks)
        v_gt.append(gt)
        v_own.append(own)

    # Save Validation Data
    np.savetxt(
        f"{OUTPUT_DIR}/val_tracks.csv",
        np.vstack(v_tracks),
        delimiter=",",
        header=header_tracks,
        comments="",
        fmt=fmt_tracks,
    )
    np.savetxt(
        f"{OUTPUT_DIR}/val_truth.csv",
        np.vstack(v_gt),
        delimiter=",",
        header=header_gt,
        comments="",
        fmt=fmt_gt,
    )
    np.savetxt(
        f"{OUTPUT_DIR}/val_ownship.csv",
        np.vstack(v_own),
        delimiter=",",
        header=header_own,
        comments="",
        fmt=fmt_own,
    )

    print("Done. Data saved to", OUTPUT_DIR)
    print("Scenario Key:", SCENARIO_MAP)


if __name__ == "__main__":
    main()
