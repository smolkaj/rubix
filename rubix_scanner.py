import cv2
import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional
import time

class RubiksCube3DScanner:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frames = []
        self.feature_matcher = cv2.FlannBasedMatcher_create()
        self.reconstruction = None
        self.camera_matrix = None

    def capture_frames(self, num_frames: int = 30):
        print("Capturing frames...")
        for i in range(num_frames):
            print(f"-> frame {i}")
            time.sleep(15/num_frames)  # Capture for 15 seconds.
            ret, frame = self.cap.read()
            if ret:
                self.frames.append(frame)
            else:
                break
        print(f"Captured {len(self.frames)} frames.")

    def detect_features(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        print("Detecting features...")
        sift = cv2.SIFT_create()
        features = []
        for frame in self.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            features.append((kp, des))
        print(f"Detected features in {len(features)} frames.")
        return features

    def match_features(self, features: List[Tuple[np.ndarray, np.ndarray]]) -> List[List[cv2.DMatch]]:
        print("Matching features...")
        matches = []
        for i in range(len(features) - 1):
            matches_pair = self.feature_matcher.knnMatch(features[i][1], features[i+1][1], k=2)
            good_matches = []
            for m, n in matches_pair:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            matches.append(good_matches)
        print(f"Matched features between {len(matches)} pairs of frames.")
        return matches

    def estimate_camera_poses(self, features: List[Tuple[np.ndarray, np.ndarray]], matches: List[List[cv2.DMatch]]) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        print("Estimating camera poses...")
        camera_poses = []
        prev_points = None
        prev_pose = np.eye(4)

        # Estimate camera matrix (assuming fixed camera parameters)
        if self.camera_matrix is None:
            frame_size = self.frames[0].shape[:2][::-1]
            self.camera_matrix = np.array([
                [frame_size[0], 0, frame_size[0]/2],
                [0, frame_size[0], frame_size[1]/2],
                [0, 0, 1]
            ], dtype=np.float32)

        for i, (feature_pair, match_list) in enumerate(zip(zip(features[:-1], features[1:]), matches)):
            kp1, _ = feature_pair[0]
            kp2, _ = feature_pair[1]

            points1 = np.float32([kp1[m.queryIdx].pt for m in match_list]).reshape(-1, 1, 2)
            points2 = np.float32([kp2[m.trainIdx].pt for m in match_list]).reshape(-1, 1, 2)

            E, mask = cv2.findEssentialMat(points2, points1, self.camera_matrix, cv2.RANSAC, 0.999, 1.0)
            _, R, t, mask = cv2.recoverPose(E, points2, points1, self.camera_matrix, mask=mask)

            # Compose the transformation with the previous pose
            current_pose = np.eye(4)
            current_pose[:3, :3] = R
            current_pose[:3, 3] = t.reshape(-1)
            composed_pose = prev_pose @ current_pose
            
            camera_poses.append((composed_pose[:3, :3], composed_pose[:3, 3]))
            prev_pose = composed_pose
            prev_points = points2

        print(f"Estimated {len(camera_poses)} camera poses.")
        return camera_poses

    def triangulate_points(self, features: List[Tuple[np.ndarray, np.ndarray]], camera_poses: List[Tuple[np.ndarray, np.ndarray]]):
        print("Triangulating points...")
        points_3d = []
        colors = []

        for i in range(len(camera_poses) - 1):
            kp1, _ = features[i]
            kp2, _ = features[i+1]
            matches = self.match_features([features[i], features[i+1]])[0]

            points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            R1, t1 = camera_poses[i]
            R2, t2 = camera_poses[i+1]

            P1 = self.camera_matrix @ np.hstack((R1, t1.reshape(-1, 1)))
            P2 = self.camera_matrix @ np.hstack((R2, t2.reshape(-1, 1)))

            points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
            points_3d_homogeneous = points_4d.T
            points_3d_euclidean = points_3d_homogeneous[:, :3] / points_3d_homogeneous[:, 3:]
            points_3d.extend(points_3d_euclidean)

            # Extract colors for the 3D points
            for pt1 in points1:
                x, y = int(pt1[0][0]), int(pt1[0][1])
                if 0 <= x < self.frames[i].shape[1] and 0 <= y < self.frames[i].shape[0]:
                    color = self.frames[i][y, x] / 255.0  # Normalize color values
                    colors.append(color)

        print(f"Triangulated {len(points_3d)} 3D points.")
        return np.array(points_3d), np.array(colors)

    def visualize_reconstruction(self):
        if self.reconstruction is not None:
            o3d.visualization.draw_geometries([self.reconstruction])

    def scan(self):
        self.capture_frames()
        features = self.detect_features()
        matches = self.match_features(features)
        camera_poses = self.estimate_camera_poses(features, matches)
        points_3d, colors = self.triangulate_points(features, camera_poses)
        
        # Create a point cloud for visualization
        self.reconstruction = o3d.geometry.PointCloud()
        self.reconstruction.points = o3d.utility.Vector3dVector(points_3d)
        self.reconstruction.colors = o3d.utility.Vector3dVector(colors)

        print("Reconstruction complete. Visualizing...")
        self.visualize_reconstruction()

        return points_3d, colors

if __name__ == "__main__":
    scanner = RubiksCube3DScanner()
    points_3d, colors = scanner.scan()
    print(f"Reconstructed {len(points_3d)} points with colors.")
