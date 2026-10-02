import cv2
import numpy as np
from typing import List, Tuple, Dict
import time

class RubiksCubeScanner:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Define color ranges for each face (in HSV)
        self.color_ranges = {
            'white': ([0, 0, 200], [180, 30, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'red': ([0, 100, 100], [10, 255, 255]),
            'orange': ([10, 100, 100], [20, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255])
        }
        self.color_names = list(self.color_ranges.keys())
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the frame for better color detection."""
        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        return blurred

    def detect_cube_face(self, frame: np.ndarray) -> Tuple[np.ndarray, List[List[str]]]:
        """Detect the colors of a cube face."""
        processed = self.preprocess_frame(frame)
        height, width = processed.shape[:2]
        
        # Create a grid for the 3x3 face
        grid_size = 3
        cell_height = height // grid_size
        cell_width = width // grid_size
        
        face_colors = []
        
        for i in range(grid_size):
            row_colors = []
            for j in range(grid_size):
                # Extract the cell region
                y1 = i * cell_height
                y2 = (i + 1) * cell_height
                x1 = j * cell_width
                x2 = (j + 1) * cell_width
                cell = processed[y1:y2, x1:x2]
                
                # Get the dominant color in the cell
                dominant_color = self.get_dominant_color(cell)
                row_colors.append(dominant_color)
            face_colors.append(row_colors)
        
        return frame, face_colors

    def get_dominant_color(self, cell: np.ndarray) -> str:
        """Determine the dominant color in a cell."""
        max_count = 0
        dominant_color = 'unknown'
        
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(cell, np.array(lower), np.array(upper))
            count = np.sum(mask > 0)
            
            if count > max_count:
                max_count = count
                dominant_color = color_name
        
        return dominant_color

    def scan_face(self) -> List[List[str]]:
        """Scan a single face of the cube."""
        print("Position the cube face in front of the camera...")
        print("Press 'c' to capture, 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Show the frame with grid overlay
            display_frame = frame.copy()
            height, width = display_frame.shape[:2]
            grid_size = 3
            cell_height = height // grid_size
            cell_width = width // grid_size
            
            # Draw grid lines
            for i in range(1, grid_size):
                cv2.line(display_frame, (0, i * cell_height), (width, i * cell_height), (255, 255, 255), 2)
                cv2.line(display_frame, (i * cell_width, 0), (i * cell_width, height), (255, 255, 255), 2)
            
            cv2.imshow('Cube Scanner', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                _, face_colors = self.detect_cube_face(frame)
                return face_colors
            elif key == ord('q'):
                return None

    def scan_cube(self) -> Dict[str, List[List[str]]]:
        """Scan all faces of the cube."""
        faces = {}
        face_names = ['front', 'right', 'back', 'left', 'up', 'down']
        
        print("Let's scan all faces of the cube!")
        for face in face_names:
            print(f"\nPosition the {face} face in front of the camera")
            face_colors = self.scan_face()
            if face_colors is None:
                print("Scanning cancelled")
                return None
            faces[face] = face_colors
            print(f"Scanned {face} face:")
            for row in face_colors:
                print(row)
        
        return faces

    def __del__(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    scanner = RubiksCubeScanner()
    cube_state = scanner.scan_cube()
    
    if cube_state:
        print("\nComplete cube state:")
        for face, colors in cube_state.items():
            print(f"\n{face} face:")
            for row in colors:
                print(row)
