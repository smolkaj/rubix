import cv2
import numpy as np
from rubix import solved_cube, color_names

# Define color ranges in HSV
COLOR_RANGES = {
    'RED': ([0, 100, 100], [10, 255, 255]),
    'ORANGE': ([10, 100, 100], [25, 255, 255]),
    'YELLOW': ([25, 100, 100], [35, 255, 255]),
    'GREEN': ([35, 100, 100], [85, 255, 255]),
    'BLUE': ([85, 100, 100], [130, 255, 255]),
    'WHITE': ([0, 0, 200], [180, 30, 255])
}

def get_color(hsv):
    for color, (lower, upper) in COLOR_RANGES.items():
        if cv2.inRange(hsv, np.array(lower), np.array(upper)).any():
            return color
    return None

def detect_cube(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            return approx
    
    return None

def extract_face_colors(frame, contour):
    try:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, M, (width, height))
        
        cell_width = width // 3
        cell_height = height // 3
        
        colors = []
        for i in range(3):
            for j in range(3):
                cell = warped[j*cell_height:(j+1)*cell_height, i*cell_width:(i+1)*cell_width]
                hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
                color = get_color(hsv)
                colors.append(color)
        
        return colors
    except Exception as e:
        print(f"Error in extract_face_colors: {str(e)}")
        return None

def scan_cube():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None
    except Exception as e:
        print(f"Error: Failed to initialize webcam. {str(e)}")
        return None

    cube_state = {}
    face_count = 0
    
    print("Scanning cube. Press 'q' to quit at any time.")

    while face_count < 6:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
        
        cube_contour = detect_cube(frame)
        if cube_contour is not None:
            cv2.drawContours(frame, [cube_contour], 0, (0, 255, 0), 2)
            
            colors = extract_face_colors(frame, cube_contour)
            if colors is None:
                print("Error: Failed to extract face colors. Skipping this frame.")
                cv2.imshow("Cube Scanner", frame)
                cv2.waitKey(1)
                continue
            cv2.putText(frame, f"Face {face_count + 1}: Press SPACE to capture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Cube Scanner", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                face_normal = list(color_names.keys())[face_count]
                cube_state[face_normal] = colors
                face_count += 1
                print(f"Face {face_count} captured.")
            elif key == ord('q'):
                print("Scanning aborted by user.")
                break
        else:
            cv2.putText(frame, "No cube detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Cube Scanner", frame)
            cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if face_count < 6:
        print("Scanning incomplete. Not all faces were captured.")
        return None
    
    return cube_state

def cube_state_to_solver_format(cube_state):
    solver_cube = list(solved_cube)
    for face_normal, colors in cube_state.items():
        for i, color in enumerate(colors):
            cubelet = tuple(c * face_normal[i] for i, c in enumerate(face_normal))
            if color is None:
                # Use a default rotation for unknown colors
                rotation = tuple(tuple(int(i == j) for i in range(3)) for j in range(3))
            else:
                rotation = tuple(tuple(int(c == color_names[color]) for c in face_normal) for _ in range(3))
            solver_cube[solver_cube.index((cubelet, solved_cube[0][1]))] = (cubelet, rotation)
    return tuple(solver_cube)

if __name__ == "__main__":
    cube_state = scan_cube()
    if cube_state:
        solver_cube = cube_state_to_solver_format(cube_state)
        print(solver_cube)
    else:
        print("Failed to scan cube. Please try again or use a randomly shuffled cube.")
