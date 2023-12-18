import json
import tkinter as tk
from tkinter import filedialog
import mediapipe as mp

joint_indices = {
    'left_hip': [11, 23, 25], 'right_hip': [12, 24, 26],
    'left_knee': [23, 25, 27], 'right_knee': [24, 26, 28],
    'left_shoulder': [23, 11, 13], 'right_shoulder': [24, 12, 14],
    'left_elbow': [11, 13, 15], 'right_elbow': [12, 14, 16]
}


class VideoObject:
    def __init__(self, path, frame_rate, frame_count, width, height, name):
        self.path = path
        self.frame_rate = frame_rate
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.landmarks = []  # Original landmarks
        self.normalized_landmarks = []  # Normalized landmarks
        self.original_joint_angles = []  # Joint angles for original frame rate
        self.normalized_joint_angles = []  # Joint angles for normalized frame rate
        self.name = name
        self.shooting_hand = ''
        self.shot_start_frame = None
        self.shot_end_frame = None
        self.more_visible_hip = None  # New property for the more visible hip

def load_video_object_from_json():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a file dialog to choose the JSON file
    file_path = filedialog.askopenfilename(
        filetypes=[("JSON files", "*.json")],
        title="Select a JSON File"
    )

    if not file_path:
        print("No file selected.")
        return None

    # Read the JSON file and reconstruct the VideoObject
    with open(file_path, 'r') as file:
        data = json.load(file)

    video_obj = VideoObject(
        path=data["path"],
        frame_rate=data["frame_rate"],
        frame_count=data["frame_count"],
        width=data["width"],
        height=data["height"],
        name=data["name"]
    )


    # Reconstruct the landmarks and joint angles
    mp_drawing = mp.solutions.drawing_utils
    landmarks_reconstructed = []
    for frame_data in data["landmarks"]:
        frame_landmarks = []
        for landmark in frame_data:
            # Convert normalized x, y to pixel coordinates
            pixel_coords = mp_drawing._normalized_to_pixel_coordinates(
                landmark[0], landmark[1], video_obj.width, video_obj.height
            )
            if pixel_coords:
                # Maintain the full structure: (x, y, z, visibility)
                x, y = pixel_coords
                frame_landmarks.append((x, y, landmark[2], landmark[3]))
            else:
                frame_landmarks.append((None, None, landmark[2], landmark[3]))
        landmarks_reconstructed.append(frame_landmarks)

    video_obj.landmarks = landmarks_reconstructed
    video_obj.original_joint_angles = data["original_joint_angles"]

    # Assign other properties
    video_obj.shooting_hand = data["shooting_hand"]
    video_obj.shot_start_frame = data["shot_start_frame"]
    video_obj.shot_end_frame = data["shot_end_frame"]
    video_obj.more_visible_hip = data["more_visible_hip"]

    print("\nLoaded Video Object:")
    print(f"Path: {video_obj.path}, Frame Rate: {video_obj.frame_rate}, Frame Count: {video_obj.frame_count}")
    print(f"Width: {video_obj.width}, Height: {video_obj.height}, Name: {video_obj.name}")
    print(f"Shooting Hand: {video_obj.shooting_hand}, More Visible Hip: {video_obj.more_visible_hip}")
    print(f"Shot Start Frame: {video_obj.shot_start_frame}, Shot End Frame: {video_obj.shot_end_frame}")

    # Print example landmarks and joint angles
    print("\nExample Landmarks (First Frame):")
    print(video_obj.landmarks[:5])  # Print first 5 landmarks of the first frame

    print("\nExample Joint Angles (First Frame):")
    print(video_obj.original_joint_angles[0])  # Print joint angles of the first frame

    return video_obj


def analyze_shot(video_obj):
    def is_efficient(video_obj):
        hip_index = joint_indices[video_obj.more_visible_hip][1]
        end_frame_index = min(video_obj.shot_end_frame, len(video_obj.landmarks) - 1)
        end_frame_hip_y = video_obj.landmarks[end_frame_index][hip_index][1]
        prev_frame_hip_y = video_obj.landmarks[end_frame_index - 1][hip_index][1]

        print(f"\nQuestion: Is it efficient?")
        print(f"  Analyzing hip movement at shot end frame.")
        print(f"  Hip Y-coordinate at shot end frame: {end_frame_hip_y}")
        print(f"  Hip Y-coordinate at previous frame: {prev_frame_hip_y}")

        efficient = end_frame_hip_y <= prev_frame_hip_y
        print(f"  Decision: {'Efficient' if efficient else 'Not efficient'}")
        return efficient

    def is_shot_while_jumping(video_obj):
        hip_index = joint_indices[video_obj.more_visible_hip][1]
        release_frame = video_obj.shot_end_frame
        frames_to_check = 5  # Number of frames before the release to check for deceleration

        # Ensure we have enough frames before the release frame
        start_frame = max(0, release_frame - frames_to_check)

        # Get hip heights for the relevant frames
        hip_heights = [video_obj.landmarks[frame][hip_index][1] for frame in range(start_frame, release_frame + 1) if video_obj.landmarks[frame][hip_index][1] is not None]

        # Calculate the change in height (deceleration) over these frames
        deceleration = sum(hip_heights[i] - hip_heights[i-1] for i in range(1, len(hip_heights)))
        

        print(f"\nQuestion: Is it a shot while jumping?")
        print(f"  Analyzing hip deceleration before shot release.")
        print(deceleration)
        
        # If deceleration is noticeable, it's more likely a pull-up shot
        if deceleration < -10: # TO BE optimized
            print("  Decision: Pull Up Shot (due to hip deceleration)")
            return False
        else:
            print("  Decision: 1 Motion Shot")
            return True


    print("\nAnalyzing Shot:")
    if is_efficient(video_obj):
        if is_shot_while_jumping(video_obj):
            print("Final Decision: 1 Motion Shot")
            return "1 motion shot"
        else:
            print("Final Decision: Pull Up Shot")
            return "pull up shot"
    else:
        print("Final Decision: Dropping Shot")
        return "dropping shot"


def print_shot_analysis(video_obj):
    shot_type = analyze_shot(video_obj)
    print(f"Shot Analysis: {shot_type}")

# Example usage
video = load_video_object_from_json()
if video:
    print("Loaded video object successfully.")
    print_shot_analysis(video)
else:
    print("Failed to load video object.")
