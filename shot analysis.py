import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import numpy as np

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



def round_frame_rate(frame_rate):
    common_frame_rates = [30, 60, 120]
    closest_rate = min(common_frame_rates, key=lambda x: abs(x - frame_rate))
    return closest_rate

def select_video():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if not file_path:
        return None
    name = os.path.splitext(os.path.basename(file_path))[0]  # Extract video name without extension
    cap = cv2.VideoCapture(file_path)
    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = round_frame_rate(original_frame_rate)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    video = VideoObject(file_path, frame_rate, frame_count, width, height, name)
    video.shooting_hand = input("Enter your shooting hand (left/right): ").lower()
    while video.shooting_hand not in ['left', 'right']:
        video.shooting_hand = input("Invalid input. Please enter 'left' or 'right': ").lower()
    return video

def detect_landmarks(video_obj):
    cap = cv2.VideoCapture(video_obj.path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

    frame_idx = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the image color to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            video_obj.landmarks.append(results.pose_landmarks)

            # Normalize to 30 FPS
            if video_obj.frame_rate != 30:
                if frame_idx % round(video_obj.frame_rate / 30) == 0:
                    video_obj.normalized_landmarks.append(results.pose_landmarks)
            else:
                video_obj.normalized_landmarks.append(results.pose_landmarks)

        frame_idx += 1

    cap.release()


def calculate_all_joint_angles(video_obj):

    # Function to calculate average visibility for a set of landmarks
    def average_visibility(landmarks, indices):
        return np.mean([landmarks.landmark[index].visibility for index in indices])

    # Function to calculate joint angles
    def calculate_joint_angles(landmarks_list, side_preference):
        joint_angles = []

        for landmarks in landmarks_list:
            frame_angles = []

            # Determine the side for hip and knee based on visibility
            hip_side = 'left_hip' if average_visibility(landmarks, joint_indices['left_hip']) > average_visibility(landmarks, joint_indices['right_hip']) else 'right_hip'
            knee_side = 'left_knee' if average_visibility(landmarks, joint_indices['left_knee']) > average_visibility(landmarks, joint_indices['right_knee']) else 'right_knee'

            selected_joints = [hip_side, knee_side, side_preference + '_shoulder', side_preference + '_elbow']

            for joint in selected_joints:
                a = np.array([landmarks.landmark[joint_indices[joint][0]].x, landmarks.landmark[joint_indices[joint][0]].y])
                b = np.array([landmarks.landmark[joint_indices[joint][1]].x, landmarks.landmark[joint_indices[joint][1]].y])
                c = np.array([landmarks.landmark[joint_indices[joint][2]].x, landmarks.landmark[joint_indices[joint][2]].y])

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)
                if angle > 180.0:
                    angle = 360 - angle
                frame_angles.append(angle)
            joint_angles.append(frame_angles)
        
        return joint_angles

    # Ask user for their shooting hand preference
    # Calculate and update joint angles
    video_obj.original_joint_angles = calculate_joint_angles(video_obj.landmarks, video_obj.shooting_hand)

def find_shot_start_end(video_obj):
    # Start frame: when preferred hand starts moving upwards
    hand_index = 15 if video_obj.shooting_hand == 'left' else 16
    hand_heights = [landmarks.landmark[hand_index].y for landmarks in video_obj.landmarks]
    start_frame = next(i for i in range(1, len(hand_heights)) if hand_heights[i] < hand_heights[i-1])

    # End frame: when elbow angle is the largest
    elbow_angle_index = -1  # Last entry in each frame of original_joint_angles
    max_elbow_angle = max(frame[elbow_angle_index] for frame in video_obj.original_joint_angles)
    end_frame = next(i for i, frame in enumerate(video_obj.original_joint_angles) if frame[elbow_angle_index] == max_elbow_angle)

    video_obj.shot_start_frame = start_frame
    video_obj.shot_end_frame = end_frame


def output_landmark_video(video_obj):
    cap = cv2.VideoCapture(video_obj.path)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Define custom colors (RGB)
    landmark_color = (255, 20, 147)  # Pink for landmarks
    connection_color = (124, 252, 0)  # Grass green for connections

    drawing_spec = mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2)
    connection_spec = mp_drawing.DrawingSpec(color=connection_color, thickness=2)

    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output directory if it doesn't exist

    output_video_path = os.path.join(output_folder, video_obj.name + '_landmarks.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_obj.frame_rate, (video_obj.width, video_obj.height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw landmarks
        if frame_idx < len(video_obj.landmarks):
            landmarks = video_obj.landmarks[frame_idx]
            mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS, 
                                      landmark_drawing_spec=drawing_spec, 
                                      connection_drawing_spec=connection_spec)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def generate_joint_angle_plots(video_obj, output_folder='output/plots'):
    joint_names = ['hip', 'knee', 'shoulder', 'elbow']
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(video_obj.original_joint_angles)):
        plt.figure(figsize=(10, 6))
        for j, joint_name in enumerate(joint_names):
            angles = [frame[j] for frame in video_obj.original_joint_angles[:i+1]]
            plt.plot(angles, label=joint_name)

        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Joint Angles Over Time')
        plt.legend()
        plt.xlim(0, len(video_obj.original_joint_angles))
        plt.ylim(0, 180)

        plot_filename = f'{output_folder}/{video_obj.name}_plot_{i:04d}.png'
        plt.savefig(plot_filename)
        plt.close()

def compile_plots_to_video(video_obj, input_folder='output/plots', output_folder='output'):
    plot_filenames = [f'{input_folder}/{video_obj.name}_plot_{i:04d}.png' for i in range(len(video_obj.original_joint_angles))]
    if not plot_filenames:
        return  # Exit if there are no plot files

    frame = cv2.imread(plot_filenames[0])
    height, width, layers = frame.shape

    output_video_path = os.path.join(output_folder, video_obj.name + '_joint_angles.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, video_obj.frame_rate, (width, height))

    for filename in plot_filenames:
        frame = cv2.imread(filename)
        video.write(frame)
        os.remove(filename)  # Delete the plot file after adding to video

    cv2.destroyAllWindows()
    video.release()

def combine_videos(video_obj, input_folder='output', output_folder='output'):
    landmark_video_path = os.path.join(input_folder, video_obj.name + '_landmarks.mp4')
    angle_video_path = os.path.join(input_folder, video_obj.name + '_joint_angles.mp4')
    output_video_path = os.path.join(output_folder, video_obj.name + '_combined.mp4')

    cap1 = cv2.VideoCapture(landmark_video_path)
    cap2 = cv2.VideoCapture(angle_video_path)

    if not (cap1.isOpened() and cap2.isOpened()):
        print("Error opening one of the videos")
        return

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_height = max(height1, height2)

    # Calculate new widths to maintain aspect ratio
    new_width1 = int(width1 * (output_height / height1))
    new_width2 = int(width2 * (output_height / height2))

    output_width = new_width1 + new_width2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_obj.frame_rate, (output_width, output_height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            break

        # Resize frames to have the same height and proportional width
        frame1 = cv2.resize(frame1, (new_width1, output_height))
        frame2 = cv2.resize(frame2, (new_width2, output_height))

        combined_frame = cv2.hconcat([frame1, frame2])
        out.write(combined_frame)

    cap1.release()
    cap2.release()
    out.release()



# Main script
video = select_video()
if video:
    detect_landmarks(video)
    calculate_all_joint_angles(video)
    find_shot_start_end(video)
    output_landmark_video(video)
    # Output statements...
    print(f"Video name: {video.name}")
    print(f"Video path: {video.path}")
    print(f"Original Frame rate: {video.frame_rate}")
    print(f"Rounded Frame rate: {video.frame_rate}")
    print(f"Frame count: {video.frame_count}")
    print(f"Resolution: {video.width}x{video.height}")
    print(f"Original Landmarks count: {len(video.landmarks)}")
    print(f"Normalized (30 FPS) Landmarks count: {len(video.normalized_landmarks)}")
    print(f"Original Joint Angles count: {len(video.original_joint_angles)}")
    print(f"Normalized (30 FPS) Joint Angles count: {len(video.normalized_joint_angles)}")

    print(f"starting at  {video.shot_start_frame}")
    print(f"release at {video.shot_end_frame}")
    generate_joint_angle_plots(video)
    compile_plots_to_video(video)
    combine_videos(video)
else:
    print("No video selected.")
