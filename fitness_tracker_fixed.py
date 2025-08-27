import cv2
import mediapipe as mp
import numpy as np
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fitness Tracking with Pose Detection')
    parser.add_argument('-v', '--video', type=str, default='Squats.mp4',
                       help='Path to input video file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Path to output video file')
    parser.add_argument('--det', type=float, default=0.5, 
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--track', type=float, default=0.5, 
                       help='Tracking confidence threshold (0.0-1.0)')
    parser.add_argument('-c', '--complexity', type=int, default=1, 
                       help='Model complexity: 0=light, 1=full, 2=heavy')
    parser.add_argument('-wt', '--workout_type', type=str, default='PushUp',
                       choices=['PushUp', 'Squat', 'General'],
                       help='Type of workout to track')
    parser.add_argument('--cam', action='store_true',
                       help='Use camera instead of video file')
    
    return parser.parse_args()


def calculate_angle(a, b, c):
    """Calculate angle between three points (a-b-c)"""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0


def get_landmark_coords(landmarks, landmark_type, image_width, image_height):
    """Extract landmark coordinates safely"""
    try:
        landmark = landmarks[landmark_type.value]
        return [landmark.x * image_width, landmark.y * image_height]
    except (AttributeError, IndexError):
        return [0, 0]


class WorkoutTracker:
    def __init__(self, workout_type='PushUp'):
        self.workout_type = workout_type
        self.reset_counters()
        
    def reset_counters(self):
        """Reset all workout counters"""
        self.pushup_counter = 0
        self.squat_counter = 0
        
        # Push-up state tracking
        self.pushup_up_pos = False
        self.pushup_down_pos = False
        
        # Squat state tracking
        self.squat_up_pos = False
        self.squat_down_pos = False
        
    def update_pushup_count(self, left_elbow_angle, right_elbow_angle):
        """Update push-up counter based on elbow angles"""
        # Use average of both arms for more reliable detection
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # Up position: arms extended
        if avg_elbow_angle > 160 and not self.pushup_up_pos:
            self.pushup_up_pos = True
            self.pushup_down_pos = False
            
        # Down position: arms bent
        elif avg_elbow_angle < 90 and self.pushup_up_pos and not self.pushup_down_pos:
            self.pushup_down_pos = True
            
        # Complete rep: back to up position
        elif avg_elbow_angle > 160 and self.pushup_down_pos:
            self.pushup_counter += 1
            self.pushup_up_pos = True
            self.pushup_down_pos = False
            
    def update_squat_count(self, left_knee_angle, right_knee_angle):
        """Update squat counter based on knee angles"""
        # Use average of both knees for more reliable detection
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # Up position: legs extended
        if avg_knee_angle > 160 and not self.squat_up_pos:
            self.squat_up_pos = True
            self.squat_down_pos = False
            
        # Down position: knees bent (squat)
        elif avg_knee_angle < 120 and self.squat_up_pos and not self.squat_down_pos:
            self.squat_down_pos = True
            
        # Complete rep: back to up position
        elif avg_knee_angle > 160 and self.squat_down_pos:
            self.squat_counter += 1
            self.squat_up_pos = True
            self.squat_down_pos = False


def draw_custom_skeleton(image, landmarks, mp_pose, image_width, image_height, line_color):
    """Draw custom skeleton with enhanced visibility"""
    # Get key landmark coordinates
    nose = get_landmark_coords(landmarks, mp_pose.PoseLandmark.NOSE, image_width, image_height)
    left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, image_width, image_height)
    right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, image_width, image_height)
    left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP, image_width, image_height)
    right_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, image_width, image_height)
    
    # Calculate midpoints
    shoulder_mid = [(left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2]
    hip_mid = [(left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2]
    neck_point = [(nose[0] + shoulder_mid[0]) // 2, (nose[1] + shoulder_mid[1]) // 2]
    torso_mid = [(shoulder_mid[0] + hip_mid[0]) // 2, (shoulder_mid[1] + hip_mid[1]) // 2]
    
    # Draw custom lines
    cv2.line(image, tuple(map(int, left_shoulder)), tuple(map(int, neck_point)), line_color, 3)
    cv2.line(image, tuple(map(int, right_shoulder)), tuple(map(int, neck_point)), line_color, 3)
    cv2.line(image, tuple(map(int, neck_point)), tuple(map(int, torso_mid)), line_color, 3)
    cv2.line(image, tuple(map(int, torso_mid)), tuple(map(int, left_hip)), line_color, 3)
    cv2.line(image, tuple(map(int, torso_mid)), tuple(map(int, right_hip)), line_color, 3)
    
    # Draw key points
    cv2.circle(image, tuple(map(int, neck_point)), 6, line_color, -1)
    cv2.circle(image, tuple(map(int, left_shoulder)), 4, line_color, -1)
    cv2.circle(image, tuple(map(int, right_shoulder)), 4, line_color, -1)
    cv2.circle(image, tuple(map(int, torso_mid)), 6, line_color, -1)


def draw_info_panel(image, angles, tracker, fps):
    """Draw information panel with angles and counters"""
    height, width = image.shape[:2]
    panel_width = 350
    panel_height = 300
    
    # Create semi-transparent panel
    overlay = image.copy()
    cv2.rectangle(overlay, (width - panel_width, 0), (width, panel_height), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
    
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 255)
    thickness = 2
    
    # Title
    cv2.putText(image, 'Workout Tracker', (width - panel_width + 10, y_offset), 
                cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)
    y_offset += 40
    
    # Angles
    cv2.putText(image, f'L Elbow: {angles["left_elbow"]}°', 
                (width - panel_width + 10, y_offset), font, font_scale, color, thickness)
    y_offset += 30
    cv2.putText(image, f'R Elbow: {angles["right_elbow"]}°', 
                (width - panel_width + 10, y_offset), font, font_scale, color, thickness)
    y_offset += 30
    cv2.putText(image, f'L Knee: {angles["left_knee"]}°', 
                (width - panel_width + 10, y_offset), font, font_scale, color, thickness)
    y_offset += 30
    cv2.putText(image, f'R Knee: {angles["right_knee"]}°', 
                (width - panel_width + 10, y_offset), font, font_scale, color, thickness)
    y_offset += 40
    
    # Counters
    cv2.putText(image, f'Push-ups: {tracker.pushup_counter}', 
                (width - panel_width + 10, y_offset), font, 0.8, (0, 255, 0), 2)
    y_offset += 30
    cv2.putText(image, f'Squats: {tracker.squat_counter}', 
                (width - panel_width + 10, y_offset), font, 0.8, (0, 255, 0), 2)
    y_offset += 30
    
    # FPS
    cv2.putText(image, f'FPS: {int(fps)}', 
                (width - panel_width + 10, y_offset), font, font_scale, (255, 255, 255), thickness)


def main():
    args = parse_args()
    
    # Initialize MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Colors
    line_color = (255, 255, 255)
    
    # Drawing specifications
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=line_color)
    connection_spec = mp_drawing.DrawingSpec(thickness=2, color=line_color)
    
    # Initialize workout tracker
    tracker = WorkoutTracker(args.workout_type)
    
    # Initialize video capture
    if args.cam:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(args.video)
    
    if not vid.isOpened():
        print(f"Error: Could not open {'camera' if args.cam else f'video file {args.video}'}")
        return
    
    # Get video properties
    if not args.cam:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
    else:
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        width, height, fps = 1280, 720, 30
    
    # Initialize video writer
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"Starting fitness tracking - Workout type: {args.workout_type}")
    print("Press 'q' or ESC to quit, 'r' to reset counters")
    
    # Initialize pose detection
    with mp_pose.Pose(
        min_detection_confidence=args.det,
        min_tracking_confidence=args.track,
        model_complexity=args.complexity,
        smooth_landmarks=True) as pose:
        
        while vid.isOpened():
            frame_start_time = time.time()
            
            success, image = vid.read()
            if not success:
                if args.cam:
                    print("Failed to read from camera")
                    continue
                else:
                    print("End of video")
                    break
            
            # Get current frame dimensions
            image_height, image_width, _ = image.shape
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            
            # Process pose
            results = pose.process(rgb_image)
            
            # Convert back to BGR
            image.flags.writeable = True
            
            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Hide face landmarks for privacy
                    face_landmarks = [
                        mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE,
                        mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE_INNER,
                        mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
                        mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.MOUTH_LEFT,
                        mp_pose.PoseLandmark.MOUTH_RIGHT, mp_pose.PoseLandmark.LEFT_EAR,
                        mp_pose.PoseLandmark.RIGHT_EAR
                    ]
                    
                    for landmark in face_landmarks:
                        landmarks[landmark.value].visibility = 0
                    
                    # Get landmark coordinates
                    left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, image_width, image_height)
                    right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, image_width, image_height)
                    left_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, image_width, image_height)
                    right_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, image_width, image_height)
                    left_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, image_width, image_height)
                    right_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, image_width, image_height)
                    left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP, image_width, image_height)
                    right_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, image_width, image_height)
                    left_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, image_width, image_height)
                    right_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, image_width, image_height)
                    left_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, image_width, image_height)
                    right_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, image_width, image_height)
                    
                    # Calculate angles
                    angles = {
                        "left_elbow": int(calculate_angle(left_shoulder, left_elbow, left_wrist)),
                        "right_elbow": int(calculate_angle(right_shoulder, right_elbow, right_wrist)),
                        "left_knee": int(calculate_angle(left_hip, left_knee, left_ankle)),
                        "right_knee": int(calculate_angle(right_hip, right_knee, right_ankle))
                    }
                    
                    # Update workout counters
                    if args.workout_type == 'PushUp' or args.workout_type == 'General':
                        tracker.update_pushup_count(angles["left_elbow"], angles["right_elbow"])
                    
                    if args.workout_type == 'Squat' or args.workout_type == 'General':
                        tracker.update_squat_count(angles["left_knee"], angles["right_knee"])
                    
                    # Draw custom skeleton
                    draw_custom_skeleton(image, landmarks, mp_pose, image_width, image_height, line_color)
                    
                    # Draw MediaPipe pose landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        drawing_spec,
                        connection_drawing_spec=connection_spec
                    )
                    
                    # Calculate FPS
                    current_fps = 1.0 / (time.time() - frame_start_time) if (time.time() - frame_start_time) > 0 else 0
                    
                    # Draw info panel
                    draw_info_panel(image, angles, tracker, current_fps)
                    
                else:
                    # No pose detected
                    cv2.putText(image, 'No pose detected', (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    current_fps = 1.0 / (time.time() - frame_start_time) if (time.time() - frame_start_time) > 0 else 0
                    cv2.putText(image, f'FPS: {int(current_fps)}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            except Exception as e:
                print(f"Error processing frame: {e}")
                current_fps = 1.0 / (time.time() - frame_start_time) if (time.time() - frame_start_time) > 0 else 0
                cv2.putText(image, f'Processing Error - FPS: {int(current_fps)}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write frame to output video
            if out is not None:
                out.write(image)
            
            # Display frame
            display_scale = 0.8 if not args.cam else 1.0
            display_frame = cv2.resize(image, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow('Fitness Tracker', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif key == ord('r'):  # 'r' to reset counters
                tracker.reset_counters()
                print("Counters reset!")
    
    # Cleanup
    vid.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"Push-ups: {tracker.pushup_counter}")
    print(f"Squats: {tracker.squat_counter}")


if __name__ == "__main__":
    main()