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
    parser.add_argument('-wt', '--workout_type', type=str, default='General',
                       choices=['PushUp', 'Squat', 'General'],
                       help='Type of workout to track')
    parser.add_argument('--cam', action='store_true',
                       help='Use camera instead of video file')
    
    return parser.parse_args()


def calculate_angle(a, b, c):
    """Calculate angle between three points (a-b-c) with error handling"""
    try:
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        c = np.array(c, dtype=np.float32)

        # Check for invalid coordinates
        if np.any(np.isnan([a, b, c])) or np.any(np.isinf([a, b, c])):
            return 0

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0


def get_landmark_coords(landmarks, landmark_type, image_width, image_height):
    """Extract landmark coordinates safely with validation"""
    try:
        if landmarks is None:
            return [0, 0]
        
        landmark = landmarks[landmark_type.value]
        
        # Check if landmark is visible and coordinates are valid
        if landmark.visibility < 0.5:
            return [0, 0]
            
        x = landmark.x * image_width
        y = landmark.y * image_height
        
        # Validate coordinates are within frame bounds
        if 0 <= x <= image_width and 0 <= y <= image_height:
            return [x, y]
        else:
            return [0, 0]
            
    except (AttributeError, IndexError, TypeError) as e:
        print(f"Error extracting landmark coordinates: {e}")
        return [0, 0]


class WorkoutTracker:
    def __init__(self, workout_type='General'):
        self.workout_type = workout_type
        self.reset_counters()
        
    def reset_counters(self):
        """Reset all workout counters and states"""
        self.pushup_counter = 0
        self.squat_counter = 0
        
        # Push-up state tracking with improved logic
        self.pushup_state = 'unknown'  # 'up', 'down', 'unknown'
        self.pushup_angle_buffer = []  # Buffer for angle smoothing
        
        # Squat state tracking
        self.squat_state = 'unknown'  # 'up', 'down', 'unknown'
        self.squat_angle_buffer = []
        
        # Timing for preventing rapid counting
        self.last_pushup_time = 0
        self.last_squat_time = 0
        self.min_rep_interval = 0.5  # Minimum seconds between reps
        
    def smooth_angle(self, angle, buffer, buffer_size=5):
        """Smooth angles using a moving average"""
        if len(buffer) >= buffer_size:
            buffer.pop(0)
        buffer.append(angle)
        return sum(buffer) / len(buffer)
        
    def update_pushup_count(self, left_elbow_angle, right_elbow_angle):
        """Update push-up counter with improved state machine"""
        current_time = time.time()
        
        # Prevent rapid counting
        if current_time - self.last_pushup_time < self.min_rep_interval:
            return
            
        # Use average of both arms and smooth the angle
        avg_angle = (left_elbow_angle + right_elbow_angle) / 2
        smoothed_angle = self.smooth_angle(avg_angle, self.pushup_angle_buffer)
        
        # State machine for push-up detection
        if smoothed_angle > 150 and self.pushup_state != 'up':
            self.pushup_state = 'up'
            
        elif smoothed_angle < 100 and self.pushup_state == 'up':
            self.pushup_state = 'down'
            
        elif smoothed_angle > 150 and self.pushup_state == 'down':
            self.pushup_counter += 1
            self.pushup_state = 'up'
            self.last_pushup_time = current_time
            print(f"Push-up completed! Count: {self.pushup_counter}")
            
    def update_squat_count(self, left_knee_angle, right_knee_angle):
        """Update squat counter with improved state machine"""
        current_time = time.time()
        
        # Prevent rapid counting
        if current_time - self.last_squat_time < self.min_rep_interval:
            return
            
        # Use average of both knees and smooth the angle
        avg_angle = (left_knee_angle + right_knee_angle) / 2
        smoothed_angle = self.smooth_angle(avg_angle, self.squat_angle_buffer)
        
        # State machine for squat detection
        if smoothed_angle > 150 and self.squat_state != 'up':
            self.squat_state = 'up'
            
        elif smoothed_angle < 110 and self.squat_state == 'up':
            self.squat_state = 'down'
            
        elif smoothed_angle > 150 and self.squat_state == 'down':
            self.squat_counter += 1
            self.squat_state = 'up'
            self.last_squat_time = current_time
            print(f"Squat completed! Count: {self.squat_counter}")


def draw_custom_skeleton(image, landmarks, mp_pose, image_width, image_height, line_color):
    """Draw custom skeleton with enhanced visibility and error handling"""
    try:
        # Get key landmark coordinates with validation
        key_points = {
            'nose': get_landmark_coords(landmarks, mp_pose.PoseLandmark.NOSE, image_width, image_height),
            'left_shoulder': get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, image_width, image_height),
            'right_shoulder': get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, image_width, image_height),
            'left_hip': get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP, image_width, image_height),
            'right_hip': get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, image_width, image_height)
        }
        
        # Check if we have valid coordinates
        invalid_points = [k for k, v in key_points.items() if v == [0, 0]]
        if len(invalid_points) > 2:  # Too many invalid points
            return
            
        # Calculate midpoints safely
        shoulder_mid = [
            (key_points['left_shoulder'][0] + key_points['right_shoulder'][0]) // 2,
            (key_points['left_shoulder'][1] + key_points['right_shoulder'][1]) // 2
        ]
        hip_mid = [
            (key_points['left_hip'][0] + key_points['right_hip'][0]) // 2,
            (key_points['left_hip'][1] + key_points['right_hip'][1]) // 2
        ]
        neck_point = [
            (key_points['nose'][0] + shoulder_mid[0]) // 2,
            (key_points['nose'][1] + shoulder_mid[1]) // 2
        ]
        torso_mid = [
            (shoulder_mid[0] + hip_mid[0]) // 2,
            (shoulder_mid[1] + hip_mid[1]) // 2
        ]
        
        # Draw custom lines with validation
        def safe_draw_line(img, pt1, pt2, color, thickness):
            if pt1 != [0, 0] and pt2 != [0, 0]:
                cv2.line(img, tuple(map(int, pt1)), tuple(map(int, pt2)), color, thickness)
                
        def safe_draw_circle(img, center, radius, color, thickness):
            if center != [0, 0]:
                cv2.circle(img, tuple(map(int, center)), radius, color, thickness)
        
        # Draw skeleton structure
        safe_draw_line(image, key_points['left_shoulder'], neck_point, line_color, 3)
        safe_draw_line(image, key_points['right_shoulder'], neck_point, line_color, 3)
        safe_draw_line(image, neck_point, torso_mid, line_color, 3)
        safe_draw_line(image, torso_mid, key_points['left_hip'], line_color, 3)
        safe_draw_line(image, torso_mid, key_points['right_hip'], line_color, 3)
        
        # Draw key points
        safe_draw_circle(image, neck_point, 6, line_color, -1)
        safe_draw_circle(image, key_points['left_shoulder'], 4, line_color, -1)
        safe_draw_circle(image, key_points['right_shoulder'], 4, line_color, -1)
        safe_draw_circle(image, torso_mid, 6, line_color, -1)
        
    except Exception as e:
        print(f"Error drawing custom skeleton: {e}")


def draw_info_panel(image, angles, tracker, fps, workout_type):
    """Draw information panel with angles, counters, and workout status"""
    try:
        height, width = image.shape[:2]
        panel_width = 400
        panel_height = 350
        
        # Create semi-transparent panel
        overlay = image.copy()
        cv2.rectangle(overlay, (width - panel_width, 0), (width, panel_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)
        
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 255)
        thickness = 2
        
        # Title
        cv2.putText(image, f'Fitness Tracker - {workout_type}', 
                   (width - panel_width + 10, y_offset), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 40
        
        # Angles section
        cv2.putText(image, 'Joint Angles:', (width - panel_width + 10, y_offset), 
                   font, font_scale, (255, 200, 0), thickness)
        y_offset += 30
        
        angle_texts = [
            f'L Elbow: {angles["left_elbow"]}°',
            f'R Elbow: {angles["right_elbow"]}°', 
            f'L Knee: {angles["left_knee"]}°',
            f'R Knee: {angles["right_knee"]}°'
        ]
        
        for text in angle_texts:
            cv2.putText(image, text, (width - panel_width + 20, y_offset), 
                       font, font_scale, color, thickness)
            y_offset += 25
        
        y_offset += 15
        
        # Counters section
        cv2.putText(image, 'Exercise Counts:', (width - panel_width + 10, y_offset), 
                   font, font_scale, (255, 200, 0), thickness)
        y_offset += 30
        
        if workout_type in ['PushUp', 'General']:
            cv2.putText(image, f'Push-ups: {tracker.pushup_counter}', 
                       (width - panel_width + 20, y_offset), font, 0.8, (0, 255, 0), 2)
            y_offset += 30
            
        if workout_type in ['Squat', 'General']:
            cv2.putText(image, f'Squats: {tracker.squat_counter}', 
                       (width - panel_width + 20, y_offset), font, 0.8, (0, 255, 0), 2)
            y_offset += 30
        
        # Status and FPS
        y_offset += 10
        cv2.putText(image, f'FPS: {int(fps)}', (width - panel_width + 20, y_offset), 
                   font, font_scale, (255, 255, 255), thickness)
        y_offset += 25
        
        # Current state display
        if hasattr(tracker, 'pushup_state'):
            cv2.putText(image, f'Push-up State: {tracker.pushup_state}', 
                       (width - panel_width + 20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
            
        if hasattr(tracker, 'squat_state'):
            cv2.putText(image, f'Squat State: {tracker.squat_state}', 
                       (width - panel_width + 20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
    except Exception as e:
        print(f"Error drawing info panel: {e}")


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
    
    # Initialize video capture with error handling
    try:
        if args.cam:
            vid = cv2.VideoCapture(0)
            if not vid.isOpened():
                print("Error: Could not open camera")
                return
            # Set camera properties
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            width, height, fps = 1280, 720, 30
        else:
            vid = cv2.VideoCapture(args.video)
            if not vid.isOpened():
                print(f"Error: Could not open video file {args.video}")
                return
            # Get video properties
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            
    except Exception as e:
        print(f"Error initializing video capture: {e}")
        return
    
    # Initialize video writer with better codec
    out = None
    if args.output:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Better codec
            out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
            if not out.isOpened():
                print("Warning: Could not initialize video writer")
                out = None
        except Exception as e:
            print(f"Error initializing video writer: {e}")
    
    print(f"Starting fitness tracking - Workout type: {args.workout_type}")
    print(f"Video resolution: {width}x{height} at {fps} FPS")
    print("Controls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Press 'r' to reset counters")
    print("  - Press SPACE to pause/resume")
    
    paused = False
    
    # Initialize pose detection with error handling
    try:
        with mp_pose.Pose(
            min_detection_confidence=args.det,
            min_tracking_confidence=args.track,
            model_complexity=args.complexity,
            smooth_landmarks=True) as pose:
            
            while vid.isOpened():
                frame_start_time = time.time()
                
                if not paused:
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
                
                if not paused:
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
                        
                        # Get all required landmark coordinates
                        coords = {
                            'left_shoulder': get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, image_width, image_height),
                            'right_shoulder': get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, image_width, image_height),
                            'left_elbow': get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, image_width, image_height),
                            'right_elbow': get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, image_width, image_height),
                            'left_wrist': get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, image_width, image_height),
                            'right_wrist': get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, image_width, image_height),
                            'left_hip': get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP, image_width, image_height),
                            'right_hip': get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, image_width, image_height),
                            'left_knee': get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, image_width, image_height),
                            'right_knee': get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, image_width, image_height),
                            'left_ankle': get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, image_width, image_height),
                            'right_ankle': get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, image_width, image_height)
                        }
                        
                        # Calculate angles with error handling
                        angles = {
                            "left_elbow": int(calculate_angle(coords['left_shoulder'], coords['left_elbow'], coords['left_wrist'])),
                            "right_elbow": int(calculate_angle(coords['right_shoulder'], coords['right_elbow'], coords['right_wrist'])),
                            "left_knee": int(calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])),
                            "right_knee": int(calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle']))
                        }
                        
                        # Update workout counters only if not paused
                        if not paused:
                            if args.workout_type in ['PushUp', 'General']:
                                tracker.update_pushup_count(angles["left_elbow"], angles["right_elbow"])
                            
                            if args.workout_type in ['Squat', 'General']:
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
                        
                    else:
                        # No pose detected
                        angles = {"left_elbow": 0, "right_elbow": 0, "left_knee": 0, "right_knee": 0}
                        cv2.putText(image, 'No pose detected', (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    angles = {"left_elbow": 0, "right_elbow": 0, "left_knee": 0, "right_knee": 0}
                
                # Calculate FPS
                frame_time = time.time() - frame_start_time
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                
                # Draw info panel
                draw_info_panel(image, angles, tracker, current_fps, args.workout_type)
                
                # Add pause indicator
                if paused:
                    cv2.putText(image, 'PAUSED - Press SPACE to resume', (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Write frame to output video (only if not paused)
                if out is not None and not paused:
                    out.write(image)
                
                # Display frame with adaptive scaling
                display_scale = 0.7 if not args.cam else 1.0
                display_frame = cv2.resize(image, (0, 0), fx=display_scale, fy=display_scale)
                cv2.imshow('Fitness Tracker', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                    break
                elif key == ord('r'):  # 'r' to reset counters
                    tracker.reset_counters()
                    print("Counters reset!")
                elif key == ord(' '):  # SPACE to pause/resume
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
    
    except Exception as e:
        print(f"Error in main loop: {e}")
    
    finally:
        # Cleanup
        if 'vid' in locals():
            vid.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Print final results
        print(f"\nFinal Results:")
        print(f"Push-ups: {tracker.pushup_counter}")
        print(f"Squats: {tracker.squat_counter}")
        print("Session completed!")


if __name__ == "__main__":
    main()