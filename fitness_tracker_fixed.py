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
    parser.add_argument('--det', type=float, default=0.7, 
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--track', type=float, default=0.7, 
                       help='Tracking confidence threshold (0.0-1.0)')
    parser.add_argument('-c', '--complexity', type=int, default=1, 
                       help='Model complexity: 0=light, 1=full, 2=heavy')
    parser.add_argument('-wt', '--workout_type', type=str, default='General',
                       choices=['PushUp', 'Squat', 'General'],
                       help='Type of workout to track')
    parser.add_argument('--cam', action='store_true',
                       help='Use camera instead of video file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed output')
    
    return parser.parse_args()


def calculate_angle(a, b, c):
    """Calculate angle between three points with validation"""
    try:
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        c = np.array(c, dtype=np.float32)

        # Validate input points
        if np.any(a == 0) and np.any(b == 0) and np.any(c == 0):
            return 0

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    except Exception as e:
        if args.debug:
            print(f"Angle calculation error: {e}")
        return 0


def get_landmark_coords(landmarks, landmark_idx, width, height):
    """Extract landmark coordinates with better error handling"""
    try:
        if landmarks is None or len(landmarks) <= landmark_idx:
            return [0, 0]
            
        landmark = landmarks[landmark_idx]
        
        # Check visibility threshold
        if hasattr(landmark, 'visibility') and landmark.visibility < 0.5:
            return [0, 0]
            
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        
        # Boundary check
        if 0 <= x <= width and 0 <= y <= height:
            return [x, y]
        else:
            return [0, 0]
            
    except Exception as e:
        if args.debug:
            print(f"Landmark extraction error: {e}")
        return [0, 0]


class WorkoutTracker:
    def __init__(self, workout_type='General'):
        self.workout_type = workout_type
        self.reset_counters()
        
    def reset_counters(self):
        """Reset all counters and states"""
        self.pushup_counter = 0
        self.squat_counter = 0
        self.pushup_stage = 'up'  # 'up' or 'down'
        self.squat_stage = 'up'   # 'up' or 'down'
        
    def update_pushup_count(self, left_angle, right_angle):
        """Update pushup counter based on arm angles"""
        # Use the average angle for more stability
        avg_angle = (left_angle + right_angle) / 2
        
        # Push-up detection logic
        if avg_angle > 160:
            self.pushup_stage = 'up'
        elif avg_angle < 90 and self.pushup_stage == 'up':
            self.pushup_stage = 'down'
            self.pushup_counter += 1
            print(f"Push-up completed! Count: {self.pushup_counter}")
            
    def update_squat_count(self, left_angle, right_angle):
        """Update squat counter based on knee angles"""
        # Use the average angle for more stability
        avg_angle = (left_angle + right_angle) / 2
        
        # Squat detection logic
        if avg_angle > 160:
            self.squat_stage = 'up'
        elif avg_angle < 90 and self.squat_stage == 'up':
            self.squat_stage = 'down'
            self.squat_counter += 1
            print(f"Squat completed! Count: {self.squat_counter}")


def draw_landmarks_and_connections(image, landmarks, mp_pose, mp_drawing):
    """Draw pose landmarks and connections with better visibility"""
    if landmarks:
        # Custom drawing specs for better visibility
        landmark_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=4, circle_radius=4)
        connection_spec = mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=3)
            
        mp_drawing.draw_landmarks(
            image, landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_spec, connection_spec)


def draw_angle_info(image, point, angle, label):
    """Draw angle information at specific points"""
    if point != [0, 0] and angle > 0:
        angle_text = f'{int(angle)}'
        cv2.putText(image, angle_text, tuple(point), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def create_info_panel(image, angles, tracker, fps):
    """Create information panel with workout stats"""
    height, width = image.shape[:2]
    
    # Panel dimensions
    panel_width = 300
    panel_height = height
    
    # Create overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (width - panel_width, 0), 
                 (width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    small_font = cv2.FONT_HERSHEY_SIMPLEX
    y = 40
    
    # Title
    cv2.putText(image, 'FITNESS TRACKER', (width - 290, y), 
               font, 0.8, (255, 255, 255), 2)
    y += 50
    
    # Workout type
    cv2.putText(image, f'Mode: {tracker.workout_type}', (width - 290, y), 
               small_font, 0.6, (0, 255, 255), 2)
    y += 40
    
    # Angles
    cv2.putText(image, 'JOINT ANGLES:', (width - 290, y), 
               small_font, 0.6, (255, 200, 0), 2)
    y += 30
    
    angle_data = [
        f'Left Elbow: {int(angles["left_elbow"])}°',
        f'Right Elbow: {int(angles["right_elbow"])}°',
        f'Left Knee: {int(angles["left_knee"])}°',
        f'Right Knee: {int(angles["right_knee"])}°'
    ]
    
    for angle_text in angle_data:
        cv2.putText(image, angle_text, (width - 280, y), 
                   small_font, 0.5, (255, 255, 255), 1)
        y += 25
    
    y += 20
    
    # Exercise counters
    cv2.putText(image, 'EXERCISE COUNT:', (width - 290, y), 
               small_font, 0.6, (255, 200, 0), 2)
    y += 30
    
    if tracker.workout_type in ['PushUp', 'General']:
        cv2.putText(image, f'Push-ups: {tracker.pushup_counter}', 
                   (width - 280, y), font, 0.7, (0, 255, 0), 2)
        y += 35
        cv2.putText(image, f'Stage: {tracker.pushup_stage}', 
                   (width - 280, y), small_font, 0.5, (200, 200, 200), 1)
        y += 25
    
    if tracker.workout_type in ['Squat', 'General']:
        cv2.putText(image, f'Squats: {tracker.squat_counter}', 
                   (width - 280, y), font, 0.7, (0, 255, 0), 2)
        y += 35
        cv2.putText(image, f'Stage: {tracker.squat_stage}', 
                   (width - 280, y), small_font, 0.5, (200, 200, 200), 1)
        y += 25
    
    # FPS and status
    y = height - 60
    cv2.putText(image, f'FPS: {int(fps)}', (width - 280, y), 
               small_font, 0.6, (255, 255, 255), 2)
    y += 25
    cv2.putText(image, 'Press Q to quit, R to reset', (width - 290, y), 
               small_font, 0.4, (200, 200, 200), 1)


def main():
    global args
    args = parse_args()
    
    print(f"Starting Fitness Tracker v2.0")
    print(f"Workout Type: {args.workout_type}")
    print(f"Detection Confidence: {args.det}")
    print(f"Tracking Confidence: {args.track}")
    print(f"Model Complexity: {args.complexity}")
    print("-" * 40)
    
    # Initialize MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Initialize tracker
    tracker = WorkoutTracker(args.workout_type)
    
    # Setup video capture
    if args.cam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open {'camera' if args.cam else args.video}")
        return
    
    # Video properties
    if not args.cam:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video FPS: {fps}, Total frames: {frame_count}")
    
    # Output video writer
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_out = fps if not args.cam else 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps_out, (frame_width, frame_height))
    
    print("Starting pose detection...")
    print("Controls: Q=quit, R=reset, SPACE=pause")
    
    paused = False
    frame_num = 0
    
    # Initialize pose estimation
    with mp_pose.Pose(
        min_detection_confidence=args.det,
        min_tracking_confidence=args.track,
        model_complexity=args.complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True) as pose:
        
        while cap.isOpened():
            start_time = time.time()
            
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or camera error")
                    break
                
                frame_num += 1
                if args.debug and frame_num % 30 == 0:
                    print(f"Processing frame {frame_num}")
            
            # Get frame dimensions
            height, width, channels = frame.shape
            
            if not paused:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                
                # Process pose
                results = pose.process(rgb_frame)
                
                # Convert back to BGR
                rgb_frame.flags.writeable = True
            
            # Initialize angles
            angles = {
                "left_elbow": 0,
                "right_elbow": 0, 
                "left_knee": 0,
                "right_knee": 0
            }
            
            # Process results
            if not paused and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get landmark coordinates
                coords = {}
                landmark_mapping = {
                    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                    'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST.value,
                    'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST.value,
                    'left_hip': mp_pose.PoseLandmark.LEFT_HIP.value,
                    'right_hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
                    'left_knee': mp_pose.PoseLandmark.LEFT_KNEE.value,
                    'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE.value,
                    'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE.value,
                    'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                }
                
                for name, idx in landmark_mapping.items():
                    coords[name] = get_landmark_coords(landmarks, idx, width, height)
                
                # Calculate angles
                angles["left_elbow"] = calculate_angle(
                    coords['left_shoulder'], coords['left_elbow'], coords['left_wrist'])
                angles["right_elbow"] = calculate_angle(
                    coords['right_shoulder'], coords['right_elbow'], coords['right_wrist'])
                angles["left_knee"] = calculate_angle(
                    coords['left_hip'], coords['left_knee'], coords['left_ankle'])
                angles["right_knee"] = calculate_angle(
                    coords['right_hip'], coords['right_knee'], coords['right_ankle'])
                
                # Update exercise counters
                if args.workout_type in ['PushUp', 'General']:
                    tracker.update_pushup_count(angles["left_elbow"], angles["right_elbow"])
                
                if args.workout_type in ['Squat', 'General']:
                    tracker.update_squat_count(angles["left_knee"], angles["right_knee"])
                
                # Draw pose landmarks
                draw_landmarks_and_connections(frame, results.pose_landmarks, mp_pose, mp_drawing)
                
                # Draw angle info on joints
                draw_angle_info(frame, coords['left_elbow'], angles["left_elbow"], 'LE')
                draw_angle_info(frame, coords['right_elbow'], angles["right_elbow"], 'RE')
                draw_angle_info(frame, coords['left_knee'], angles["left_knee"], 'LK')
                draw_angle_info(frame, coords['right_knee'], angles["right_knee"], 'RK')
                
            else:
                # No pose detected
                cv2.putText(frame, 'No pose detected', (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Calculate FPS
            processing_time = time.time() - start_time
            current_fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Create info panel
            create_info_panel(frame, angles, tracker, current_fps)
            
            # Add pause indicator
            if paused:
                cv2.putText(frame, 'PAUSED', (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            
            # Write output frame
            if out is not None and not paused:
                out.write(frame)
            
            # Display frame
            cv2.imshow('Fitness Tracker', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):  # R for reset
                tracker.reset_counters()
                print("Counters reset!")
            elif key == ord(' '):  # SPACE for pause
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Final results
    print(f"\nSession Complete!")
    print(f"Final Results:")
    print(f"- Push-ups: {tracker.pushup_counter}")
    print(f"- Squats: {tracker.squat_counter}")


if __name__ == "__main__":
    main()