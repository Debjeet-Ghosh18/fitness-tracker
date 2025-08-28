import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
from collections import deque
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Advanced Fitness Tracking with Pose Detection')
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
                       choices=['PushUp', 'Squat', 'PullUp', 'General'],
                       help='Type of workout to track')
    parser.add_argument('--cam', action='store_true',
                       help='Use camera instead of video file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed output')
    parser.add_argument('--save_data', action='store_true',
                       help='Save workout data to JSON file')
    parser.add_argument('--calibrate', action='store_true',
                       help='Enable automatic threshold calibration')
    
    return parser.parse_args()


class AngleCalculator:
    """Enhanced angle calculation with multiple methods"""
    
    @staticmethod
    def calculate_angle_vector(a, b, c):
        """Calculate angle using vector method (most accurate)"""
        try:
            a = np.array(a, dtype=np.float32)
            b = np.array(b, dtype=np.float32)
            c = np.array(c, dtype=np.float32)

            # Validate input points
            if np.any(np.isnan([a, b, c])) or np.any(a == 0) and np.any(b == 0) and np.any(c == 0):
                return 0

            ba = a - b
            bc = c - b

            # Avoid division by zero
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            
            if norm_ba == 0 or norm_bc == 0:
                return 0

            cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        except Exception as e:
            return 0

    @staticmethod
    def calculate_angle_atan2(a, b, c):
        """Alternative angle calculation using atan2"""
        try:
            a = np.array(a, dtype=np.float32)
            b = np.array(b, dtype=np.float32)
            c = np.array(c, dtype=np.float32)

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            return angle
        except Exception:
            return 0


class LandmarkExtractor:
    """Enhanced landmark extraction with validation"""
    
    def __init__(self, mp_pose):
        self.mp_pose = mp_pose
        
    def get_landmark_coords(self, landmarks, landmark_idx, width, height):
        """Extract landmark coordinates with comprehensive validation"""
        try:
            if landmarks is None or len(landmarks) <= landmark_idx:
                return [0, 0]
                
            landmark = landmarks[landmark_idx]
            
            # Check visibility and presence thresholds
            visibility_threshold = 0.5
            presence_threshold = 0.5
            
            if (hasattr(landmark, 'visibility') and landmark.visibility < visibility_threshold) or \
               (hasattr(landmark, 'presence') and landmark.presence < presence_threshold):
                return [0, 0]
                
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            
            # Boundary validation with tolerance
            tolerance = 10
            if -tolerance <= x <= width + tolerance and -tolerance <= y <= height + tolerance:
                return [max(0, min(x, width)), max(0, min(y, height))]
            else:
                return [0, 0]
                
        except Exception as e:
            return [0, 0]
    
    def extract_all_coords(self, landmarks, width, height):
        """Extract all required landmark coordinates"""
        coords = {}
        landmark_mapping = {
            'nose': self.mp_pose.PoseLandmark.NOSE.value,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP.value,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        }
        
        for name, idx in landmark_mapping.items():
            coords[name] = self.get_landmark_coords(landmarks, idx, width, height)
            
        return coords


class WorkoutTracker:
    """Advanced workout tracking with adaptive thresholds and statistics"""
    
    def __init__(self, workout_type='General', enable_calibration=False):
        self.workout_type = workout_type
        self.enable_calibration = enable_calibration
        self.reset_counters()
        self.setup_thresholds()
        self.workout_data = []
        
    def reset_counters(self):
        """Reset all counters and states"""
        self.pushup_counter = 0
        self.squat_counter = 0
        self.pullup_counter = 0
        
        # Enhanced state tracking
        self.pushup_stage = 'unknown'
        self.squat_stage = 'unknown'
        self.pullup_stage = 'unknown'
        
        # Angle smoothing buffers
        self.pushup_angle_buffer = deque(maxlen=10)
        self.squat_angle_buffer = deque(maxlen=10)
        self.pullup_angle_buffer = deque(maxlen=10)
        
        # Timing for rep validation
        self.last_pushup_time = 0
        self.last_squat_time = 0
        self.last_pullup_time = 0
        self.min_rep_interval = 0.8  # Minimum seconds between reps
        
        # Quality metrics
        self.rep_quality_scores = []
        self.form_feedback = []
        
    def setup_thresholds(self):
        """Setup exercise-specific thresholds"""
        self.thresholds = {
            'pushup': {
                'up_angle': 160,
                'down_angle': 90,
                'min_range': 50  # Minimum range of motion
            },
            'squat': {
                'up_angle': 170,
                'down_angle': 100,
                'min_range': 60
            },
            'pullup': {
                'up_angle': 40,   # Arms bent at top
                'down_angle': 160, # Arms straight at bottom
                'min_range': 80
            }
        }
        
        # Calibration data for adaptive thresholds
        self.calibration_data = {
            'pushup_angles': deque(maxlen=100),
            'squat_angles': deque(maxlen=100),
            'pullup_angles': deque(maxlen=100)
        }
        
    def smooth_angle(self, angle, buffer):
        """Smooth angles using weighted moving average"""
        if angle <= 0:
            return 0
            
        buffer.append(angle)
        if len(buffer) < 3:
            return angle
            
        # Weighted average - more recent values have higher weight
        weights = np.linspace(0.5, 1.0, len(buffer))
        weights = weights / np.sum(weights)
        
        return np.average(list(buffer), weights=weights)
    
    def calibrate_thresholds(self, exercise, angle):
        """Automatically adjust thresholds based on user's range of motion"""
        if not self.enable_calibration:
            return
            
        calibration_buffer = self.calibration_data.get(f'{exercise}_angles')
        if calibration_buffer is not None:
            calibration_buffer.append(angle)
            
            if len(calibration_buffer) >= 50:  # Enough data for calibration
                angles = np.array(list(calibration_buffer))
                
                # Calculate percentiles for adaptive thresholds
                p10, p90 = np.percentile(angles, [10, 90])
                range_motion = p90 - p10
                
                if range_motion > 30:  # Valid range of motion
                    if exercise == 'pushup':
                        self.thresholds['pushup']['down_angle'] = p10 + 10
                        self.thresholds['pushup']['up_angle'] = p90 - 10
                    elif exercise == 'squat':
                        self.thresholds['squat']['down_angle'] = p10 + 15
                        self.thresholds['squat']['up_angle'] = p90 - 10
    
    def assess_form_quality(self, left_angle, right_angle):
        """Assess form quality and provide feedback"""
        angle_difference = abs(left_angle - right_angle)
        
        if angle_difference < 10:
            return "Excellent form"
        elif angle_difference < 20:
            return "Good form"
        elif angle_difference < 30:
            return "Fair form - balance both sides"
        else:
            return "Poor form - significant imbalance"
    
    def update_pushup_count(self, left_elbow_angle, right_elbow_angle):
        """Enhanced pushup counting with form analysis"""
        current_time = time.time()
        
        if current_time - self.last_pushup_time < self.min_rep_interval:
            return
            
        avg_angle = (left_elbow_angle + right_elbow_angle) / 2
        smoothed_angle = self.smooth_angle(avg_angle, self.pushup_angle_buffer)
        
        if smoothed_angle <= 0:
            return
            
        # Calibrate thresholds
        self.calibrate_thresholds('pushup', smoothed_angle)
        
        up_threshold = self.thresholds['pushup']['up_angle']
        down_threshold = self.thresholds['pushup']['down_angle']
        
        # Enhanced state machine
        if smoothed_angle > up_threshold and self.pushup_stage != 'up':
            self.pushup_stage = 'up'
            
        elif smoothed_angle < down_threshold and self.pushup_stage == 'up':
            self.pushup_stage = 'down'
            
        elif smoothed_angle > up_threshold and self.pushup_stage == 'down':
            self.pushup_counter += 1
            self.pushup_stage = 'up'
            self.last_pushup_time = current_time
            
            # Quality assessment
            form_quality = self.assess_form_quality(left_elbow_angle, right_elbow_angle)
            self.form_feedback.append(f"Push-up #{self.pushup_counter}: {form_quality}")
            
            # Store workout data
            self.workout_data.append({
                'timestamp': current_time,
                'exercise': 'pushup',
                'count': self.pushup_counter,
                'left_angle': left_elbow_angle,
                'right_angle': right_elbow_angle,
                'form_quality': form_quality
            })
            
            print(f"Push-up #{self.pushup_counter} completed! {form_quality}")
    
    def update_squat_count(self, left_knee_angle, right_knee_angle):
        """Enhanced squat counting with depth analysis"""
        current_time = time.time()
        
        if current_time - self.last_squat_time < self.min_rep_interval:
            return
            
        avg_angle = (left_knee_angle + right_knee_angle) / 2
        smoothed_angle = self.smooth_angle(avg_angle, self.squat_angle_buffer)
        
        if smoothed_angle <= 0:
            return
            
        # Calibrate thresholds
        self.calibrate_thresholds('squat', smoothed_angle)
        
        up_threshold = self.thresholds['squat']['up_angle']
        down_threshold = self.thresholds['squat']['down_angle']
        
        # State machine with depth analysis
        if smoothed_angle > up_threshold and self.squat_stage != 'up':
            self.squat_stage = 'up'
            
        elif smoothed_angle < down_threshold and self.squat_stage == 'up':
            self.squat_stage = 'down'
            
        elif smoothed_angle > up_threshold and self.squat_stage == 'down':
            self.squat_counter += 1
            self.squat_stage = 'up'
            self.last_squat_time = current_time
            
            # Depth analysis
            depth_quality = "Full depth" if min(left_knee_angle, right_knee_angle) < 100 else "Partial depth"
            form_quality = self.assess_form_quality(left_knee_angle, right_knee_angle)
            
            feedback = f"{depth_quality}, {form_quality}"
            self.form_feedback.append(f"Squat #{self.squat_counter}: {feedback}")
            
            # Store workout data
            self.workout_data.append({
                'timestamp': current_time,
                'exercise': 'squat',
                'count': self.squat_counter,
                'left_angle': left_knee_angle,
                'right_angle': right_knee_angle,
                'depth_quality': depth_quality,
                'form_quality': form_quality
            })
            
            print(f"Squat #{self.squat_counter} completed! {feedback}")
    
    def save_workout_data(self, filename=None):
        """Save workout data to JSON file"""
        if not self.workout_data:
            return
            
        if filename is None:
            timestamp = int(time.time())
            filename = f"workout_data_{timestamp}.json"
            
        try:
            # Calculate summary statistics
            summary = {
                'workout_type': self.workout_type,
                'total_pushups': self.pushup_counter,
                'total_squats': self.squat_counter,
                'total_exercises': len(self.workout_data),
                'duration': self.workout_data[-1]['timestamp'] - self.workout_data[0]['timestamp'] if self.workout_data else 0,
                'form_feedback': self.form_feedback[-10:]  # Last 10 feedback items
            }
            
            data = {
                'summary': summary,
                'detailed_data': self.workout_data
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Workout data saved to {filename}")
            
        except Exception as e:
            print(f"Error saving workout data: {e}")


class AdvancedRenderer:
    """Enhanced rendering with better visualizations"""
    
    def __init__(self):
        self.colors = {
            'primary': (0, 255, 255),      # Cyan
            'secondary': (255, 255, 0),    # Yellow
            'success': (0, 255, 0),        # Green
            'warning': (0, 165, 255),      # Orange
            'error': (0, 0, 255),          # Red
            'text': (255, 255, 255),       # White
            'background': (0, 0, 0),       # Black
            'skeleton': (255, 255, 255)    # White
        }
    
    def draw_enhanced_skeleton(self, image, landmarks, mp_pose, mp_drawing, width, height):
        """Draw enhanced skeleton with better visibility"""
        if not landmarks:
            return
            
        # Custom drawing specs
        landmark_spec = mp_drawing.DrawingSpec(
            color=self.colors['success'], thickness=6, circle_radius=6)
        connection_spec = mp_drawing.DrawingSpec(
            color=self.colors['skeleton'], thickness=4)
        
        # Draw pose landmarks and connections
        mp_drawing.draw_landmarks(
            image, landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_spec, connection_spec)
    
    def draw_angle_arc(self, image, center, angle, radius=40):
        """Draw angle arc visualization"""
        if center == [0, 0] or angle <= 0:
            return
            
        # Convert angle to arc
        start_angle = 0
        end_angle = int(angle)
        
        color = self.colors['success'] if 90 <= angle <= 170 else self.colors['warning']
        
        # Draw arc
        cv2.ellipse(image, tuple(center), (radius, radius), 
                   0, start_angle, end_angle, color, 3)
        
        # Draw angle text
        cv2.putText(image, f'{int(angle)}°', 
                   (center[0] - 20, center[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def create_advanced_info_panel(self, image, angles, tracker, fps, frame_count=0):
        """Create comprehensive information panel"""
        height, width = image.shape[:2]
        panel_width = 400
        panel_height = height
        
        # Create semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (width - panel_width, 0), 
                     (width, panel_height), self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
        
        y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Header
        cv2.putText(image, 'ADVANCED FITNESS TRACKER', 
                   (width - panel_width + 10, y), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.8, self.colors['primary'], 2)
        y += 50
        
        # Workout info
        cv2.putText(image, f'Mode: {tracker.workout_type}', 
                   (width - panel_width + 10, y), 
                   font, 0.6, self.colors['secondary'], 2)
        y += 30
        
        cv2.putText(image, f'Frame: {frame_count}', 
                   (width - panel_width + 10, y), 
                   font, 0.5, self.colors['text'], 1)
        y += 30
        
        # Joint angles section
        cv2.putText(image, 'JOINT ANGLES:', 
                   (width - panel_width + 10, y), 
                   font, 0.6, self.colors['secondary'], 2)
        y += 25
        
        angle_data = [
            ('Left Elbow', angles['left_elbow']),
            ('Right Elbow', angles['right_elbow']),
            ('Left Knee', angles['left_knee']),
            ('Right Knee', angles['right_knee'])
        ]
        
        for name, angle in angle_data:
            color = self.colors['success'] if 30 <= angle <= 180 else self.colors['warning']
            cv2.putText(image, f'{name}: {int(angle)}°', 
                       (width - panel_width + 20, y), 
                       font, 0.5, color, 2)
            y += 25
        
        y += 20
        
        # Exercise counters
        cv2.putText(image, 'EXERCISE COUNT:', 
                   (width - panel_width + 10, y), 
                   font, 0.6, self.colors['secondary'], 2)
        y += 30
        
        if tracker.workout_type in ['PushUp', 'General']:
            cv2.putText(image, f'Push-ups: {tracker.pushup_counter}', 
                       (width - panel_width + 20, y), 
                       font, 0.8, self.colors['success'], 2)
            y += 30
            cv2.putText(image, f'Stage: {tracker.pushup_stage}', 
                       (width - panel_width + 30, y), 
                       font, 0.5, self.colors['text'], 1)
            y += 25
        
        if tracker.workout_type in ['Squat', 'General']:
            cv2.putText(image, f'Squats: {tracker.squat_counter}', 
                       (width - panel_width + 20, y), 
                       font, 0.8, self.colors['success'], 2)
            y += 30
            cv2.putText(image, f'Stage: {tracker.squat_stage}', 
                       (width - panel_width + 30, y), 
                       font, 0.5, self.colors['text'], 1)
            y += 25
        
        # Form feedback
        if tracker.form_feedback:
            y += 20
            cv2.putText(image, 'FORM FEEDBACK:', 
                       (width - panel_width + 10, y), 
                       font, 0.6, self.colors['secondary'], 2)
            y += 25
            
            # Show last feedback
            last_feedback = tracker.form_feedback[-1] if tracker.form_feedback else "No feedback"
            lines = last_feedback.split(': ')
            for i, line in enumerate(lines):
                if i == 0:
                    cv2.putText(image, line + ':', 
                               (width - panel_width + 20, y), 
                               font, 0.4, self.colors['text'], 1)
                    y += 20
                else:
                    cv2.putText(image, line, 
                               (width - panel_width + 20, y), 
                               font, 0.4, self.colors['success'], 1)
                    y += 20
        
        # Performance metrics
        y = height - 100
        cv2.putText(image, f'FPS: {int(fps)}', 
                   (width - panel_width + 20, y), 
                   font, 0.6, self.colors['text'], 2)
        y += 25
        
        cv2.putText(image, 'Controls: Q=quit, R=reset, SPACE=pause', 
                   (width - panel_width + 10, y), 
                   font, 0.4, self.colors['text'], 1)


def main():
    args = parse_args()
    
    print(f"🏋️ Advanced Fitness Tracker v3.0")
    print(f"Workout Type: {args.workout_type}")
    print(f"Detection Confidence: {args.det}")
    print(f"Tracking Confidence: {args.track}")
    print(f"Model Complexity: {args.complexity}")
    print(f"Calibration: {'Enabled' if args.calibrate else 'Disabled'}")
    print("-" * 50)
    
    # Initialize components
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    angle_calc = AngleCalculator()
    landmark_extractor = LandmarkExtractor(mp_pose)
    tracker = WorkoutTracker(args.workout_type, args.calibrate)
    renderer = AdvancedRenderer()
    
    # Setup video capture
    if args.cam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open {'camera' if args.cam else args.video}")
        return
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if not args.cam else 30
    
    # Output video writer
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print("🚀 Starting advanced pose detection...")
    print("Controls: Q=quit, R=reset, SPACE=pause, S=save data")
    
    paused = False
    frame_count = 0
    
    # Initialize pose estimation
    with mp_pose.Pose(
        min_detection_confidence=args.det,
        min_tracking_confidence=args.track,
        model_complexity=args.complexity,
        smooth_landmarks=True,
        enable_segmentation=False) as pose:
        
        while cap.isOpened():
            start_time = time.time()
            
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video or camera error")
                    break
                
                frame_count += 1
                
                if args.debug and frame_count % 60 == 0:
                    print(f"🔄 Processing frame {frame_count}")
            
            height, width, _ = frame.shape
            
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
                'left_elbow': 0,
                'right_elbow': 0,
                'left_knee': 0,
                'right_knee': 0
            }
            
            # Process results
            if not paused and results.pose_landmarks:
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
                
                # Extract coordinates
                coords = landmark_extractor.extract_all_coords(landmarks, width, height)
                
                # Calculate angles using vector method
                angles['left_elbow'] = angle_calc.calculate_angle_vector(
                    coords['left_shoulder'], coords['left_elbow'], coords['left_wrist'])
                angles['right_elbow'] = angle_calc.calculate_angle_vector(
                    coords['right_shoulder'], coords['right_elbow'], coords['right_wrist'])
                angles['left_knee'] = angle_calc.calculate_angle_vector(
                    coords['left_hip'], coords['left_knee'], coords['left_ankle'])
                angles['right_knee'] = angle_calc.calculate_angle_vector(
                    coords['right_hip'], coords['right_knee'], coords['right_ankle'])
                
                # Update exercise counters
                if args.workout_type in ['PushUp', 'General']:
                    tracker.update_pushup_count(angles['left_elbow'], angles['right_elbow'])
                
                if args.workout_type in ['Squat', 'General']:
                    tracker.update_squat_count(angles['left_knee'], angles['right_knee'])
                
                # Draw enhanced skeleton
                renderer.draw_enhanced_skeleton(frame, results.pose_landmarks, mp_pose, mp_drawing, width, height)
                
                # Draw angle arcs for visual feedback
                if coords['left_elbow'] != [0, 0]:
                    renderer.draw_angle_arc(frame, coords['left_elbow'], angles['left_elbow'])
                if coords['right_elbow'] != [0, 0]:
                    renderer.draw_angle_arc(frame, coords['right_elbow'], angles['right_elbow'])
                if coords['left_knee'] != [0, 0]:
                    renderer.draw_angle_arc(frame, coords['left_knee'], angles['left_knee'])
                if coords['right_knee'] != [0, 0]:
                    renderer.draw_angle_arc(frame, coords['right_knee'], angles['right_knee'])
                
            else:
                # No pose detected
                cv2.putText(frame, 'No pose detected - Please ensure good lighting', (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Calculate FPS
            processing_time = time.time() - start_time
            current_fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Create advanced info panel
            renderer.create_advanced_info_panel(frame, angles, tracker, current_fps, frame_count)
            
            # Add pause indicator
            if paused:
                cv2.putText(frame, 'PAUSED - Press SPACE to resume', (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
            # Write output frame
            if out is not None and not paused:
                out.write(frame)
            
            # Display frame with adaptive scaling
            display_scale = 0.8
            if width > 1920:
                display_scale = 0.6
            elif width < 640:
                display_scale = 1.2
                
            if display_scale != 1.0:
                display_width = int(width * display_scale)
                display_height = int(height * display_scale)
                display_frame = cv2.resize(frame, (display_width, display_height))
            else:
                display_frame = frame
                
            cv2.imshow('Advanced Fitness Tracker', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):  # R for reset
                tracker.reset_counters()
                print("Counters and data reset!")
            elif key == ord(' '):  # SPACE for pause
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):  # S for save data
                if args.save_data:
                    tracker.save_workout_data()
                else:
                    print("Data saving not enabled. Use --save_data flag.")
            elif key == ord('c'):  # C for calibration toggle
                args.calibrate = not args.calibrate
                tracker.enable_calibration = args.calibrate
                print(f"Calibration {'enabled' if args.calibrate else 'disabled'}")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Save final workout data
    if args.save_data and tracker.workout_data:
        tracker.save_workout_data()
    
    # Final results and statistics
    print(f"\nWorkout Session Complete!")
    print("-" * 40)
    print(f"Final Results:")
    print(f"  Push-ups: {tracker.pushup_counter}")
    print(f"  Squats: {tracker.squat_counter}")
    print(f"  Total exercises: {tracker.pushup_counter + tracker.squat_counter}")
    print(f"  Frames processed: {frame_count}")
    
    if tracker.workout_data:
        duration = tracker.workout_data[-1]['timestamp'] - tracker.workout_data[0]['timestamp']
        print(f"  Session duration: {duration:.1f} seconds")
        print(f"  Average pace: {len(tracker.workout_data)/duration:.1f} exercises/minute")
    
    # Show recent form feedback
    if tracker.form_feedback:
        print(f"\nRecent Form Feedback:")
        for feedback in tracker.form_feedback[-5:]:
            print(f"  {feedback}")


if __name__ == "__main__":
    main()