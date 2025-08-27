import cv2
import mediapipe as mp
import numpy as np
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Testing for Pose Detection.')
    parser.add_argument('-v','--video', type=str, default='Squats.mp4')
    parser.add_argument('-o','--output', type=str, default=None)
    parser.add_argument('--det', type=float, default=0.5, help='Detection confidence')
    parser.add_argument('--track', type=float, default=0.5, help='Tracking confidence')
    parser.add_argument('-c','--complexity', type=int, default=0, help='Complexity of the model options 0,1,2')
    
    return parser.parse_args()


args = parse_args()

line_color = (255, 255, 255)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Drawing specifications
drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=4, color=line_color)
drawing_spec_points = mp_drawing.DrawingSpec(thickness=5, circle_radius=4, color=line_color)

# Configuration from arguments
detection_confidence = args.det
tracking_confidence = args.track
complexity = args.complexity

# Initialize video capture
vid = cv2.VideoCapture(args.video)

# Check if video opened successfully
if not vid.isOpened():
    print(f"Error: Could not open video file {args.video}")
    exit()

# Get video properties
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))

# Initialize video writer if output is specified
out = None
if args.output:
    codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(args.output, codec, fps, (width, height))

# Initialize MediaPipe Pose
with mp_pose.Pose(
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence,
    model_complexity=complexity,
    smooth_landmarks=True) as pose:

    while vid.isOpened():
        start_time = time.time()
        
        success, image = vid.read()

        if not success:
            print("End of video or failed to read frame")
            break

        # Convert BGR to RGB for MediaPipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
   
        image.flags.writeable = False
        results = pose.process(image)

        # Process pose landmarks
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract landmark coordinates
                left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
                right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]

                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                       landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                # Get hip and leg coordinates
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Get ear coordinates (even though we'll hide them)
                left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                # Hide face landmarks by setting visibility to 0
                face_landmarks = [
                    mp_pose.PoseLandmark.LEFT_EYE,
                    mp_pose.PoseLandmark.RIGHT_EYE,
                    mp_pose.PoseLandmark.LEFT_EYE_INNER,
                    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
                    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
                    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
                    mp_pose.PoseLandmark.NOSE,
                    mp_pose.PoseLandmark.MOUTH_LEFT,
                    mp_pose.PoseLandmark.MOUTH_RIGHT,
                    mp_pose.PoseLandmark.LEFT_EAR,
                    mp_pose.PoseLandmark.RIGHT_EAR
                ]
                
                for landmark in face_landmarks:
                    landmarks[landmark.value].visibility = 0

                # Calculate midpoints
                midpoint_shoulder_x = (int(shoulder[0] * image_width) + int(shoulder_r[0] * image_width)) / 2
                midpoint_shoulder_y = (int(shoulder[1] * image_height) + int(shoulder_r[1] * image_height)) / 2

                midpoint_hip_x = (int(left_hip[0] * image_width) + int(right_hip[0] * image_width)) / 2
                midpoint_hip_y = (int(left_hip[1] * image_height) + int(right_hip[1] * image_height)) / 2

                based_mid_x = int((midpoint_shoulder_x + midpoint_hip_x) / 2)
                based_mid_y = int((midpoint_shoulder_y + midpoint_hip_y) / 2)

                neck_point_x = int((int(nose[0] * image_width) + int(midpoint_shoulder_x)) / 2)
                neck_point_y = int((int(nose[1] * image_height) + int(midpoint_shoulder_y)) / 2)

                # Calculate angles
                left_arm_angle = int(calculate_angle(shoulder, elbow, wrist))
                right_arm_angle = int(calculate_angle(shoulder_r, elbow_r, wrist_r))
                left_leg_angle = int(calculate_angle(left_hip, left_knee, left_ankle))
                right_leg_angle = int(calculate_angle(right_hip, right_knee, right_ankle))

                # Calculate additional metrics
                left_arm_length = np.linalg.norm(np.array(shoulder) - np.array(elbow))

                # Hide shoulder landmarks (they will be drawn manually)
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility = 0
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility = 0

                # Make image writable again for drawing
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw custom skeleton lines
                cv2.line(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)),
                        (int(neck_point_x), int(neck_point_y)), line_color, 3)
                cv2.line(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)),
                        (int(neck_point_x), int(neck_point_y)), line_color, 3)

                cv2.line(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)),
                        (int(elbow[0] * image_width), int(elbow[1] * image_height)), line_color, 3)
                cv2.line(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)),
                        (int(elbow_r[0] * image_width), int(elbow_r[1] * image_height)), line_color, 3)

                # Neck to mid point
                cv2.line(image, (int(neck_point_x), int(neck_point_y)),
                        (int(based_mid_x), int(based_mid_y)), line_color, 3)

                # Mid to hips
                cv2.line(image, (int(based_mid_x), int(based_mid_y)),
                        (int(left_hip[0] * image_width), int(left_hip[1] * image_height)), line_color, 3)
                cv2.line(image, (int(based_mid_x), int(based_mid_y)),
                        (int(right_hip[0] * image_width), int(right_hip[1] * image_height)), line_color, 3)

                # Draw circles at key points
                cv2.circle(image, (int(neck_point_x), int(neck_point_y)), 4, line_color, 5)
                cv2.circle(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)), 4, line_color, 3)
                cv2.circle(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)), 4, line_color, 3)
                cv2.circle(image, (int(based_mid_x), int(based_mid_y)), 4, line_color, 5)

                # Create info panel
                cv2.rectangle(image, (image_width, 0), (image_width - 300, 250), (0, 0, 0), -1)
                cv2.putText(image, 'Angles', (image_width - 300, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(image, f'Left Elbow Angle: {left_arm_angle}', (image_width - 290, 70),
                           cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Right Elbow Angle: {right_arm_angle}', (image_width - 290, 110),
                           cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Left Knee Angle: {left_leg_angle}', (image_width - 290, 150),
                           cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Right Knee Angle: {right_leg_angle}', (image_width - 290, 190),
                           cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error processing landmarks: {e}")
            # Convert back to BGR even if there's an error
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw MediaPipe pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                drawing_spec_points,
                connection_drawing_spec=drawing_spec)

        # Calculate and display FPS
        current_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        cv2.putText(image, f'FPS: {int(current_fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame if output is specified
        if args.output and out is not None:
            out.write(image)

        # Resize for display
        display_frame = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Pose', display_frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Clean up
vid.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()