import cv2
import numpy as np
import argparse
import time
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Pose Detection using OpenCV')
    parser.add_argument('-v','--video', type=str, default='Squats.mp4')
    parser.add_argument('-o','--output', type=str, default=None)
    parser.add_argument('-c','--confidence', type=float, default=0.1)
    
    return parser.parse_args()

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# COCO body parts
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

def load_pose_model():
    """Load OpenPose model"""
    # Download these files if not present:
    # wget -O pose_iter_440000.caffemodel http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
    # wget -O pose_deploy_linevec.prototxt https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt
    
    proto_file = "pose_deploy_linevec.prototxt"
    weights_file = "pose_iter_440000.caffemodel"
    
    try:
        net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
        return net
    except:
        print(f"Could not load model files. Using simplified detection.")
        return None

def detect_pose_simple(frame):
    """Simplified pose detection using background subtraction and contour detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive threshold to detect edges
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the person)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:  # Minimum area threshold
            return largest_contour
    
    return None

def main():
    args = parse_args()
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output is specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Try to load OpenPose model
    net = load_pose_model()
    
    # Variables for exercise counting
    push_up_counter = 0
    up_position = False
    down_position = False
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            print("End of video")
            break
        
        frame_height, frame_width = frame.shape[:2]
        
        if net is not None:
            # OpenPose detection
            blob = cv2.dnn.blobFromImage(frame, 1.0/255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
            net.setInput(blob)
            output = net.forward()
            
            points = []
            for i in range(len(BODY_PARTS)):
                heatMap = output[0, i, :, :]
                _, conf, _, point = cv2.minMaxLoc(heatMap)
                
                x = int((frame_width * point[0]) / output.shape[3])
                y = int((frame_height * point[1]) / output.shape[2])
                
                if conf > args.confidence:
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                    cv2.putText(frame, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    points.append(None)
            
            # Draw skeleton
            for pair in POSE_PAIRS:
                partA = BODY_PARTS[pair[0]]
                partB = BODY_PARTS[pair[1]]
                
                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)
            
            # Calculate angles if key points are detected
            try:
                if points[BODY_PARTS["LShoulder"]] and points[BODY_PARTS["LElbow"]] and points[BODY_PARTS["LWrist"]]:
                    left_elbow_angle = calculate_angle(
                        points[BODY_PARTS["LShoulder"]], 
                        points[BODY_PARTS["LElbow"]], 
                        points[BODY_PARTS["LWrist"]]
                    )
                    cv2.putText(frame, f'L Elbow: {int(left_elbow_angle)}°', (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Push-up counting logic
                    if left_elbow_angle > 160 and not up_position:
                        up_position = True
                        down_position = False
                    elif left_elbow_angle < 90 and up_position and not down_position:
                        down_position = True
                    elif left_elbow_angle > 160 and down_position:
                        push_up_counter += 1
                        up_position = False
                        down_position = False
                
                if points[BODY_PARTS["RShoulder"]] and points[BODY_PARTS["RElbow"]] and points[BODY_PARTS["RWrist"]]:
                    right_elbow_angle = calculate_angle(
                        points[BODY_PARTS["RShoulder"]], 
                        points[BODY_PARTS["RElbow"]], 
                        points[BODY_PARTS["RWrist"]]
                    )
                    cv2.putText(frame, f'R Elbow: {int(right_elbow_angle)}°', (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if points[BODY_PARTS["LHip"]] and points[BODY_PARTS["LKnee"]] and points[BODY_PARTS["LAnkle"]]:
                    left_knee_angle = calculate_angle(
                        points[BODY_PARTS["LHip"]], 
                        points[BODY_PARTS["LKnee"]], 
                        points[BODY_PARTS["LAnkle"]]
                    )
                    cv2.putText(frame, f'L Knee: {int(left_knee_angle)}°', (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if points[BODY_PARTS["RHip"]] and points[BODY_PARTS["RKnee"]] and points[BODY_PARTS["RAnkle"]]:
                    right_knee_angle = calculate_angle(
                        points[BODY_PARTS["RHip"]], 
                        points[BODY_PARTS["RKnee"]], 
                        points[BODY_PARTS["RAnkle"]]
                    )
                    cv2.putText(frame, f'R Knee: {int(right_knee_angle)}°', (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except:
                pass
        
        else:
            # Simplified detection without model
            contour = detect_pose_simple(frame)
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.putText(frame, 'Person Detected (Simplified)', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display counter and FPS
        cv2.putText(frame, f'Push-ups: {push_up_counter}', (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        current_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        cv2.putText(frame, f'FPS: {int(current_fps)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame if output is specified
        if out:
            out.write(frame)
        
        # Display frame
        cv2.imshow('Pose Detection', cv2.resize(frame, (800, 600)))
        
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()