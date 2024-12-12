import cv2
import mediapipe as mp
import numpy as np
import time

class GestureVideoControl:
    def __init__(self, camera_index=1, video_path="D:\psa\Welcome to India ! [CINEMATIC TRAVEL FILM].mp4"):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.video = cv2.VideoCapture(video_path)
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0), 
            thickness=5,       
            circle_radius=5    
        )
        
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 0, 0),  
            thickness=2       
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.playing = False
        self.frame_count = 0
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        return frame, results

    def detect_gesture(self, frame, results):
        current_time = time.time()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.connection_drawing_spec
                )

                if current_time - self.last_gesture_time > self.gesture_cooldown:
                    index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_x = int(index_finger_tip.x * frame.shape[1])
                    index_y = int(index_finger_tip.y * frame.shape[0])

                    if index_y < frame.shape[0] // 3: 
                        if not self.playing:
                            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
                            self.playing = True
                            self.last_gesture_time = current_time

                    elif index_y > frame.shape[0] // 3 * 2: 
                        if self.playing:
                            self.frame_count = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                            self.playing = False
                            self.last_gesture_time = current_time

        return frame

    def run(self):
        try:
            while self.cap.isOpened():
                frame_data = self.process_frame()
                if frame_data is None:
                    break
                
                frame, results = frame_data

                frame = self.detect_gesture(frame, results)

                if self.playing:
                    ret, video_frame = self.video.read()
                    if not ret:
                        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                        continue
                    
                    video_frame = cv2.resize(video_frame, (frame.shape[1], frame.shape[0]))
                    cv2.imshow('Video Playback', video_frame)
                else:
                    cv2.imshow('Webcam', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            self.video.release()
            cv2.destroyAllWindows()

def list_available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
            cap.release()
        index += 1
    print("Available camera indices:", arr)
    return arr

def main():
    available_cameras = list_available_cameras()
    
    if available_cameras:
        video_controller = GestureVideoControl(camera_index=available_cameras[0])
        video_controller.run()
    else:
        print("No cameras found!")

if __name__ == "__main__":
    main()