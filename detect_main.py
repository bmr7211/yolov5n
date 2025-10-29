import torch
import cv2
import numpy as np
from datetime import datetime
import os
import time

class AnimalDetector:
    def __init__(self, model_path='yolov5s.pt', confidence=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 디바이스: {self.device}")
        
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.conf = confidence
        
        self.classes = [
            '고라니', '멧돼지', '다람쥐', '너구리', '반달가슴곰', 
            '멧토끼', '족제비', '왜가리', '개', '고양이'
        ]
        
        self.capture_dir = 'captured_animals'
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)
    
    def detect_animals(self, frame):
        results = self.model(frame)
        detections = []
        
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if int(cls) < len(self.classes):
                animal_name = self.classes[int(cls)]
                detections.append({
                    'name': animal_name,
                    'confidence': conf,
                    'box': box
                })
        
        return detections
    
    def save_capture(self, frame, animal_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{animal_name}_{timestamp}.jpg"
        filepath = os.path.join(self.capture_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"동물 감지: {animal_name} - 저장됨: {filepath}")
    
    def process_stream(self, source=0):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("스트리밍 시작. 'q'를 눌러 종료.")
        
        last_capture_time = {}
        capture_cooldown = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            detections = self.detect_animals(frame)
            
            for detection in detections:
                animal_name = detection['name']
                confidence = detection['confidence']
                box = detection['box']
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{animal_name}: {confidence:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                current_time = time.time()
                if (animal_name not in last_capture_time or 
                    current_time - last_capture_time[animal_name] > capture_cooldown):
                    self.save_capture(frame, animal_name)
                    last_capture_time[animal_name] = current_time
            
            cv2.imshow('동물 감지 시스템', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = AnimalDetector()
    detector.process_stream()
