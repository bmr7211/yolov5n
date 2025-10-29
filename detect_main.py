# import torch
# import cv2
# import numpy as np
# from datetime import datetime
# import os
# import time
#
# class AnimalDetector:
#     def __init__(self, model_path='animal_model.pt', confidence=0.2):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"사용 중인 디바이스: {self.device}")
#
#         self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
#         self.model.conf = confidence
#
#         self.classes = [
#             'Goat', 'Wild boar', 'Squirrel', 'Raccoon', 'Asiatic black bear',
#             'Hare', 'Weasel', 'Heron', 'Dog', 'Cat'
#         ]
#
#         self.korean_names = [
#             '고라니', '멧돼지', '다람쥐', '너구리', '반달가슴곰',
#             '멧토끼', '족제비', '왜가리', '개', '고양이'
#         ]
#
#         self.capture_dir = 'captured_animals'
#         if not os.path.exists(self.capture_dir):
#             os.makedirs(self.capture_dir)
#
#     def detect_animals(self, frame):
#         results = self.model(frame)
#         detections = []
#
#         for *box, conf, cls in results.xyxy[0].cpu().numpy():
#             # 정확히 10개 클래스만 체크 (0~9번)
#             if int(cls) < len(self.classes):
#                 animal_name = self.classes[int(cls)]
#                 korean_name = self.korean_names[int(cls)]
#
#                 # 최소 신뢰도 체크 추가
#                 if conf >= self.model.conf:
#                     detections.append({
#                         'name': animal_name,
#                         'korean_name': korean_name,
#                         'confidence': conf,
#                         'box': box
#                     })
#                     print(f"동물 감지: {korean_name} (신뢰도: {conf:.3f})")
#
#         return detections
#
#     def save_capture(self, frame, korean_name):
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{korean_name}_{timestamp}.jpg"
#         filepath = os.path.join(self.capture_dir, filename)
#         cv2.imwrite(filepath, frame)
#         print(f"동물 감지: {korean_name} - 저장됨: {filepath}")
#
#     def process_stream(self, source=0):
#         cap = cv2.VideoCapture(source)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#         print("스트리밍 시작. 'q'를 눌러 종료.")
#
#         last_capture_time = {}
#         capture_cooldown = 5
#
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("프레임을 읽을 수 없습니다.")
#                 break
#
#             detections = self.detect_animals(frame)
#
#             for detection in detections:
#                 animal_name = detection['name']
#                 korean_name = detection['korean_name']
#                 confidence = detection['confidence']
#                 box = detection['box']
#
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{korean_name}: {confidence:.2f}",
#                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#                 current_time = time.time()
#                 if (korean_name not in last_capture_time or
#                     current_time - last_capture_time[korean_name] > capture_cooldown):
#                     self.save_capture(frame, korean_name)
#                     last_capture_time[korean_name] = current_time
#
#             cv2.imshow('동물 감지 시스템', frame)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         cap.release()
#         cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     detector = AnimalDetector(confidence=0.1)  # 매우 낮음
#     print(f"현재 신뢰도: {detector.model.conf}")
#
#     detector.process_stream("http://192.168.35.187:81/stream") # :81/stream
#     detector.process_stream(0)  # 스트림이 안되면 이걸로

import torch
import cv2
import numpy as np
from datetime import datetime
import os
import time
import urllib.request

class AnimalDetector:
    def __init__(self, model_path='animal_model.pt', confidence=0.2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 디바이스: {self.device}")

        # torch.hub yolov5 커스텀 로드
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.conf = confidence  # 신뢰도 임계값

        self.classes = [
            'Goat', 'Wild boar', 'Squirrel', 'Raccoon', 'Asiatic black bear',
            'Hare', 'Weasel', 'Heron', 'Dog', 'Cat'
        ]
        self.korean_names = [
            '고라니', '멧돼지', '다람쥐', '너구리', '반달가슴곰',
            '멧토끼', '족제비', '왜가리', '개', '고양이'
        ]

        self.capture_dir = 'captured_animals'
        os.makedirs(self.capture_dir, exist_ok=True)

    # -------------------- 내부 유틸 --------------------
    def _read_capture(self, ip, timeout=2):
        """http://<ip>/capture 스틸샷 1장을 읽어 프레임으로 반환"""
        url = f"http://{ip}/capture"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            jpg = r.read()
        arr = np.frombuffer(jpg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def _open_stream_with_fallback(self, ip):
        """
        스트림 후보 URL들을 FFmpeg 백엔드로 순차 시도.
        성공 시 VideoCapture 반환, 실패 시 None.
        """
        urls = [
            f"http://{ip}:81/stream",
            f"http://{ip}/stream",
            f"http://{ip}:81/",
            f"http://{ip}:81/stream?dummy=1.mjpg",  # 캐시/프록시 회피용
        ]
        for u in urls:
            print("Trying:", u)
            cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)  # 🔑 FFmpeg 백엔드
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            t0 = time.time()
            while time.time() - t0 < 5:  # 5초 내에 한 프레임이라도 오면 성공
                ok, frame = cap.read()
                if ok and frame is not None:
                    print("Connected:", u)
                    return cap
                time.sleep(0.1)
            cap.release()
        return None

    # -------------------- 추론/저장 --------------------
    def detect_animals(self, frame):
        results = self.model(frame)
        detections = []
        # yolov5 hub 모델: results.xyxy[0] 사용
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if int(cls) < len(self.classes):  # 0~9번만
                if conf >= self.model.conf:
                    idx = int(cls)
                    detections.append({
                        'name': self.classes[idx],
                        'korean_name': self.korean_names[idx],
                        'confidence': float(conf),
                        'box': box
                    })
                    print(f"동물 감지: {self.korean_names[idx]} (신뢰도: {conf:.3f})")
        return detections

    def save_capture(self, frame, korean_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{korean_name}_{timestamp}.jpg"
        filepath = os.path.join(self.capture_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"동물 감지: {korean_name} - 저장됨: {filepath}")

    # -------------------- 메인 스트리밍 루프 --------------------
    def process_stream(self, source=0):
        """
        source:
          - 정수(0 등): 웹캠
          - 문자열: "http://<ip>:81/stream" 또는 "<ip>" 만 넘겨도 됨 (자동 처리)
        """
        # 소스가 IP/URL이면 IP만 추출
        ip = None
        cap = None
        use_capture = False  # /capture 폴링 모드 여부

        if isinstance(source, str):
            # "http://192.168.35.187:81/stream" 또는 "192.168.35.187" 지원
            s = source.replace("http://", "").replace("https://", "")
            ip = s.split("/")[0]              # "192.168.35.187:81"
            ip = ip.split(":")[0]             # "192.168.35.187"
            # 1) 스트림 먼저 시도
            cap = self._open_stream_with_fallback(ip)
            use_capture = cap is None
        else:
            # 로컬 웹캠
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("스트리밍 시작. 'q'를 눌러 종료. 모드:",
              "CAPTURE 폴링" if use_capture else ("STREAM" if isinstance(source, str) else "LOCAL CAM"))

        last_capture_time = {}
        capture_cooldown = 5  # 클래스별 저장 쿨다운(초)

        while True:
            # 프레임 획득
            if use_capture:
                try:
                    frame = self._read_capture(ip)
                except Exception as e:
                    print("capture 실패:", e)
                    time.sleep(0.2)
                    continue
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if isinstance(source, str):
                        # 스트림 끊기면 /capture 폴링으로 전환
                        print("스트림 끊김 → capture 폴링 전환")
                        use_capture = True
                        continue
                    else:
                        print("프레임을 읽을 수 없습니다.")
                        break

            # 추론
            detections = self.detect_animals(frame)

            # 시각화/저장
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection['box'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detection['korean_name']}: {detection['confidence']:.2f}",
                            (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                now = time.time()
                kn = detection['korean_name']
                if kn not in last_capture_time or now - last_capture_time[kn] > capture_cooldown:
                    self.save_capture(frame, kn)
                    last_capture_time[kn] = now

            cv2.imshow('동물 감지 시스템', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 발열/부하 완화
            time.sleep(0.05)

        if cap and not use_capture:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 낮은 신뢰도로 더 많이 감지 (환경에 맞게 조정)
    detector = AnimalDetector(confidence=0.3)
    print(f"현재 신뢰도: {detector.model.conf}")

    # ★ 스트림 테스트: IP만 넘겨도 자동으로 스트림 시도 → 실패 시 /capture 폴백
    # detector.process_stream("192.168.35.187")

    # ★ 로컬 웹캠(옵션)
    detector.process_stream(0)