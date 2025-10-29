장고 프로젝트에 추가할 3개 파일:
  1. animal_model.pt (3.7MB) - 훈련된 모델
  2. django_integration.py (8.6KB) - 장고 호환 래퍼
  3. requirements.txt - 패키지 의존성

  사용 방법:
  # 동영상 표시 되는 파일에서
  from django_integration import start_animal_detection, stop_animal_detection, get_detection_status

  # 감지 시작
  start_animal_detection("http://<IP>:81/stream")
