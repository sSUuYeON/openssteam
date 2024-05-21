import torch
import cv2
import numpy as np
import pygame

# Pygame 초기화 및 알람 소리 설정
pygame.init()
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('C:/car/stopCHU.wav')  # 알람 소리 파일 경로 (경로 수정됨)

# YOLOv5 모델 로드 (사전 학습된 모델)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 웹캠 캡처 초기화
cap = cv2.VideoCapture(0)

# 프레임 카운터 초기화
frame_counter = 0
alarm_playing = False
alarm_duration = 1000  # 알람 소리 지속 시간 (밀리초)
last_alarm_time = pygame.time.get_ticks()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 스킵 설정 (예: 5프레임마다 한 번씩 객체 감지)
    frame_counter += 1
    if frame_counter % 5 != 0:
        continue

    # 객체 검출
    results = model(frame)

    # 결과 시각화
    for det in results.xyxy[0]:  # 각 검출된 객체에 대해
        xmin, ymin, xmax, ymax, conf, cls = det
        if int(cls) == 2:  # 2번 클래스: 자동차 (COCO 데이터셋 기준)
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, f'car {conf:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 알람 소리 재생 관리
            current_time = pygame.time.get_ticks()
            if not alarm_playing or (current_time - last_alarm_time > alarm_duration):
                alarm_sound.play()
                last_alarm_time = current_time
                alarm_playing = True

    cv2.imshow('YOLOv5', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
