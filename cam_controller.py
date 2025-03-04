import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading

# Configurações
CAMERA_INDEX = 0  # Índice da câmera (geralmente 0 para a câmera padrão)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()  # Resolução da tela
CLICK_DISTANCE = 10  # Distância máxima para considerar dedos juntos
TARGET_FPS = 120  # Limitar a taxa de frames para reduzir a carga de processamento

# Variáveis globais para compartilhamento entre threads
frame_lock = threading.Lock()
current_frame = None
left_hand_positions = None
right_hand_positions = None

# Inicializar o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Função para capturar frames da câmera
def capture_frames():
    global current_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espelhar o frame para evitar confusão de direção
        frame = cv2.flip(frame, 1)

        with frame_lock:
            current_frame = frame

    cap.release()

# Função para processar a detecção de mãos
def process_hands():
    global left_hand_positions, right_hand_positions

    while True:
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        # Converter o frame para RGB (MediaPipe requer RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar o frame com o MediaPipe Hands
        results = hands.process(rgb_frame)

        left_hand = None
        right_hand = None

        # Verificar se mãos foram detectadas
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Obter a posição dos dedos
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                height, width, _ = frame.shape
                thumb_x, thumb_y = int(thumb.x * width), int(thumb.y * height)
                index_x, index_y = int(index_finger.x * width), int(index_finger.y * height)
                middle_x, middle_y = int(middle_finger.x * width), int(middle_finger.y * height)

                # Determinar se é a mão esquerda ou direita
                if results.multi_handedness[i].classification[0].label == "Left":
                    left_hand = (thumb_x, thumb_y), (index_x, index_y), (middle_x, middle_y)
                else:
                    right_hand = (thumb_x, thumb_y), (index_x, index_y), (middle_x, middle_y)

        # Atualizar as posições das mãos
        with frame_lock:
            left_hand_positions = left_hand
            right_hand_positions = right_hand

        # Limitar a taxa de frames
        time.sleep(1 / TARGET_FPS)

# Função principal para controle do mouse
def control_mouse():
    while True:
        with frame_lock:
            left_hand = left_hand_positions
            right_hand = right_hand_positions

        # Mover o mouse com a mão direita
        if right_hand:
            (thumb_x, thumb_y), (index_x, index_y), _ = right_hand

            # Mapear a posição do dedo indicador para a tela
            screen_x = np.interp(index_x, [0, 640], [0, SCREEN_WIDTH])
            screen_y = np.interp(index_y, [0, 480], [0, SCREEN_HEIGHT])

            # Mover o mouse para a posição mapeada
            pyautogui.moveTo(screen_x, screen_y)

        # Cliques com a mão esquerda
        if left_hand:
            (thumb_x, thumb_y), (index_x, index_y), (middle_x, middle_y) = left_hand

            # Calcular a distância entre o polegar e o indicador
            distance_index_thumb = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5

            # Calcular a distância entre o polegar e o dedo do meio
            distance_middle_thumb = ((middle_x - thumb_x) ** 2 + (middle_y - thumb_y) ** 2) ** 0.5

            # Clique esquerdo: juntar indicador e polegar
            if distance_index_thumb < CLICK_DISTANCE:
                pyautogui.click(button="left")

            # Clique direito: juntar dedo do meio e polegar
            if distance_middle_thumb < CLICK_DISTANCE:
                pyautogui.click(button="right")

        # Limitar a taxa de frames
        time.sleep(1 / TARGET_FPS)

# Função para exibir o frame (opcional, para debug)
def display_frame():
    while True:
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        # Desenhar os landmarks das mãos (opcional, para debug)
        if left_hand_positions:
            (thumb_x, thumb_y), (index_x, index_y), (middle_x, middle_y) = left_hand_positions
            cv2.circle(frame, (thumb_x, thumb_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (index_x, index_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (middle_x, middle_y), 5, (0, 255, 0), -1)

        if right_hand_positions:
            (thumb_x, thumb_y), (index_x, index_y), _ = right_hand_positions
            cv2.circle(frame, (thumb_x, thumb_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (index_x, index_y), 5, (0, 0, 255), -1)

        # Mostrar o frame
        cv2.imshow("Hand Tracking", frame)

        # Parar o loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Iniciar threads
    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=process_hands, daemon=True).start()
    threading.Thread(target=control_mouse, daemon=True).start()

    # Exibir o frame (opcional, para debug)
    display_frame()