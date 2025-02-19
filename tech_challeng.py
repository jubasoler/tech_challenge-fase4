import cv2
import os
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
import mediapipe as mp

def detect_faces_and_emotions_and_pose(video_path, output_path, log_path):

    # Inicializar o MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_drawing = mp.solutions.drawing_utils

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dominant_emotions = []
    anomalies = 0
    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    file = open(log_path, "w")

    # Loop para processar cada frame do vídeo com barra de progresso
    for i in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()        

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Analisar o frame para detectar faces e expressões
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Desenhar as anotações da pose no frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Iterar sobre cada face detectada pelo DeepFace        
        for face in result:

            number_of_faces_in_frame = len(result)

            file.write("Existem {} face(s) ou anomalias no frame #{}.".format(number_of_faces_in_frame, i)) 
            file.write("\n") 

            # Obter a caixa delimitadora da face
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            
            face_confidence = face['face_confidence']
            if (float(face_confidence) < 0.85 and float(face_confidence) > 0.5):
                anomalies = anomalies + 1

                # Desenhar um retângulo ao redor da anomalia
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 1)
                cv2.putText(frame, "Anomalia", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                file.write("Frame #{} - Uma anomalia está localizada em -> Topo: {}, Esquerda: {}, Fundo: {}, Direita: {}.".format(i, x, y, w, h)) 
                file.write("\n") 
            else:
                # Desenhar um retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Obter a emoção dominante
                dominant_emotion = face['dominant_emotion']
                dominant_emotions.append(dominant_emotion)

                # Escrever a emoção dominante acima da face
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                file.write("Frame #{} - Uma face está localizada em -> Topo: {}, Esquerda: {}, Fundo: {}, Direita: {}. Emoção Dominante: {}".format(i, x, y, w, h, dominant_emotion)) 
                file.write("\n") 

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    file.write("TOTAL DE FRAMES: {}".format(total_frames))
    file.write("\n") 
    file.write("FPS: {}".format(fps))
    file.write("\n") 
    file.write("ANOMALIAS: {}".format(anomalies))
    file.write("\n") 
    file.write("EXPRESSOES EMOCIONAIS")
    file.write("\n") 
    for emotion in dominant_emotions:
        file.write(emotion)
        file.write("\n") 
    
    # Liberar a captura de vídeo e fechar todas as janelas
    file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')
output_video_path = os.path.join(script_dir, 'teste_output_final2.mp4')
log_path = os.path.join(script_dir, 'log2.txt')

# Chamar a função para detectar emoções e reconhecer faces no vídeo, salvando o vídeo processado
detect_faces_and_emotions_and_pose(input_video_path, output_video_path, log_path)