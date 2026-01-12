import cv2
import numpy as np
cap = cv2.VideoCapture(0)
windows_positioned = False
seeds = []               # <--- nueva lista de semillas (x,y)
region_labels = None

def seeded_region_growing(gray, seeds, thresh=10):
    h, w = gray.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 1
    for sx, sy in seeds:
        if sx < 0 or sy < 0 or sx >= w or sy >= h:
            continue
        if labels[sy, sx] != 0:
            continue
        q = [(sy, sx)]
        labels[sy, sx] = current_label
        sum_int = int(gray[sy, sx])
        count = 1
        mean_int = float(sum_int)
        while q:
            r, c = q.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if labels[nr, nc] != 0:
                        continue
                    if abs(int(gray[nr, nc]) - mean_int) <= thresh:
                        labels[nr, nc] = current_label
                        q.append((nr, nc))
                        sum_int += int(gray[nr, nc])
                        count += 1
                        mean_int = sum_int / count
        current_label += 1
    return labels

def labels_to_color(labels):
    if labels is None:
        return None
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    unique = np.unique(labels)
    rng = np.random.RandomState(0)
    colors = {0: (0, 0, 0)}
    for lbl in unique:
        if lbl == 0:
            continue
        colors[lbl] = tuple(int(x) for x in rng.randint(50, 255, size=3))
    for lbl, col in colors.items():
        out[labels == lbl] = col
    return out

def on_mouse(event, x, y, flags, param):
    global seeds
    if event == cv2.EVENT_LBUTTONDOWN:
        seeds.append((x, y))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Algoritmo Canny: Detector de bordes multi-etapa
    # 1. Suavizado: Reduce ruido con filtro Gaussiano
    # 2. Gradiente: Calcula intensidad y dirección de bordes (Sobel)
    # 3. Supresión no máxima: Adelgaza bordes a 1 píxel de ancho
    # 4. Histéresis: Umbralización con dos thresholds
    #
    # Parámetros de umbralización:
    # - threshold1 (100): Umbral inferior. Bordes por debajo se descartan
    # - threshold2 (200): Umbral superior. Bordes por encima se aceptan
    # - Bordes entre umbral1 y umbral2 se aceptan si conectan con bordes fuertes
    #  - apertureSize (3): Tamaño del kernel Sobel (3, 5, 7). Afecta precisión del gradiente
    #  - L2gradient (False): Si True, usa fórmula L2 para gradiente (más preciso, más lento)
    #
    # Recomendaciones de configuración:
    # - Ratio threshold2:threshold1 = 2:1 a 3:1 (ej: 150-50, 200-100, 450-150)
    # - Valores bajos (50-100): Detecta más bordes (sensible, más ruido)
    # - Valores altos (200+): Detecta menos bordes (selectivo, más limpio)
    # - Ajusta según calidad de imagen y nivel de ruido
    edges = cv2.Canny(gray, 
                      threshold1=50, 
                      threshold2=100,
                      apertureSize=3, 
                      L2gradient=False)
    
    # Posicionar ventanas una sola vez al iniciar
    if not windows_positioned:
        print('Forma de la imagen:', frame.shape)
        h, w = frame.shape[:2]
        cv2.namedWindow('Original')
        cv2.moveWindow('Original', 0, 0)
        cv2.setMouseCallback('Original', on_mouse)
        cv2.namedWindow('Grayscale')
        cv2.moveWindow('Grayscale', w + 10, 0)
        cv2.namedWindow('Edges')
        cv2.moveWindow('Edges', 0, h + 10)
        cv2.namedWindow('Regions', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Regions', w + 10, h + 10)
        windows_positioned = True
    
    # dibuja semillas en la imagen original
    vis = frame.copy()
    for (sx, sy) in seeds:
        cv2.circle(vis, (sx, sy), 4, (0, 0, 255), -1)
    
    cv2.imshow('Original', vis)
    cv2.imshow('Grayscale', gray)
    cv2.imshow('Edges', edges)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        seeds = []
        region_labels = None
    elif key == ord('g'):
        # ejecutar crecimiento con umbral (ajustable)
        region_labels = seeded_region_growing(gray, seeds, thresh=12)
        color = labels_to_color(region_labels)
        if color is not None:
            cv2.imshow('Regions', color)
cap.release()
cv2.destroyAllWindows()
