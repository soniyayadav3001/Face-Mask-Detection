import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import deque
import pickle
import uuid  # for generating unique IDs per face

# ========= Load model + class labels =========
MODEL_PATH = "mask_detector_model.h5"
LABELS_PATH = "class_labels.pkl"
INPUT_SIZE = (128, 128)

model = load_model(MODEL_PATH)

with open(LABELS_PATH, "rb") as f:
    label_to_id = pickle.load(f)  # {'with_mask': 0, 'without_mask': 1}
id_to_label = {v: k for k, v in label_to_id.items()}


def pretty(name: str) -> str:
    n = name.replace("_", " ").strip().lower()
    if "without" in n or (("no" in n) and ("mask" in n)) or "unmasked" in n:
        return "Without Mask"
    if "with" in n and "mask" in n or n == "mask" or "masked" in n:
        return "With Mask"
    return name.replace("_", " ").title()


POS_LABEL = pretty(id_to_label.get(1, "class_1"))
NEG_LABEL = pretty(id_to_label.get(0, "class_0"))

# ========= MediaPipe face detection =========
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# ========= Helpers =========
def expand_and_square(x, y, w, h, img_w, img_h, expand=1.35):
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * expand
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = int(round(cx + side / 2))
    y1 = int(round(cy + side / 2))
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(img_w, x1); y1 = min(img_h, y1)
    if x1 <= x0: x1 = min(img_w, x0 + 1)
    if y1 <= y0: y1 = min(img_h, y0 + 1)
    return x0, y0, x1, y1


def clahe_color(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def preprocess_face(face_bgr):
    face_bgr = clahe_color(face_bgr)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    arr = image.img_to_array(face_resized) / 255.0
    return np.expand_dims(arr, axis=0)


# ========= Per-face smoothing =========
HI_THR = 0.60
LO_THR = 0.40
FACE_MEMORY = {}  # {face_id: {"deque": deque, "label": str, "center": (x,y)}}


def get_face_id(center, max_dist=80):
    """Assign ID to face based on proximity to existing ones."""
    for fid, data in FACE_MEMORY.items():
        cx, cy = data["center"]
        if np.linalg.norm(np.array(center) - np.array((cx, cy))) < max_dist:
            return fid
    return str(uuid.uuid4())  # new ID


def smooth_label(fid, prob):
    """Per-face smoothing with hysteresis."""
    data = FACE_MEMORY[fid]
    data["deque"].append(float(prob))
    mean_p = float(np.mean(data["deque"]))

    if data["label"] is None:
        data["label"] = POS_LABEL if mean_p >= 0.5 else NEG_LABEL
    elif mean_p >= HI_THR and data["label"] != POS_LABEL:
        data["label"] = POS_LABEL
    elif mean_p <= LO_THR and data["label"] != NEG_LABEL:
        data["label"] = NEG_LABEL

    return data["label"]


# ========= Realtime loop =========
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open camera.")

print("Press 'q' to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            x0, y0, x1, y1 = expand_and_square(x, y, bw, bh, w, h, expand=1.35)
            face_roi = frame[y0:y1, x0:x1]
            if face_roi.size == 0:
                continue

            # Face center → assign ID
            center = ((x0 + x1) // 2, (y0 + y1) // 2)
            fid = get_face_id(center)

            # Initialize memory for new face
            if fid not in FACE_MEMORY:
                FACE_MEMORY[fid] = {"deque": deque(maxlen=9), "label": None, "center": center}

            FACE_MEMORY[fid]["center"] = center

            # Predict
            face_input = preprocess_face(face_roi)
            prob = model.predict(face_input, verbose=0)[0][0]
            label = smooth_label(fid, prob)

            # Draw
            color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, label, (x0, max(25, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Multi-face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
