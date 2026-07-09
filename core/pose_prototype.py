"""YOLO-pose prototip: kişi-bazlı kafa bölgesi çıkarımı.

Her insanı tespit eder, 17 COCO keypoint'ine indirger, kafa keypoint'lerinden
bir "kafa kutusu" oluşturur ve (ileride) bir baret kutusuyla örtüşmesini (IoU)
ölçerek bareti doğru kişiye atamak için zemin hazırlar.

Kullanım:
    python pose_prototype.py --source ornek.jpg
    python pose_prototype.py --source video.mp4
    python pose_prototype.py --source 0            # webcam
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

# --- COCO pose keypoint indeksleri -----------------------------------------
# 0=burun, 1=sol göz, 2=sağ göz, 3=sol kulak, 4=sağ kulak
HEAD_KPT_IDS = (0, 1, 2, 3, 4)
KPT_CONF_THRESH = 0.30          # bu değerin altındaki keypoint güvenilmez sayılır
HEAD_PADDING_RATIO = 0.20       # kafa keypoint kümesine eklenen kenar payı

# COCO iskeletindeki kemik bağlantıları (0-indeksli keypoint çiftleri)
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),           # kafa
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # kollar / omuz
    (5, 11), (6, 12), (11, 12),               # gövde
    (11, 13), (13, 15), (12, 14), (14, 16),   # bacaklar
)


@dataclass
class Person:
    """Tek bir tespit edilmiş kişi."""

    person_id: int
    person_box: tuple[int, int, int, int]           # x1, y1, x2, y2
    keypoints: np.ndarray                            # (17, 3) -> x, y, conf
    head_box: tuple[int, int, int, int] | None
    head_kpt_count: int                              # güvenle bulunan kafa kpt sayısı
    head_from_fallback: bool                         # kutu fallback ile mi bulundu


# --- Geometri fonksiyonları (saf, test edilebilir) --------------------------
def head_box_from_keypoints(
    keypoints: np.ndarray,
    person_box: tuple[int, int, int, int],
    conf_thresh: float = KPT_CONF_THRESH,
    padding_ratio: float = HEAD_PADDING_RATIO,
) -> tuple[tuple[int, int, int, int], int, bool]:
    """Kafa keypoint'lerinden kafa kutusu üret.

    Yeterli güvenilir kafa keypoint'i varsa onların sınırlayıcı kutusunu (pay
    ekleyerek) döndürür. Yoksa fallback olarak insan kutusunun üst %25'ini alır.

    Dönüş: (head_box, güvenilir_kafa_kpt_sayısı, fallback_kullanildi_mi)
    """
    head_pts = []
    for idx in HEAD_KPT_IDS:
        x, y, conf = keypoints[idx]
        if conf >= conf_thresh:
            head_pts.append((x, y))

    if len(head_pts) >= 2:
        pts = np.array(head_pts, dtype=np.float32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        # Keypoint'ler kafanın merkezine yakındır; kutuyu payla genişlet.
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        pad_x = w * padding_ratio + w * 0.5
        pad_y = h * padding_ratio + h * 0.7   # dikeyde biraz daha (alın/çene)
        box = (
            int(round(x1 - pad_x)),
            int(round(y1 - pad_y)),
            int(round(x2 + pad_x)),
            int(round(y2 + pad_y)),
        )
        return _clip_box(box, person_box), len(head_pts), False

    # Fallback: insan kutusunun üst %25'i.
    px1, py1, px2, py2 = person_box
    box = (px1, py1, px2, py1 + int((py2 - py1) * 0.25))
    return _clip_box(box, person_box), len(head_pts), True


def helmet_iou(
    head_box: tuple[int, int, int, int],
    helmet_box: tuple[int, int, int, int],
) -> float:
    """Kafa kutusu ile baret kutusu arasındaki IoU (0..1).

    Baret adımında, tespit edilen her baret kutusu her kişinin kafa kutusuyla
    karşılaştırılıp en yüksek IoU'ya sahip kişiye atanacak.

    TODO(baret): Baret modeli eklendiğinde:
        - Baret tespitlerini bu fonksiyonla kafalara eşle (Hungarian / greedy).
        - IoU eşiğini (ör. > 0.1) kalibre et; kafa kutusu payını buna göre ayarla.
        - Örtüşme yoksa kişi "baretsiz" etiketlensin.
    """
    return _iou(head_box, helmet_box)


def _iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """İki eksen-hizalı kutu arasındaki Intersection-over-Union."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _clip_box(
    box: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Kutuyu verilen sınırların (insan kutusu) içine kırp."""
    x1, y1, x2, y2 = box
    bx1, by1, bx2, by2 = bounds
    return (
        max(x1, bx1),
        max(y1, by1),
        min(x2, bx2),
        min(y2, by2),
    )


# --- Çıkarım + çizim --------------------------------------------------------
def build_people(result) -> list[Person]:
    """Bir Ultralytics sonucundan Person listesi üret."""
    people: list[Person] = []
    if result.keypoints is None or result.boxes is None:
        return people

    boxes = result.boxes.xyxy.cpu().numpy()
    kpts = result.keypoints.data.cpu().numpy()   # (N, 17, 3)

    for i, (box, kp) in enumerate(zip(boxes, kpts)):
        person_box = tuple(int(v) for v in box[:4])
        head_box, head_count, fallback = head_box_from_keypoints(kp, person_box)
        people.append(
            Person(
                person_id=i,
                person_box=person_box,
                keypoints=kp,
                head_box=head_box,
                head_kpt_count=head_count,
                head_from_fallback=fallback,
            )
        )
    return people


def draw_person(frame: np.ndarray, person: Person) -> None:
    """İskelet, kafa keypoint'leri, kafa kutusu ve insan kutusunu çiz."""
    kp = person.keypoints

    # İnsan kutusu (yeşil)
    px1, py1, px2, py2 = person.person_box
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 0), 2)
    cv2.putText(
        frame, f"id {person.person_id}", (px1, max(py1 - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA,
    )

    # İskelet kemikleri (mavi)
    for a, b in SKELETON:
        xa, ya, ca = kp[a]
        xb, yb, cb = kp[b]
        if ca >= KPT_CONF_THRESH and cb >= KPT_CONF_THRESH:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                     (255, 160, 0), 2, cv2.LINE_AA)

    # Kafa keypoint'leri (kırmızı nokta)
    for idx in HEAD_KPT_IDS:
        x, y, conf = kp[idx]
        if conf >= KPT_CONF_THRESH:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)

    # Kafa kutusu (sarı; fallback ise kesikli görünüm için ince)
    if person.head_box is not None:
        hx1, hy1, hx2, hy2 = person.head_box
        color = (0, 220, 220) if not person.head_from_fallback else (0, 140, 220)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)


def log_person(person: Person) -> None:
    """Konsola kişi özetini bas."""
    tag = "fallback" if person.head_from_fallback else "keypoint"
    print(
        f"  kisi {person.person_id:>2} | kafa_kutusu={person.head_box} "
        f"| guvenli_kafa_kpt={person.head_kpt_count} ({tag})"
    )


def process_frame(model: YOLO, frame: np.ndarray, conf: float) -> np.ndarray:
    """Tek bir kareyi işle: çıkarım → çizim → log. Çizili kareyi döndür."""
    result = model.predict(frame, conf=conf, verbose=False)[0]
    people = build_people(result)
    print(f"[frame] tespit edilen kisi: {len(people)}")
    for person in people:
        draw_person(frame, person)
        log_person(person)
    return frame


# --- Girdi yönetimi ---------------------------------------------------------
def is_image(source: str) -> bool:
    return source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    )


def run(source: str, weights: str, conf: float, save: str | None, show: bool) -> None:
    """Kaynağı (resim/video/webcam) işle."""
    print(f"[model] yukleniyor: {weights}")
    model = YOLO(weights)          # ağırlık yoksa otomatik iner

    if is_image(source):
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Goruntu okunamadi: {source}")
        out = process_frame(model, frame, conf)
        if save:
            cv2.imwrite(save, out)
            print(f"[kaydedildi] {save}")
        if show:
            cv2.imshow("pose_prototype", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Video ya da webcam
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Kaynak acilamadi: {source}")

    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = process_frame(model, frame, conf)
        if save:
            if writer is None:
                h, w = out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                writer = cv2.VideoWriter(save, fourcc, fps, (w, h))
            writer.write(out)
        if show:
            cv2.imshow("pose_prototype", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[kaydedildi] {save}")
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO-pose kafa bolgesi prototipi")
    p.add_argument("--source", required=True,
                   help="resim/video yolu veya webcam icin 0")
    p.add_argument("--weights", default="model/yolo11n-pose.pt",
                   help="YOLO-pose agirligi (otomatik iner, model/ altina kaydedilir)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="tespit guven esigi")
    p.add_argument("--save", default=None,
                   help="cikti dosyasi (resim/video)")
    p.add_argument("--show", action="store_true",
                   help="pencerede goster")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.source, args.weights, args.conf, args.save, args.show)


if __name__ == "__main__":
    main()
