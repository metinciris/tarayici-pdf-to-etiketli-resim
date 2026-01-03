import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import re
import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
import shutil
import tempfile
from typing import List, Tuple, Optional

# Optional "done" sound (Windows)
try:
    import winsound
except Exception:
    winsound = None

# =========================
# KULLANICI AYARLARI
# =========================

# Etiket numarası varsayılan aralığı (GUI kapalıyken kullanılır)
LABEL_MIN_DEFAULT = 100
LABEL_MAX_DEFAULT = 45000

# GUI'de min–max sor (default kapalı)
ASK_LABEL_RANGE_GUI = False
ASK_LABEL_RANGE_CONFIRM_FIRST = True  # True: "değiştirmek ister misin?" sorar

# 1) Microsoft Picture Manager "Orta ton -100" benzeri
APPLY_MIDTONE_MINUS_100 = True
MIDTONE_GAMMA = 0.88  # 0.85–0.92 önerilir

# 2) Hafif renk/kontrast dokunuşu
APPLY_COLOR_TWEAK = True
BRIGHTNESS = 0
CONTRAST = 6
SATURATION = 1.00

# 3) IrfanView "Auto Adjust Colors" benzeri hafif kanal-stretch
APPLY_IRFAN_AUTO_ADJUST = True
IRFAN_LOW_PCT = 1.0
IRFAN_HIGH_PCT = 99.0

JPEG_QUALITY = 95

# Multi-PDF modu: alt klasörleri de tara
RECURSIVE_SCAN_SUBFOLDERS = True

# =========================


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_number_from_image(img_bgr):
    """Yellow label -> OCR red digits. Returns int or None."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Yellow label range (tweak if needed)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    label_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(label_contour)
    box = cv2.boxPoints(rect)
    box = np.int_(box)
    box = order_points(box)

    (tl, tr, br, bl) = box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < 20 or maxHeight < 20:
        return None

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(box.astype("float32"), dst)
    label_img = cv2.warpPerspective(img_bgr, M, (maxWidth, maxHeight))

    hsv_label = cv2.cvtColor(label_img, cv2.COLOR_BGR2HSV)

    # Red ranges
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_label, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_label, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    red_numbers = cv2.bitwise_and(label_img, label_img, mask=mask_red)
    gray = cv2.cvtColor(red_numbers, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    custom_config = r"--oem 3 --psm 6 outputbase digits"
    text = pytesseract.image_to_string(thresh, config=custom_config)
    text = "".join(filter(str.isdigit, text.strip()))
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def find_label_and_orientation(img_bgr, min_number, max_number):
    """
    Tries normal orientation; if fail/out of range, tries 180 rotation.
    Returns (label_number, chosen_img_bgr, rotated_used) or (None, img_bgr, False)
    """
    number_normal = extract_number_from_image(img_bgr)
    need_rotate = (number_normal is None) or (number_normal < min_number or number_normal > max_number)

    if (not need_rotate) and (min_number <= number_normal <= max_number):
        return number_normal, img_bgr, False

    img_rot = cv2.rotate(img_bgr, cv2.ROTATE_180)
    number_rot = extract_number_from_image(img_rot)
    if number_rot is not None and (min_number <= number_rot <= max_number):
        return number_rot, img_rot, True

    return None, img_bgr, False


def safe_folder_name_from_pdf(pdf_path):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    base = re.sub(r'[<>:"/\\|?*]+', "_", base).strip()
    return base or "pdf_cikti"


def extract_best_image_per_page(pdf_path, tmp_dir):
    """
    Extract original embedded images from each page (no re-render),
    store temporary JPGs in tmp_dir: page_XXX.jpg
    Returns list of temp paths.
    """
    doc = fitz.open(pdf_path)
    out = []

    for i in range(len(doc)):
        page = doc[i]
        imgs = page.get_images(full=True)
        if not imgs:
            print(f"  [WARN] Sayfa {i+1}: gömülü resim yok.")
            continue

        # Pick largest image by pixel count (typical scanned PDFs)
        best = None
        best_pixels = -1
        for img in imgs:
            xref = img[0]
            info = doc.extract_image(xref)
            w = info.get("width", 0)
            h = info.get("height", 0)
            px = w * h
            if px > best_pixels:
                best_pixels = px
                best = info

        if best is None:
            print(f"  [WARN] Sayfa {i+1}: uygun resim bulunamadı.")
            continue

        ext = (best.get("ext") or "bin").lower()
        raw_path = os.path.join(tmp_dir, f"page_{i+1:03d}.{ext}")
        with open(raw_path, "wb") as f:
            f.write(best["image"])

        # Normalize to JPG for downstream
        if ext in ("jpg", "jpeg"):
            final = os.path.join(tmp_dir, f"page_{i+1:03d}.jpg")
            if os.path.abspath(raw_path) != os.path.abspath(final):
                os.replace(raw_path, final)
        else:
            img_bgr = cv2.imread(raw_path)
            final = os.path.join(tmp_dir, f"page_{i+1:03d}.jpg")
            if img_bgr is None:
                print(f"  [WARN] {raw_path} okunamadı, atlanıyor.")
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
                continue
            cv2.imwrite(final, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            os.remove(raw_path)

        out.append(final)
        print(f"  [OK] Sayfa {i+1} çıkarıldı -> {os.path.basename(final)}")

    doc.close()
    return out


def clear_folder_contents(folder):
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        except Exception:
            pass


def beep_done():
    if winsound is None:
        return
    try:
        winsound.MessageBeep(winsound.MB_OK)
    except Exception:
        try:
            winsound.Beep(880, 200)
            winsound.Beep(988, 200)
        except Exception:
            pass


# =========================
# Görüntü işlemleri
# =========================
def apply_gamma(img_bgr, gamma):
    if gamma <= 0:
        return img_bgr
    table = (np.linspace(0, 1, 256) ** (1.0 / gamma)) * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, table)


def apply_brightness_contrast(img_bgr, brightness=0, contrast=0):
    beta = float(brightness)
    c = float(contrast)
    if c != 0:
        alpha = (131 * (c + 127)) / (127 * (131 - c))
    else:
        alpha = 1.0
    return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)


def apply_saturation(img_bgr, saturation=1.0):
    if abs(saturation - 1.0) < 1e-6:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(saturation)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def irfan_auto_adjust_like(img_bgr, low_pct=1.0, high_pct=99.0):
    out = img_bgr.copy()
    for ch in range(3):
        c = out[..., ch]
        lo = np.percentile(c, low_pct)
        hi = np.percentile(c, high_pct)
        if hi <= lo + 1:
            continue
        c = (c.astype(np.float32) - lo) * (255.0 / (hi - lo))
        out[..., ch] = np.clip(c, 0, 255).astype(np.uint8)
    return out


def final_tone_and_color(img_bgr):
    out = img_bgr

    if APPLY_MIDTONE_MINUS_100:
        out = apply_gamma(out, MIDTONE_GAMMA)

    if APPLY_COLOR_TWEAK:
        out = apply_brightness_contrast(out, BRIGHTNESS, CONTRAST)
        out = apply_saturation(out, SATURATION)

    if APPLY_IRFAN_AUTO_ADJUST:
        out = irfan_auto_adjust_like(out, IRFAN_LOW_PCT, IRFAN_HIGH_PCT)

    return out


def get_label_range_gui(root, default_min, default_max):
    """Returns (min,max). If user cancels, returns (default_min, default_max)."""
    if ASK_LABEL_RANGE_CONFIRM_FIRST:
        ans = messagebox.askyesno(
            "Etiket aralığı",
            f"Etiket aralığını değiştirmek ister misin?\n\nVarsayılan:\nMin: {default_min}\nMax: {default_max}"
        )
        if not ans:
            return default_min, default_max

    min_val = simpledialog.askinteger(
        "Etiket aralığı (Min)",
        "Minimum etiket numarası:",
        initialvalue=default_min,
        minvalue=0,
        parent=root
    )
    if min_val is None:
        return default_min, default_max

    max_val = simpledialog.askinteger(
        "Etiket aralığı (Max)",
        "Maksimum etiket numarası:",
        initialvalue=default_max,
        minvalue=min_val,
        parent=root
    )
    if max_val is None:
        return default_min, default_max

    return int(min_val), int(max_val)


def find_pdfs_in_folder(root_folder: str, recursive: bool = True) -> List[str]:
    pdfs = []
    if recursive:
        for dirpath, _, filenames in os.walk(root_folder):
            for fn in filenames:
                if fn.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root_folder):
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root_folder, fn))
    pdfs.sort()
    return pdfs


def ensure_output_folder(pdf_path: str) -> Tuple[str, str]:
    """Returns (out_dir, folder_name). out_dir is sibling of pdf."""
    pdf_dir = os.path.dirname(pdf_path)
    folder_name = safe_folder_name_from_pdf(pdf_path)
    out_dir = os.path.join(pdf_dir, folder_name)
    return out_dir, folder_name


# ---------- GLOBAL POLICY ----------
# "ask" | "overwrite_all" | "skip_all"
def choose_global_policy_for_existing_outputs(root) -> Optional[str]:
    """
    Returns one of:
      - "ask"
      - "overwrite_all"
      - "skip_all"
    Returns None if cancelled.
    """
    res = messagebox.askyesnocancel(
        "Global politika",
        "Çıktı klasörü zaten varsa ne yapalım?\n\nYes: Hepsini ÜZERİNE YAZ\nNo: Hepsini ATLA\nCancel: Her PDF için TEK TEK SOR"
    )
    if res is None:
        # Cancel => ask each time
        return "ask"
    if res is True:
        return "overwrite_all"
    return "skip_all"


def resolve_existing_output_folder(root, out_dir: str, folder_name: str, policy: str) -> str:
    """
    policy: "ask" | "overwrite_all" | "skip_all"
    Returns one of: "overwrite", "skip", "abort"
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return "overwrite"  # proceed

    if policy == "overwrite_all":
        clear_folder_contents(out_dir)
        return "overwrite"

    if policy == "skip_all":
        return "skip"

    # policy == "ask"
    ans = messagebox.askyesnocancel(
        "Klasör var",
        f"'{folder_name}' klasörü zaten var.\n\nYes: Üzerine yaz\nNo: Atla\nCancel: Tüm işlemi durdur"
    )
    if ans is None:
        return "abort"
    if ans is False:
        return "skip"
    clear_folder_contents(out_dir)
    return "overwrite"


def process_single_pdf(root, pdf_path: str, label_min: int, label_max: int, policy: str) -> Tuple[int, int, bool, bool]:
    """
    Returns (written, unread, continue_all, processed)
      - continue_all: False if user aborted all
      - processed: False if skipped
    """
    out_dir, folder_name = ensure_output_folder(pdf_path)
    action = resolve_existing_output_folder(root, out_dir, folder_name, policy)

    if action == "abort":
        print(f"[ABORT] {pdf_path}")
        return 0, 0, False, False
    if action == "skip":
        print(f"[SKIP] {pdf_path}")
        return 0, 0, True, False

    print(f"\n[PDF] {pdf_path}")
    print(f"[OUT] {out_dir}")

    written = 0
    unread = 0

    with tempfile.TemporaryDirectory(prefix="pdf_extract_") as tmp_dir:
        extracted = extract_best_image_per_page(pdf_path, tmp_dir)
        if not extracted:
            print("[ERR] PDF içinden resim çıkarılamadı.")
            return 0, 0, True, True

        for p in extracted:
            img = cv2.imread(p)
            if img is None:
                print(f"  [WARN] {os.path.basename(p)} okunamadı.")
                continue

            label, chosen_img, rotated = find_label_and_orientation(img, label_min, label_max)
            out_img = final_tone_and_color(chosen_img)

            if label is None:
                unread += 1
                base_name = os.path.basename(p)
                out_path = os.path.join(out_dir, base_name)
                print(f"  [WARN] Etiket yok/okunamadı -> {base_name} (isim değişmedi)")
            else:
                base = os.path.join(out_dir, f"{label}")
                out_path = f"{base}.jpg"
                k = 2
                while os.path.exists(out_path):
                    out_path = f"{base}_{k}.jpg"
                    k += 1
                print(f"  [OK] Etiket {label} bulundu{' (180° döndürüldü)' if rotated else ''} -> {os.path.basename(out_path)}")

            cv2.imwrite(out_path, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            written += 1

    print(f"[PDF DONE] Yazılan: {written} | Etiket okunamayan: {unread}")
    return written, unread, True, True


def main():
    root = tk.Tk()
    root.withdraw()

    # Mode selection
    mode_is_folder = messagebox.askyesno(
        "Çalışma modu",
        "Klasör modu ile çalışılsın mı?\n\nEvet: Klasör seç (içindeki tüm PDF'ler)\nHayır: Tek PDF seç"
    )

    # Label range
    label_min, label_max = LABEL_MIN_DEFAULT, LABEL_MAX_DEFAULT
    if ASK_LABEL_RANGE_GUI:
        label_min, label_max = get_label_range_gui(root, LABEL_MIN_DEFAULT, LABEL_MAX_DEFAULT)

    print(f"[INFO] Etiket aralığı: {label_min} .. {label_max}")
    print(f"[INFO] Midtone(-100)={'Açık' if APPLY_MIDTONE_MINUS_100 else 'Kapalı'} | gamma={MIDTONE_GAMMA}")
    print(f"[INFO] Renk tweak={'Açık' if APPLY_COLOR_TWEAK else 'Kapalı'} | brightness={BRIGHTNESS} contrast={CONTRAST} saturation={SATURATION}")
    print(f"[INFO] Irfan AutoAdjust={'Açık' if APPLY_IRFAN_AUTO_ADJUST else 'Kapalı'} | low={IRFAN_LOW_PCT} high={IRFAN_HIGH_PCT}")

    total_written = 0
    total_unread = 0
    total_pdfs_found = 0
    total_pdfs_processed = 0
    total_pdfs_skipped = 0

    if not mode_is_folder:
        pdf_path = filedialog.askopenfilename(
            title="İşlenecek PDF dosyasını seçin",
            filetypes=[("PDF files", "*.pdf")]
        )
        if not pdf_path:
            print("[INFO] PDF seçilmedi.")
            return

        # Single PDF: policy forced to "ask"
        policy = "ask"
        w, u, cont, processed = process_single_pdf(root, pdf_path, label_min, label_max, policy)
        total_written += w
        total_unread += u
        total_pdfs_found = 1
        total_pdfs_processed = 1 if processed else 0
        total_pdfs_skipped = 0 if processed else 1

    else:
        folder = filedialog.askdirectory(title="PDF klasörünü seçin (alt klasörler dahil taranır)")
        if not folder:
            print("[INFO] Klasör seçilmedi.")
            return

        pdfs = find_pdfs_in_folder(folder, recursive=RECURSIVE_SCAN_SUBFOLDERS)
        total_pdfs_found = len(pdfs)

        print(f"[INFO] Klasör: {folder}")
        print(f"[INFO] Bulunan PDF sayısı: {len(pdfs)} (alt klasörler={'Evet' if RECURSIVE_SCAN_SUBFOLDERS else 'Hayır'})")
        if not pdfs:
            print("[INFO] PDF bulunamadı.")
            return

        # GLOBAL POLICY selection
        policy = choose_global_policy_for_existing_outputs(root)
        # policy is always one of "ask/overwrite_all/skip_all"
        print(f"[INFO] Global politika: {policy}")

        for idx, pdf_path in enumerate(pdfs, start=1):
            print(f"\n=== ({idx}/{len(pdfs)}) ===")
            w, u, cont, processed = process_single_pdf(root, pdf_path, label_min, label_max, policy)
            total_written += w
            total_unread += u

            if processed:
                total_pdfs_processed += 1
            else:
                total_pdfs_skipped += 1

            if not cont:
                print("[INFO] Kullanıcı tüm işlemi durdurdu.")
                break

    print("\n[ALL DONE]")
    print(f"  Bulunan PDF: {total_pdfs_found}")
    print(f"  İşlenen PDF: {total_pdfs_processed}")
    print(f"  Atlanan PDF: {total_pdfs_skipped}")
    print(f"  Toplam yazılan resim: {total_written}")
    print(f"  Toplam etiket okunamayan: {total_unread}")
    beep_done()


if __name__ == "__main__":
    main()
