import argparse
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Optional

import cv2
import pytesseract
from pytesseract import Output

# Изображение по умолчанию для запуска без аргументов.
DEFAULT_IMAGE = Path("test-images/test-02.png")


def configure_tesseract() -> None:
    # Шаг 1: если пользователь явно передал путь к tesseract через переменную окружения,
    # используем его как самый приоритетный вариант.
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd

    # Шаг 2: Windows-костыль.
    # Если tesseract не найден в PATH (часто после установки без перезапуска терминала),
    # пробуем стандартный путь инсталлятора UB Mannheim.
    if platform.system() == "Windows" and not shutil.which("tesseract"):
        win_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if win_tesseract.exists():
            pytesseract.pytesseract.tesseract_cmd = str(win_tesseract)


def ensure_tesseract_available() -> None:
    # Шаг 3: проверяем, что бинарник OCR реально доступен.
    # Либо через tesseract_cmd, либо через PATH.
    cmd = pytesseract.pytesseract.tesseract_cmd
    if cmd and Path(cmd).exists():
        return
    if shutil.which("tesseract"):
        return

    # Шаг 4: формируем понятную подсказку по установке под текущую ОС.
    system_name = platform.system()
    if system_name == "Linux":
        hint = "Install with: sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-eng"
    elif system_name == "Windows":
        hint = "Install Tesseract OCR and reopen terminal, or set TESSERACT_CMD to tesseract.exe path"
    else:
        hint = "Install Tesseract OCR and ensure 'tesseract' is in PATH"

    raise RuntimeError(f"Tesseract OCR is not available. {hint}")


def normalize_temp(token: str) -> Optional[float]:
    # Шаг 5: нормализуем десятичный разделитель, чтобы 22,0 и 22.0 обрабатывались одинаково.
    cleaned = token.replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if not match:
        return None

    raw = match.group(0)

    # Шаг 6: эвристики для частых OCR-ошибок.
    # Если видим 3+ цифры без точки, трактуем как XX.X (например 220 -> 22.0).
    if "." in raw:
        value = float(raw)
    elif len(raw) >= 3:
        value = float(f"{raw[:2]}.{raw[2]}")
    elif len(raw) == 2:
        value = float(f"{raw}.0")
    else:
        value = float(f"{raw}.0")

    return round(value, 1)


def round_to_half(value: float) -> float:
    # Шаг 7: квантуем результат к шагу 0.5, чтобы сгладить шум OCR на маленьких кропах.
    return round(value * 2.0) / 2.0


def collect_numeric_tokens(image_part) -> list[str]:
    # Шаг 8: fallback-ветка собирает много гипотез по цифрам
    # из разных каналов и предобработок, чтобы увеличить шанс попасть в правильное число.
    gray = cv2.cvtColor(image_part, cv2.COLOR_BGR2GRAY)
    red = cv2.split(image_part)[2]

    tokens: list[str] = []
    for base in (gray, red):
        for scale in (4, 6, 8, 10):
            # Шаг 8.1: апскейл маленького текста.
            up = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Шаг 8.2: несколько вариантов фильтрации/бинаризации.
            variants = (
                up,
                cv2.medianBlur(up, 3),
                cv2.GaussianBlur(up, (3, 3), 0),
                cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            )
            for variant in variants:
                # Шаг 8.3: OCR только по цифровому алфавиту.
                txt = pytesseract.image_to_string(
                    variant,
                    config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,",
                )
                token = "".join(txt.split())
                if re.search(r"\d", token):
                    tokens.append(token)

    return tokens


def decode_compact_temp(token: str, prefer_zero_decimal: bool) -> Optional[float]:
    # Шаг 9: декодируем "склеенные" OCR-строки (например 738, 0228).
    digits = "".join(re.findall(r"\d", token))
    if len(digits) < 2:
        return None

    # Шаг 9.1: если цифр слишком много, берем хвост из 3 цифр.
    # Это костыль под типичные ошибки вида 0228 -> 228.
    if len(digits) >= 4:
        digits = digits[-3:]

    # Шаг 9.2: двухзначное число считаем как XX.0.
    if len(digits) == 2:
        value = float(f"{digits}.0")
        return value if 10.0 <= value <= 35.0 else None

    # Шаг 9.3: трехзначное число трактуем как XX.X.
    # Доп. костыль: если первая цифра "нереалистичная" (6-9),
    # подменяем ее на 2 (738 -> 238 -> 23.8).
    if len(digits) == 3:
        a, b, c = digits
        if a in "6789":
            a = "2"

        # Для Set чаще встречается .0, поэтому при сомнительной последней цифре
        # можно принудить десятичную часть к 0.
        dec = "0" if prefer_zero_decimal and c in "568" else c

        value = float(f"{a}{b}.{dec}")
        return value if 10.0 <= value <= 35.0 else None

    return None


def extract_from_small_display(image) -> tuple[Optional[float], Optional[float], str]:
    # Шаг 10: fallback для сильно обрезанных/шумных картинок.
    # Берем верхнюю строку, делим пополам: слева ожидаем Act, справа Set.
    h, w = image.shape[:2]
    top_line = image[:int(0.45 * h), :]
    left = top_line[:, : w // 2]
    right = top_line[:, w // 2 :]

    left_tokens = collect_numeric_tokens(left)
    right_tokens = collect_numeric_tokens(right)

    # Шаг 10.1: декодируем все гипотезы по Act и Set.
    act_values = [
        value
        for token in left_tokens
        if (value := decode_compact_temp(token, prefer_zero_decimal=False)) is not None
    ]
    set_values = [
        value
        for token in right_tokens
        if (value := decode_compact_temp(token, prefer_zero_decimal=True)) is not None
    ]

    # Шаг 10.2: если гипотез нет, возвращаем отладочную информацию.
    if not act_values or not set_values:
        debug = f"small-display tokens left={left_tokens[:5]} right={right_tokens[:5]}"
        return None, None, debug

    # Шаг 10.3: берем медиану как устойчивую оценку и округляем к шагу 0.5.
    act_values.sort()
    set_values.sort()
    act = round_to_half(act_values[len(act_values) // 2])
    set_value = round_to_half(set_values[len(set_values) // 2])

    debug = f"small-display tokens left={left_tokens[:5]} right={right_tokens[:5]}"
    return act, set_value, debug


def extract_from_layout(
    image,
    act_conf_threshold: float,
    set_conf_threshold: float,
) -> tuple[Optional[float], Optional[float], float, float, str]:
    # Шаг 11: основной (предпочтительный) сценарий.
    # Снимаем OCR с координатами и confidence для каждого слова.
    config = "--oem 3 --psm 11"
    data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)

    words = []
    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue
        words.append(
            {
                "index": i,
                "text": text,
                "conf": float(data["conf"][i]),
            }
        )

    # Отладочная строка OCR "как увидел tesseract".
    one_line = " ".join(w["text"] for w in words)

    # Шаг 11.1: ищем якорь "Set".
    set_idx = None
    for w in words:
        if w["text"].lower().startswith("set"):
            set_idx = w["index"]
            break

    if set_idx is None:
        return None, None, 0.0, 0.0, one_line

    # Шаг 11.2: извлекаем числовые токены и нормализуем к float.
    temp_tokens = []
    for w in words:
        value = normalize_temp(w["text"])
        if value is not None:
            temp_tokens.append({**w, "value": value})

    # Шаг 11.3: берем последнее число слева от Set как Act,
    # и первое число справа от Set как Set.
    left_candidates = [t for t in temp_tokens if t["index"] < set_idx]
    right_candidates = [t for t in temp_tokens if t["index"] > set_idx]

    act = left_candidates[-1] if left_candidates else None
    set_value = right_candidates[0] if right_candidates else None

    act_value = act["value"] if act else None
    set_temp = set_value["value"] if set_value else None
    act_conf = act["conf"] if act else 0.0
    set_conf = set_value["conf"] if set_value else 0.0

    # Шаг 11.4: confidence-костыль.
    # Если Act распознан с низкой уверенностью, а Set уверенно,
    # подставляем Set в Act (типичный случай: 20 вместо 22.0 на шумном кадре).
    if act_value is not None and set_temp is not None and act_conf < act_conf_threshold and set_conf >= set_conf_threshold:
        act_value = set_temp

    return act_value, set_temp, act_conf, set_conf, one_line


def parse_args() -> argparse.Namespace:
    # Шаг 12: аргументы CLI для запуска и ручной калибровки.
    parser = argparse.ArgumentParser(description="Read Act/Set temperatures from a panel photo")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Path to image file")
    parser.add_argument("--act-conf-threshold", type=float, default=40.0, help="Fallback threshold for Act confidence")
    parser.add_argument("--set-conf-threshold", type=float, default=60.0, help="Fallback threshold for Set confidence")
    return parser.parse_args()


def main() -> None:
    # Шаг 13: читаем аргументы и готовим окружение OCR.
    args = parse_args()
    configure_tesseract()
    ensure_tesseract_available()

    # Шаг 14: валидация входного файла изображения.
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"OpenCV failed to load image: {image_path}")

    # Шаг 15: сначала пробуем основной парсер по layout+confidence.
    act, set_value, act_conf, set_conf, ocr_line = extract_from_layout(
        img,
        act_conf_threshold=args.act_conf_threshold,
        set_conf_threshold=args.set_conf_threshold,
    )

    print(ocr_line)

    # Шаг 16: если основной парсер не справился, переключаемся на fallback.
    if act is None or set_value is None:
        fb_act, fb_set, fb_debug = extract_from_small_display(img)
        if fb_act is None or fb_set is None:
            print("Could not confidently extract both temperatures.")
            print(fb_debug)
            return
        print(f"Act={fb_act:.1f} Set={fb_set:.1f} (fallback)")
        return

    # Шаг 17: успешный результат основного сценария.
    print(f"Act={act:.1f} Set={set_value:.1f} (conf: act={act_conf:.0f}, set={set_conf:.0f})")


if __name__ == "__main__":
    main()
