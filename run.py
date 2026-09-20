"""Reproduce the E5 hard-case cascade submission in one readable script."""
from pathlib import Path
import argparse
import json
import re
import time

import cv2
import numpy as np
import pandas as pd
from paddleocr import TextLineOrientationClassification, TextRecognition


PACKAGE_DIR = Path(__file__).resolve().parent
CJK_CHARACTERS = re.compile(r"[\u3400-\u9fff\u3040-\u30ff]")
EPSILON = 1e-6


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, float), EPSILON, 1 - EPSILON)
    return np.log(probability / (1 - probability))


def result_data(result) -> dict:
    raw_result = result.json
    if isinstance(raw_result, str):
        raw_result = json.loads(raw_result)
    return raw_result["res"]


def p180(result) -> float:
    prediction = result_data(result)
    predicted_score = float(prediction["scores"][0])
    if prediction["label_names"][0] == "180_degree":
        return predicted_score
    return 1 - predicted_score


def load_images(image_paths: list[Path]) -> list[np.ndarray]:
    images = [cv2.imread(str(path)) for path in image_paths]
    if any(image is None for image in images):
        unreadable = [str(path) for path, image in zip(image_paths, images) if image is None]
        raise RuntimeError(f"Cannot read images: {unreadable[:3]}")
    return images


def orientation_logits(
    image_paths: list[Path],
    model_name: str,
    device: str,
    batch_size: int,
    model_dir: Path | None = None,
) -> np.ndarray:
    model_options = {"model_name": model_name, "device": device}
    if model_dir is not None:
        model_options["model_dir"] = str(model_dir)
    orientation_model = TextLineOrientationClassification(**model_options)
    symmetry_logits = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        original_images = load_images(batch_paths)
        rotated_images = [cv2.rotate(image, cv2.ROTATE_180) for image in original_images]
        original_results = list(orientation_model.predict_iter(original_images, batch_size=batch_size))
        rotated_results = list(orientation_model.predict_iter(rotated_images, batch_size=batch_size))
        original_probability = np.array([p180(result) for result in original_results])
        rotated_probability = np.array([p180(result) for result in rotated_results])

        # R180 меняет местами классы. Усреднение в логистическом пространстве позволяет получить прогноз.
        # Ассиметрикал
        batch_logits = (logit(original_probability) - logit(rotated_probability)) / 2
        symmetry_logits.extend(batch_logits)

    return np.asarray(symmetry_logits)


def ocr_margin(
    image_paths: list[Path],
    model_name: str,
    device: str,
    batch_size: int,
    return_text: bool = False,
) -> tuple[np.ndarray, list[str], list[str]]:
    recognition_model = TextRecognition(model_name=model_name, device=device)
    orientation_margins = []
    original_texts = []
    rotated_texts = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        original_images = load_images(batch_paths)
        rotated_images = [cv2.rotate(image, cv2.ROTATE_180) for image in original_images]

        original_results = list(recognition_model.predict_iter(original_images, batch_size=batch_size))
        rotated_results = list(recognition_model.predict_iter(rotated_images, batch_size=batch_size))

        for original_result, rotated_result in zip(original_results, rotated_results):
            original = result_data(original_result)
            rotated = result_data(rotated_result)
            # Наличие отступов означает, что данная система OCR легче распознает изображение, повернутое на 180 градусов (R180),
            # и следовательно поддерживает класс поворота на 180 градусов.
            orientation_margins.append(float(rotated["rec_score"]) - float(original["rec_score"]))
            if return_text:
                original_texts.append(str(original["rec_text"]))
                rotated_texts.append(str(rotated["rec_text"]))

    return np.asarray(orientation_margins), original_texts, rotated_texts


def sparse_ocr_margin(
    all_image_paths: list[Path],
    routed_mask: np.ndarray,
    model_name: str,
    device: str,
    original_batch_size: int,
) -> np.ndarray:
    """Run OCR on routed crops while preserving full-batch resize geometry."""
    recognition_model = TextRecognition(model_name=model_name, device=device)
    orientation_margins = []

    for start in range(0, len(all_image_paths), original_batch_size):
        block_paths = all_image_paths[start : start + original_batch_size]
        block_mask = routed_mask[start : start + original_batch_size]
        if not block_mask.any():
            continue

        # PaddleOCR выбирает ширину распознавания на основе самого широкого фрагмента.
        # Пустое изображение с такими размерами позволяет сохранить результаты
        block_images = load_images(block_paths)
        routed_images = [image for image, is_routed in zip(block_images, block_mask) if is_routed]
        widest_image = max(block_images, key=lambda image: image.shape[1] / image.shape[0])
        padding_reference = np.zeros_like(widest_image)
        sparse_batch = routed_images + [padding_reference]

        original_results = list(
            recognition_model.predict_iter(sparse_batch, batch_size=original_batch_size)
        )[:-1]
        rotated_batch = [cv2.rotate(image, cv2.ROTATE_180) for image in sparse_batch]
        rotated_results = list(
            recognition_model.predict_iter(rotated_batch, batch_size=original_batch_size)
        )[:-1]
        for original_result, rotated_result in zip(original_results, rotated_results):
            original = result_data(original_result)
            rotated = result_data(rotated_result)
            orientation_margins.append(float(rotated["rec_score"]) - float(original["rec_score"]))

    return np.asarray(orientation_margins)


def weighted_sum(signals: dict[str, np.ndarray], settings: dict) -> np.ndarray:
    result = np.zeros(len(next(iter(signals.values()))), dtype=float)
    for signal_name, signal_values in signals.items():
        signal_settings = settings[signal_name]
        result += signal_values / signal_settings["scale"] * signal_settings["weight"]
    return result


def predict_e2(
    image_paths: list[Path], config: dict, device: str, batch_size: int
) -> np.ndarray:
    print("Base E2: pretrained orientation model")
    pretrained_x1_logit = orientation_logits(
        image_paths, "PP-LCNet_x1_0_textline_ori", device, batch_size
    )
    print("Base E2: expanded-clean orientation model")
    expanded_clean_x1_logit = orientation_logits(
        image_paths,
        "PP-LCNet_x1_0_textline_ori",
        device,
        batch_size,
        PACKAGE_DIR / "e1_model",
    )

    print("Base E2: generic, East-Slavic and Cyrillic OCR")
    generic_margin, generic_original_text, generic_rotated_text = ocr_margin(
        image_paths, "PP-OCRv5_mobile_rec", device, batch_size, return_text=True
    )
    east_slavic_margin, _, _ = ocr_margin(
        image_paths, "eslav_PP-OCRv5_mobile_rec", device, batch_size
    )
    cyrillic_margin, _, _ = ocr_margin(
        image_paths, "cyrillic_PP-OCRv5_mobile_rec", device, batch_size
    )
    contains_cjk = np.array([
        bool(CJK_CHARACTERS.search(original + " " + rotated))
        for original, rotated in zip(generic_original_text, generic_rotated_text)
    ], dtype=float)

    d5_logit = weighted_sum(
        {
            "pretrained_x1_logit": pretrained_x1_logit,
            "generic_ocr_margin": generic_margin,
            "east_slavic_ocr_margin": east_slavic_margin,
            "cyrillic_ocr_margin": cyrillic_margin,
            "cjk_x_east_slavic_margin": contains_cjk * east_slavic_margin,
            "cjk_x_cyrillic_margin": contains_cjk * cyrillic_margin,
        },
        config["d5_fusion"],
    )
    e2_settings = config["e2_fusion"]
    return (
        e2_settings["d5_logit_weight"] * d5_logit
        + e2_settings["expanded_clean_x1_logit_weight"] * expanded_clean_x1_logit
    )


def run_cascade(
    e2_logit: np.ndarray,
    image_paths: list[Path],
    config: dict,
    device: str,
    batch_size: int,
    inference_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    e2_probability = sigmoid(e2_logit)
    orientation_confidence = np.maximum(e2_probability, 1 - e2_probability)
    needs_extra_ocr = orientation_confidence < config["confidence_threshold"]
    routed_count = int(needs_extra_ocr.sum())
    print(f"Hard-case cascade: routing {routed_count} of {len(image_paths)} images")

    if inference_mode == "sparse":
        # Фаст деплой
        english_margin = sparse_ocr_margin(
            image_paths, needs_extra_ocr, "en_PP-OCRv5_mobile_rec", device, batch_size
        )
        latin_margin = sparse_ocr_margin(
            image_paths, needs_extra_ocr, "latin_PP-OCRv5_mobile_rec", device, batch_size
        )
        v6_small_margin = sparse_ocr_margin(
            image_paths, needs_extra_ocr, "PP-OCRv6_small_rec", device, batch_size
        )
    else:
        # Режим точного воспроизведения сохраняет полные пакеты данных
        english_all, _, _ = ocr_margin(
            image_paths, "en_PP-OCRv5_mobile_rec", device, batch_size
        )
        latin_all, _, _ = ocr_margin(
            image_paths, "latin_PP-OCRv5_mobile_rec", device, batch_size
        )
        v6_small_all, _, _ = ocr_margin(
            image_paths, "PP-OCRv6_small_rec", device, batch_size
        )
        english_margin = english_all[needs_extra_ocr]
        latin_margin = latin_all[needs_extra_ocr]
        v6_small_margin = v6_small_all[needs_extra_ocr]

    hard_case_logit = weighted_sum(
        {
            "e2_logit": e2_logit[needs_extra_ocr],
            "english_ocr_margin": english_margin,
            "latin_ocr_margin": latin_margin,
            "ocr_v6_small_margin": v6_small_margin,
        },
        config["hard_case_fusion"],
    )
    final_probability = e2_probability.copy()
    final_probability[needs_extra_ocr] = sigmoid(hard_case_logit)
    return final_probability, needs_extra_ocr


def find_images(images_dir: Path, official_ids: list[str]) -> list[Path]:
    image_by_id = {path.stem: path for path in images_dir.iterdir() if path.is_file()}
    missing_ids = [image_id for image_id in official_ids if image_id not in image_by_id]
    if missing_ids:
        raise ValueError(f"Images missing for {len(missing_ids)} IDs, for example: {missing_ids[:3]}")
    return [image_by_id[image_id] for image_id in official_ids]


def check_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["image_id", "p_180"]:
        raise ValueError("Submission columns must be exactly: image_id, p_180")
    if len(sample_submission) == 20000 and len(submission) != 20000:
        raise ValueError(f"Expected 20000 rows, got {len(submission)}")
    if submission.image_id.duplicated().any():
        raise ValueError("Submission contains duplicate image_id values")
    if not submission.image_id.equals(sample_submission.image_id):
        missing = set(sample_submission.image_id) - set(submission.image_id)
        extra = set(submission.image_id) - set(sample_submission.image_id)
        raise ValueError(f"IDs/order differ from sample_submission: missing={len(missing)}, extra={len(extra)}")
    if submission.p_180.isna().any():
        raise ValueError("Submission contains NaN probabilities")
    if not submission.p_180.between(0, 1).all():
        raise ValueError("Every p_180 value must be between 0 and 1")


def compare_csv(submission: pd.DataFrame, reference_path: Path) -> None:
    reference = pd.read_csv(reference_path)
    comparison = reference.merge(submission, on="image_id", suffixes=("_reference", "_new"), validate="one_to_one")
    difference = np.abs(comparison.p_180_reference - comparison.p_180_new)
    reference_label = comparison.p_180_reference >= 0.5
    new_label = comparison.p_180_new >= 0.5
    print(f"Reference max absolute difference: {difference.max():.12g}")
    print(f"Reference mean absolute difference: {difference.mean():.12g}")
    print(f"Reference hard-label differences: {int((reference_label != new_label).sum())}")


def make_submission(
    images: str | Path,
    sample: str | Path,
    output: str | Path = "submission.csv",
    device: str = "gpu:0",
    batch_size: int = 128,
    mode: str = "exact",
    reference: str | Path | None = None,
    limit: int | None = None,
    reuse_e2: str | Path | None = None,
) -> pd.DataFrame:
    """Run the complete solution and save a submission CSV."""
    if mode not in {"exact", "sparse"}:
        raise ValueError("mode must be either 'exact' or 'sparse'")

    images = Path(images).resolve()
    sample = Path(sample).resolve()
    output = Path(output).resolve()
    reference = Path(reference).resolve() if reference else None
    reuse_e2 = Path(reuse_e2).resolve() if reuse_e2 else None

    started_at = time.perf_counter()
    config = json.loads((PACKAGE_DIR / "config.json").read_text(encoding="utf-8"))
    sample_submission = pd.read_csv(sample)
    if limit:
        sample_submission = sample_submission.head(limit).copy()
    image_paths = find_images(images, sample_submission.image_id.tolist())

    if reuse_e2:
        saved_e2 = pd.read_csv(reuse_e2)
        saved_e2 = sample_submission[["image_id"]].merge(saved_e2, on="image_id", validate="one_to_one")
        e2_logit = logit(saved_e2.p_180.to_numpy(float))
        print(f"Reused E2 probabilities from: {reuse_e2}")
    else:
        e2_logit = predict_e2(image_paths, config, device, batch_size)
    final_probability, routed = run_cascade(
        e2_logit, image_paths, config, device, batch_size, mode
    )
    submission = pd.DataFrame({"image_id": sample_submission.image_id, "p_180": final_probability})
    check_submission(submission, sample_submission)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Routed images: {int(routed.sum())}")
    print(f"Total runtime: {(time.perf_counter() - started_at) / 60:.2f} minutes")
    if reference:
        compare_csv(submission, reference)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the E5 Avito orientation submission")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--limit", type=int, help="Run only the first N rows for a smoke test")
    parser.add_argument("--reuse-e2", type=Path, help="Reuse saved E2 probabilities for verification")
    parser.add_argument(
        "--mode", choices=("exact", "sparse"), default="exact",
        help="exact reproduces the submitted CSV; sparse runs extra OCR only on hard cases",
    )
    args = parser.parse_args()
    make_submission(
        images=args.images,
        sample=args.sample,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        mode=args.mode,
        reference=args.reference,
        limit=args.limit,
        reuse_e2=args.reuse_e2,
    )


if __name__ == "__main__":
    main()
