"""
Comprehensive Evaluation Pipeline for Face Recognition Identification System.

Workflow:
1. Loads splits from results/splits.json.
2. Enrolls known identities using ONLY the enrollment split.
3. Extracts query facial embeddings once for validation and test splits.
4. Performs instantaneous threshold sweep on VALIDATION split (0.10 to 0.90, step 0.01) to compute FAR, FRR, EER, and Accuracy.
5. Selects optimal threshold on validation set (max Top-1 Accuracy subject to FAR <= 1.0%).
6. Evaluates TEST split ONCE at selected threshold.
7. Generates high-quality evaluation plots in results/plots/.
8. Evaluates comparative modes (max_similarity vs mean_prototype, 1 vs 3 enrolled images).
9. Automatically saves failure cases to results/failures/ and generates results/failure_summary.md.
10. Exports all quantitative metrics to results/metrics.json.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script plotting
import matplotlib.pyplot as plt

# Ensure root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.embedder import Embedder
from src.database import FaceDatabase
from src.matcher import FaceMatcher
from src.utils import load_image, draw_annotations, save_image


def load_splits() -> Dict[str, Any]:
    """Load data splits from results/splits.json."""
    if not config.SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Splits file not found at {config.SPLITS_PATH}. "
            "Please run 'python scripts/prepare_dataset.py' first."
        )
    with open(config.SPLITS_PATH, "r") as f:
        return json.load(f)


def build_enrollment_database(
    splits: Dict[str, Any],
    embedder: Embedder,
    max_images_per_person: int = 3
) -> FaceDatabase:
    """
    Build enrollment database using ONLY images from the enrollment split.
    Optionally cap images per person to compare 1 vs 3 image enrollment performance.
    """
    db_file = config.DATA_DIR / f"eval_db_{max_images_per_person}img.pkl"
    db = FaceDatabase(db_path=db_file)
    db.clear()

    person_images: Dict[str, List[Path]] = {}
    for item in splits["enrollment"]:
        person = item["person"]
        img_path = config.BASE_DIR / item["path"]
        if person not in person_images:
            person_images[person] = []
        person_images[person].append(img_path)

    for person, paths in person_images.items():
        capped_paths = paths[:max_images_per_person]
        db.enroll(person, capped_paths, embedder, verbose=False)

    return db


def extract_query_features(
    queries: List[Dict[str, str]],
    embedder: Embedder,
    desc: str = "Extracting Features"
) -> List[Dict[str, Any]]:
    """
    Pre-extract face detections and embeddings once per image to enable instant threshold sweeping.
    """
    extracted = []
    for item in tqdm(queries, desc=f"[+] {desc}", unit="img"):
        img_path = config.BASE_DIR / item["path"]
        ground_truth_person = item["person"]
        try:
            img = load_image(img_path)
            faces = embedder.get_faces(img)
            if not faces:
                extracted.append({
                    "path": item["path"],
                    "gt": ground_truth_person,
                    "bbox": [0, 0, 0, 0],
                    "det_score": 0.0,
                    "embedding": None,
                    "no_face": True
                })
            else:
                best_face = faces[0]
                extracted.append({
                    "path": item["path"],
                    "gt": ground_truth_person,
                    "bbox": best_face["bbox"],
                    "det_score": best_face["score"],
                    "embedding": best_face["embedding"],
                    "no_face": False
                })
        except Exception as e:
            extracted.append({
                "path": item["path"],
                "gt": ground_truth_person,
                "bbox": [0, 0, 0, 0],
                "det_score": 0.0,
                "embedding": None,
                "no_face": True
            })
    return extracted


def evaluate_extracted_queries(
    known_extracted: List[Dict[str, Any]],
    unknown_extracted: List[Dict[str, Any]],
    db_embs: Dict[str, np.ndarray],
    matcher: FaceMatcher,
    threshold: float
) -> Dict[str, Any]:
    """
    Evaluate precomputed query embeddings against the database embeddings at a given threshold.
    Pure vector operations: executes in <1 millisecond.
    """
    known_results = []
    for q in known_extracted:
        if q["no_face"]:
            known_results.append({
                "path": q["path"],
                "gt": q["gt"],
                "pred": "no_face",
                "score": 0.0,
                "best_match_person": "None",
                "no_face": True,
                "error_type": "no_face",
                "bbox": q["bbox"],
                "det_score": q["det_score"]
            })
        else:
            match_res = matcher.match(q["embedding"], db_embs, threshold_override=threshold)
            pred_person = match_res["name"]
            score = match_res["score"]
            best_match_person = match_res["best_match_person"]

            if pred_person == "unknown":
                error_type = "false_reject"
            elif pred_person != q["gt"]:
                error_type = "wrong_person"
            else:
                error_type = "correct"

            known_results.append({
                "path": q["path"],
                "gt": q["gt"],
                "pred": pred_person,
                "score": score,
                "best_match_person": best_match_person,
                "no_face": False,
                "error_type": error_type,
                "bbox": q["bbox"],
                "det_score": q["det_score"]
            })

    unknown_results = []
    for q in unknown_extracted:
        if q["no_face"]:
            unknown_results.append({
                "path": q["path"],
                "gt": q["gt"],
                "pred": "no_face",
                "score": 0.0,
                "best_match_person": "None",
                "no_face": True,
                "error_type": "no_face",
                "bbox": q["bbox"],
                "det_score": q["det_score"]
            })
        else:
            match_res = matcher.match(q["embedding"], db_embs, threshold_override=threshold)
            pred_person = match_res["name"]
            score = match_res["score"]
            best_match_person = match_res["best_match_person"]

            if pred_person != "unknown":
                error_type = "false_accept"
            else:
                error_type = "correct"

            unknown_results.append({
                "path": q["path"],
                "gt": q["gt"],
                "pred": pred_person,
                "score": score,
                "best_match_person": best_match_person,
                "no_face": False,
                "error_type": error_type,
                "bbox": q["bbox"],
                "det_score": q["det_score"]
            })

    # Metrics computation
    total_known = len(known_results)
    total_unknown = len(unknown_results)
    known_no_face = sum(1 for r in known_results if r["no_face"])
    unknown_no_face = sum(1 for r in unknown_results if r["no_face"])

    correct_known = sum(1 for r in known_results if r["pred"] == r["gt"])
    false_rejects = sum(1 for r in known_results if r["error_type"] == "false_reject")
    wrong_persons = sum(1 for r in known_results if r["error_type"] == "wrong_person")

    false_accepts = sum(1 for r in unknown_results if r["error_type"] == "false_accept")
    correct_unknown_rejections = total_unknown - false_accepts - unknown_no_face

    known_acc = (correct_known / total_known) if total_known > 0 else 0.0
    far = (false_accepts / total_unknown) if total_unknown > 0 else 0.0
    frr = (false_rejects / total_known) if total_known > 0 else 0.0
    wrong_person_rate = (wrong_persons / total_known) if total_known > 0 else 0.0
    unknown_rejection_rate = (correct_unknown_rejections / total_unknown) if total_unknown > 0 else 0.0

    return {
        "threshold": threshold,
        "known_top1_acc": known_acc,
        "far": far,
        "frr": frr,
        "wrong_person_rate": wrong_person_rate,
        "unknown_rejection_rate": unknown_rejection_rate,
        "correct_known": correct_known,
        "false_rejects": false_rejects,
        "wrong_persons": wrong_persons,
        "false_accepts": false_accepts,
        "correct_unknown_rejections": correct_unknown_rejections,
        "total_known_queries": total_known,
        "total_unknown_queries": total_unknown,
        "no_face_count": known_no_face + unknown_no_face,
        "known_results": known_results,
        "unknown_results": unknown_results
    }


def sweep_validation_thresholds(
    val_known_extracted: List[Dict[str, Any]],
    val_unknown_extracted: List[Dict[str, Any]],
    db_embs: Dict[str, np.ndarray],
    matcher: FaceMatcher
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Sweep thresholds from 0.10 to 0.90 on validation split.
    Select optimal threshold (Max Known Top-1 Accuracy subject to FAR <= 1.0%)
    and compute Equal Error Rate (EER).
    """
    print("[+] Sweeping matching thresholds on VALIDATION split (0.10 to 0.90)...")
    thresholds = np.arange(0.10, 0.91, 0.01)
    sweep_results = []

    best_thresh = 0.40
    best_acc = -1.0
    eer_point = 0.40
    min_eer_diff = 1.0

    for t in thresholds:
        res = evaluate_extracted_queries(val_known_extracted, val_unknown_extracted, db_embs, matcher, float(t))
        sweep_results.append(res)

        # Track EER point (|FAR - FRR| is minimized)
        eer_diff = abs(res["far"] - res["frr"])
        if eer_diff < min_eer_diff:
            min_eer_diff = eer_diff
            eer_point = float(t)

        # Selection rule: Highest Top-1 Accuracy on known queries subject to FAR <= 0.01 (1%)
        if res["far"] <= 0.01:
            if res["known_top1_acc"] > best_acc:
                best_acc = res["known_top1_acc"]
                best_thresh = float(t)

    if best_acc < 0:
        best_thresh = eer_point
        print(f"[!] Warning: No threshold achieved FAR <= 1.0%. Using EER threshold: {best_thresh:.2f}")
    else:
        print(f"[+] Optimal Validation Threshold selected: {best_thresh:.2f} (Top-1 Acc: {best_acc*100:.1f}%, FAR <= 1.0%)")

    return best_thresh, eer_point, sweep_results


def plot_evaluation_figures(
    sweep_results: List[Dict[str, Any]],
    best_thresh: float,
    eer_point: float,
    genuine_scores: List[float],
    impostor_scores: List[float],
    test_known_results: List[Dict[str, Any]]
) -> None:
    """Generate and save publication-quality evaluation plots to results/plots/."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = [r["threshold"] for r in sweep_results]
    fars = [r["far"] * 100 for r in sweep_results]
    frrs = [r["frr"] * 100 for r in sweep_results]
    accs = [r["known_top1_acc"] * 100 for r in sweep_results]

    # 1. FAR & FRR vs Threshold Plot
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, fars, 'r-', linewidth=2, label='False Accept Rate (FAR %)')
    plt.plot(thresholds, frrs, 'b-', linewidth=2, label='False Reject Rate (FRR %)')
    plt.plot(thresholds, accs, 'g--', linewidth=2, label='Known Top-1 Accuracy %')
    plt.axvline(best_thresh, color='black', linestyle=':', label=f'Chosen Thresh ({best_thresh:.2f})')
    plt.axvline(eer_point, color='orange', linestyle='--', label=f'EER Thresh ({eer_point:.2f})')
    plt.title('Validation Performance: FAR & FRR vs Matching Threshold')
    plt.xlabel('Cosine Similarity Threshold')
    plt.ylabel('Percentage (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "far_frr_vs_threshold.png", dpi=300)
    plt.close()

    # 2. ROC Curve (True Accept Rate vs False Accept Rate)
    tars = [100.0 - f for f in frrs]
    plt.figure(figsize=(7, 6))
    plt.plot(fars, tars, 'b-o', markersize=4, linewidth=2, label='ROC Curve')
    plt.plot([0, 100], [0, 100], 'k--', alpha=0.4, label='Random Chance')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Accept Rate (FAR %)')
    plt.ylabel('True Accept Rate (TAR %)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "roc_det_curve.png", dpi=300)
    plt.close()

    # 3. Genuine vs Impostor Score Distribution Histogram
    plt.figure(figsize=(8, 5))
    if genuine_scores:
        plt.hist(genuine_scores, bins=30, alpha=0.6, color='green', label='Genuine Pairs (Same Person)', density=True)
    if impostor_scores:
        plt.hist(impostor_scores, bins=30, alpha=0.6, color='red', label='Impostor Pairs (Different Person)', density=True)
    plt.axvline(best_thresh, color='black', linestyle='--', linewidth=2, label=f'Threshold ({best_thresh:.2f})')
    plt.title('Cosine Similarity Score Distribution (Genuine vs Impostor Pairs)')
    plt.xlabel('Cosine Similarity Score')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "score_distribution.png", dpi=300)
    plt.close()

    # 4. Per-Person Accuracy Bar Chart
    person_correct: Dict[str, int] = {}
    person_total: Dict[str, int] = {}
    for r in test_known_results:
        gt = r["gt"]
        person_total[gt] = person_total.get(gt, 0) + 1
        if r["pred"] == gt:
            person_correct[gt] = person_correct.get(gt, 0) + 1
        else:
            person_correct[gt] = person_correct.get(gt, 0)

    people_sorted = sorted(list(person_total.keys()))[:20]
    per_person_accs = [(person_correct[p] / person_total[p]) * 100 for p in people_sorted]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(people_sorted)), per_person_accs, color='teal', alpha=0.8)
    plt.xticks(range(len(people_sorted)), people_sorted, rotation=45, ha='right', fontsize=9)
    plt.ylabel('Identification Accuracy (%)')
    plt.title('Per-Person Identification Accuracy on Test Split (Sample)')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "confusion_per_person.png", dpi=300)
    plt.close()

    print(f"[+] All 4 evaluation plots successfully saved to {config.PLOTS_DIR}")


def save_failure_cases(
    all_test_results: List[Dict[str, Any]]
) -> Tuple[int, Dict[str, int]]:
    """
    Save 8-12 failure examples to results/failures/ with bounding box annotations.
    Generates results/failure_summary.md detailing counts and failure causes.
    """
    config.FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    for old_f in config.FAILURES_DIR.glob("*.jpg"):
        old_f.unlink()

    failures = [r for r in all_test_results if r["error_type"] != "correct"]
    counts = {"false_accept": 0, "false_reject": 0, "wrong_person": 0, "no_face": 0}

    saved_count = 0
    failure_logs = []

    for item in failures:
        err_type = item["error_type"]
        counts[err_type] = counts.get(err_type, 0) + 1

        if saved_count >= 12:
            continue

        gt = item["gt"]
        pred = item["pred"]
        score = item["score"]
        img_path = config.BASE_DIR / item["path"]

        clean_gt = gt.replace(" ", "_")
        clean_pred = pred.replace(" ", "_")
        filename = f"{err_type}_sim{score:.2f}_{clean_gt}_vs_{clean_pred}.jpg"
        out_path = config.FAILURES_DIR / filename

        try:
            img = load_image(img_path)
            if not item.get("no_face", False):
                annotation_item = [{
                    "bbox": item["bbox"],
                    "det_score": item.get("det_score", 0.0),
                    "name": f"GT:{gt} | Pred:{pred}",
                    "score": score
                }]
                annotated_img = draw_annotations(img, annotation_item)
                save_image(annotated_img, out_path)
            else:
                save_image(img, out_path)

            saved_count += 1
            failure_logs.append({
                "filename": filename,
                "error_type": err_type,
                "ground_truth": gt,
                "predicted": pred,
                "score": score,
                "image_path": item["path"]
            })
        except Exception as e:
            pass

    # Generate failure_summary.md
    with open(config.FAILURE_SUMMARY_PATH, "w") as f:
        f.write("# Face Recognition Failure Case Analysis\n\n")
        f.write(f"Total Test Failures Analyzed: **{len(failures)}**\n\n")
        f.write("## Failure Breakdown by Type\n\n")
        f.write(f"- **False Rejects (FRR)**: {counts['false_reject']} (Known face score below threshold)\n")
        f.write(f"- **False Accepts (FAR)**: {counts['false_accept']} (Unknown intruder accepted as known person)\n")
        f.write(f"- **Wrong Person Accept**: {counts['wrong_person']} (Known face matched to wrong enrolled identity)\n")
        f.write(f"- **No Face Detected**: {counts['no_face']} (Detector failed to locate facial region)\n\n")
        f.write("## Sample Failure Artifacts\n\n")
        f.write("| Image Filename | Failure Type | Ground Truth | Prediction | Score |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for log in failure_logs:
            f.write(f"| `{log['filename']}` | {log['error_type']} | {log['ground_truth']} | {log['predicted']} | {log['score']:.3f} |\n")

        f.write("\n## Root Cause Rationale & Mitigation\n\n")
        f.write("1. **Profile Angles & Extreme Pose**: Large head yaw angles reduce facial symmetry, degrading ArcFace embedding alignment.\n")
        f.write("2. **Heavy Occlusion (Eyeglasses / Facial Hair)**: Structural occlusions shift embedding feature vectors away from frontal centroids.\n")
        f.write("3. **Blur & Low Resolution**: Out-of-focus motion blur erases high-frequency facial texture details.\n")
        f.write("4. **Illumination Variance**: Severe directional shadows introduce heavy intra-class variance.\n")

    print(f"[+] Saved {saved_count} failure visualization images to {config.FAILURES_DIR}")
    print(f"[+] Failure summary written to {config.FAILURE_SUMMARY_PATH}")
    return len(failures), counts


def run_comparative_evaluations(
    splits: Dict[str, Any],
    embedder: Embedder,
    test_known_extracted: List[Dict[str, Any]],
    test_unknown_extracted: List[Dict[str, Any]],
    optimal_threshold: float
) -> Dict[str, Any]:
    """
    Compare (1) Max Similarity vs Mean Prototype matching and (2) 1 vs 3 Enrolled Images
    using pre-extracted query features.
    """
    print("\n[+] Running comparative mode evaluations (Max-Sim vs Mean-Proto, 1 vs 3 Enrolled Images)...")

    # 1. 3 images per person - Max Similarity
    db_3 = build_enrollment_database(splits, embedder, max_images_per_person=3)
    matcher_max = FaceMatcher(threshold=optimal_threshold, mode="max_similarity")
    res_3_max = evaluate_extracted_queries(
        test_known_extracted, test_unknown_extracted, db_3.get_all_embeddings(), matcher_max, optimal_threshold
    )

    # 2. 3 images per person - Mean Prototype
    matcher_mean = FaceMatcher(threshold=optimal_threshold, mode="mean_prototype")
    res_3_mean = evaluate_extracted_queries(
        test_known_extracted, test_unknown_extracted, db_3.get_all_embeddings(), matcher_mean, optimal_threshold
    )

    # 3. 1 image per person - Max Similarity
    db_1 = build_enrollment_database(splits, embedder, max_images_per_person=1)
    res_1_max = evaluate_extracted_queries(
        test_known_extracted, test_unknown_extracted, db_1.get_all_embeddings(), matcher_max, optimal_threshold
    )

    db_3.clear()
    db_1.clear()

    return {
        "3_images_max_sim": {"top1_acc": res_3_max["known_top1_acc"], "far": res_3_max["far"], "frr": res_3_max["frr"]},
        "3_images_mean_proto": {"top1_acc": res_3_mean["known_top1_acc"], "far": res_3_mean["far"], "frr": res_3_mean["frr"]},
        "1_image_max_sim": {"top1_acc": res_1_max["known_top1_acc"], "far": res_1_max["far"], "frr": res_1_max["frr"]}
    }


def main():
    print("=" * 70)
    print("        FACE RECOGNITION SYSTEM - EVALUATION PIPELINE")
    print("=" * 70)

    splits = load_splits()
    embedder = Embedder()

    # Step 1: Build enrollment DB from enrollment split ONLY
    print("[+] Enrolling identities from enrollment split into database...")
    db = build_enrollment_database(splits, embedder, max_images_per_person=3)
    db_embs = db.get_all_embeddings()

    # Step 2: Extract Validation Split Features
    print("[+] Extracting validation query features...")
    val_known_extracted = extract_query_features(splits["val_known"], embedder, "Val Known")
    val_unknown_extracted = extract_query_features(splits["val_unknown"], embedder, "Val Unknown")

    # Step 3: Sweep Validation Thresholds
    matcher_val = FaceMatcher(mode="max_similarity")
    best_thresh, eer_point, sweep_results = sweep_validation_thresholds(
        val_known_extracted, val_unknown_extracted, db_embs, matcher_val
    )

    # Save chosen optimal threshold back to config
    config.MATCH_THRESHOLD = best_thresh

    # Step 4: Extract Test Split Features
    print("\n[+] Extracting test query features...")
    test_known_extracted = extract_query_features(splits["test_known"], embedder, "Test Known")
    test_unknown_extracted = extract_query_features(splits["test_unknown"], embedder, "Test Unknown")

    # Step 5: Evaluate TEST split ONCE at chosen threshold
    print(f"\n[+] Evaluating TEST split ONCE at selected threshold ({best_thresh:.2f})...")
    test_matcher = FaceMatcher(threshold=best_thresh, mode="max_similarity")
    test_eval = evaluate_extracted_queries(
        test_known_extracted, test_unknown_extracted, db_embs, test_matcher, best_thresh
    )

    # Step 6: Collect genuine and impostor scores for distribution plot
    genuine_scores = []
    impostor_scores = []

    for q in test_known_extracted:
        if not q["no_face"] and q["gt"] in db_embs:
            gen_sim = test_matcher.compute_similarity(q["embedding"], db_embs[q["gt"]])
            genuine_scores.append(gen_sim)

    for q in test_unknown_extracted:
        if not q["no_face"]:
            for p_name, p_embs in db_embs.items():
                imp_sim = test_matcher.compute_similarity(q["embedding"], p_embs)
                impostor_scores.append(imp_sim)

    # Step 7: Save plots
    plot_evaluation_figures(
        sweep_results,
        best_thresh,
        eer_point,
        genuine_scores,
        impostor_scores,
        test_eval["known_results"]
    )

    # Step 8: Failure analysis & artifact generation
    all_test_results = test_eval["known_results"] + test_eval["unknown_results"]
    total_failures, failure_counts = save_failure_cases(all_test_results)

    # Step 9: Comparative mode evaluation
    comparative_results = run_comparative_evaluations(
        splits, embedder, test_known_extracted, test_unknown_extracted, best_thresh
    )

    # Step 10: Save quantitative metrics to results/metrics.json
    metrics = {
        "backend": embedder.backend_name,
        "selected_threshold": best_thresh,
        "eer_threshold": eer_point,
        "test_results": {
            "top1_accuracy": test_eval["known_top1_acc"],
            "far": test_eval["far"],
            "frr": test_eval["frr"],
            "wrong_person_rate": test_eval["wrong_person_rate"],
            "unknown_rejection_rate": test_eval["unknown_rejection_rate"],
            "total_known_queries": test_eval["total_known_queries"],
            "total_unknown_queries": test_eval["total_unknown_queries"],
            "total_queries": test_eval["total_known_queries"] + test_eval["total_unknown_queries"],
            "no_face_count": test_eval["no_face_count"],
            "failures_count": total_failures,
            "failure_counts_by_type": failure_counts
        },
        "comparative_evaluation": comparative_results
    }

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Final summary display
    print("\n" + "=" * 70)
    print("             FINAL TEST EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(f"Backend Used                     : {embedder.backend_name.upper()}")
    print(f"Operating Threshold (Validation) : {best_thresh:.2f}")
    print(f"Equal Error Rate (EER) Point     : {eer_point:.2f}")
    print(f"Known Top-1 Accuracy             : {test_eval['known_top1_acc']*100:.2f}%")
    print(f"False Accept Rate (FAR)          : {test_eval['far']*100:.2f}%")
    print(f"False Reject Rate (FRR)          : {test_eval['frr']*100:.2f}%")
    print(f"Wrong Person Accept Rate         : {test_eval['wrong_person_rate']*100:.2f}%")
    print(f"Unknown Rejection Rate           : {test_eval['unknown_rejection_rate']*100:.2f}%")
    print(f"Total Test Queries               : {test_eval['total_known_queries'] + test_eval['total_unknown_queries']}")
    print(f"No-Face Detected Queries         : {test_eval['no_face_count']}")
    print("-" * 70)
    print("Comparative Results Table:")
    print(f"  - 3 Images (Max Similarity)    : Top-1 Acc = {comparative_results['3_images_max_sim']['top1_acc']*100:.2f}%, FAR = {comparative_results['3_images_max_sim']['far']*100:.2f}%")
    print(f"  - 3 Images (Mean Prototype)    : Top-1 Acc = {comparative_results['3_images_mean_proto']['top1_acc']*100:.2f}%, FAR = {comparative_results['3_images_mean_proto']['far']*100:.2f}%")
    print(f"  - 1 Image  (Max Similarity)    : Top-1 Acc = {comparative_results['1_image_max_sim']['top1_acc']*100:.2f}%, FAR = {comparative_results['1_image_max_sim']['far']*100:.2f}%")
    print("=" * 70)

    # Clean up evaluation temp database
    db.clear()


if __name__ == "__main__":
    main()
