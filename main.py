"""
Command Line Interface (CLI) for Face Recognition Identification System.

Supported commands:
- enroll:   python main.py enroll --name "Alice" --images img1.jpg img2.jpg (or --folder ./alice_imgs)
- identify: python main.py identify --image query.jpg [--threshold 0.40] [--save-annotated out.jpg]
- list:     python main.py list
- remove:   python main.py remove --name "Alice"
- evaluate: python main.py evaluate
"""

import argparse
import sys
from pathlib import Path
from typing import List
import numpy as np

import config
from src.embedder import Embedder
from src.database import FaceDatabase
from src.matcher import FaceMatcher
from src.utils import load_image, draw_annotations, save_image


def handle_enroll(args: argparse.Namespace) -> None:
    """Handle enrollment command."""
    image_paths: List[Path] = []
    
    if args.images:
        for p in args.images:
            image_paths.append(Path(p))
    elif args.folder:
        folder_p = Path(args.folder)
        if not folder_p.exists() or not folder_p.is_dir():
            print(f"[!] Error: Folder path '{args.folder}' does not exist or is not a directory.")
            sys.exit(1)
        # Collect image files
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]:
            image_paths.extend(list(folder_p.glob(ext)))
    else:
        print("[!] Error: Either --images or --folder must be specified for enrollment.")
        sys.exit(1)

    if not image_paths:
        print(f"[!] Error: No valid image files found to enroll for '{args.name}'.")
        sys.exit(1)

    print(f"[+] Initializing system for enrollment of '{args.name}' ({len(image_paths)} images)...")
    embedder = Embedder()
    db = FaceDatabase()
    
    added, skipped = db.enroll(args.name, image_paths, embedder, verbose=True)
    print(f"[+] Enrollment complete for '{args.name}'. {added} faces added, {skipped} skipped.")


def handle_identify(args: argparse.Namespace) -> None:
    """Handle identification command."""
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[!] Error: Query image path '{args.image}' does not exist.")
        sys.exit(1)

    print(f"[+] Identifying faces in '{img_path}'...")
    embedder = Embedder()
    db = FaceDatabase()
    db_embs = db.get_all_embeddings()

    if not db_embs:
        print("[!] Warning: Face database is currently empty. Please enroll identities first.")

    threshold = args.threshold if args.threshold is not None else config.MATCH_THRESHOLD
    matcher = FaceMatcher(threshold=threshold)

    try:
        img = load_image(img_path)
    except Exception as e:
        print(f"[!] Error loading image: {e}")
        sys.exit(1)

    faces = embedder.get_faces(img)
    if not faces:
        print("[!] Result: NO FACE DETECTED in image.")
        sys.exit(0)

    print(f"[+] Detected {len(faces)} face(s) in image.")
    annotated_results = []

    for idx, face in enumerate(faces, start=1):
        query_emb = face["embedding"]
        match_res = matcher.match(query_emb, db_embs, top_k=3, threshold_override=threshold)

        name = match_res["name"]
        score = match_res["score"]
        best_match_person = match_res["best_match_person"]

        print(f"\n--- Face #{idx} (Confidence: {face['score']:.2f}, Box: {face['bbox']}) ---")
        if name != "unknown":
            print(f"  --> MATCH FOUND : '{name}' (Similarity Score: {score:.3f} >= Threshold {threshold:.2f})")
        else:
            print(f"  --> REJECTED    : UNKNOWN (Best Match: '{best_match_person}' with Score: {score:.3f} < Threshold {threshold:.2f})")

        print("  Top-3 Candidates:")
        for cand_name, cand_score in match_res["top_candidates"]:
            print(f"     - {cand_name:<20}: Similarity = {cand_score:.4f}")

        annotated_results.append({
            "bbox": face["bbox"],
            "det_score": face["score"],
            "name": name,
            "score": score
        })

    if args.save_annotated:
        out_p = Path(args.save_annotated)
        annotated_img = draw_annotations(img, annotated_results)
        save_image(annotated_img, out_p)
        print(f"\n[+] Annotated output saved to {out_p}")


def handle_list(args: argparse.Namespace) -> None:
    """Handle database listing command."""
    db = FaceDatabase()
    people = db.list_people()

    print("\n" + "=" * 60)
    print("           ENROLLED IDENTITY DATABASE LIST")
    print("=" * 60)
    if not people:
        print("Database is empty.")
    else:
        print(f"{'Identity Name':<30} | {'Enrolled Images':<15}")
        print("-" * 60)
        for p in people:
            print(f"{p['name']:<30} | {p['num_images']:<15}")
    print("=" * 60)


def handle_remove(args: argparse.Namespace) -> None:
    """Handle identity removal command."""
    db = FaceDatabase()
    db.remove(args.name)


def handle_evaluate(args: argparse.Namespace) -> None:
    """Handle full evaluation command."""
    from src.evaluate import main as eval_main
    eval_main()


def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition Identification System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="System sub-commands")

    # Command: enroll
    parser_enroll = subparsers.add_parser("enroll", help="Enroll a person identity into database")
    parser_enroll.add_argument("--name", type=str, required=True, help="Person name identity")
    parser_enroll.add_argument("--images", nargs="+", help="List of image file paths")
    parser_enroll.add_argument("--folder", type=str, help="Directory containing person images")
    parser_enroll.set_defaults(func=handle_enroll)

    # Command: identify
    parser_identify = subparsers.add_parser("identify", help="Identify person in a query image")
    parser_identify.add_argument("--image", type=str, required=True, help="Query image file path")
    parser_identify.add_argument("--threshold", type=float, help="Override default matching threshold")
    parser_identify.add_argument("--save-annotated", type=str, help="Save annotated output image path")
    parser_identify.set_defaults(func=handle_identify)

    # Command: list
    parser_list = subparsers.add_parser("list", help="List all enrolled identities in database")
    parser_list.set_defaults(func=handle_list)

    # Command: remove
    parser_remove = subparsers.add_parser("remove", help="Remove an identity from database")
    parser_remove.add_argument("--name", type=str, required=True, help="Person name identity to remove")
    parser_remove.set_defaults(func=handle_remove)

    # Command: evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Run full system evaluation pipeline on LFW split")
    parser_eval.set_defaults(func=handle_evaluate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
