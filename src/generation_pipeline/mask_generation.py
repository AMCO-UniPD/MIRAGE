"""
Mask Generation Pipeline for Anomalous Images

Generates binary ground truth masks for anomalous images using Gemini API.
Takes both normal and anomalous images as input to generate precise segmentation masks.
"""

import re
import sys
import time
import json
import argparse
import threading
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import trange, tqdm
from datetime import datetime
from google import genai
from dotenv import load_dotenv
import google.genai.errors

# Import configurations and utilities from main.py
from main import DATASET_CONFIGS, _project_root

# Global variables for mask tracking
mask_tracking_csv_path = None
generated_dataset_dir = None


def create_mask_prompt(object_name: str, defect_description: str) -> str:
    """
    Create prompt for mask generation.

    Args:
        object_name: Name of the object (e.g., "bottle", "cable")
        defect_description: Description of the defect

    Returns:
        Formatted prompt string for mask generation
    """
    return (
        f"You are analyzing two images of a {object_name}:\n"
        f"Image 1: A normal, defect-free {object_name}\n"
        f"Image 2: The same {object_name} with this specific defect: {defect_description}\n\n"
        f"TASK: Create a PURE BINARY SEGMENTATION MASK ON TOP OF the defective region.\n\n"
        f"CRITICAL REQUIREMENTS:\n"
        f"The defective area should be covered by white in the output, and the rest should be black background.\n"
        f"Mark ONLY the defective/anomalous region which is different between the two images\n"
        f"Generate the binary mask now."
    )


def generate_mask(
    client,
    normal_img: Image.Image,
    anomalous_img: Image.Image,
    prompt: str,
    max_retries: int = 3,
) -> Image.Image:
    """
    Generate mask using Gemini API with retry logic.

    Args:
        client: Gemini API client
        normal_img: Normal/base image
        anomalous_img: Anomalous image
        prompt: Generation prompt
        max_retries: Maximum number of retry attempts

    Returns:
        Generated mask image, or None if failed
    """
    # Spinner animation
    spinner_active = True

    def spinner():
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        while spinner_active:
            sys.stdout.write(f"\r      Generating mask... {chars[idx % len(chars)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1

    for attempt in range(max_retries):
        spinner_thread = threading.Thread(target=spinner)
        spinner_thread.start()

        try:
            response = client.models.generate_content(
                # model="gemini-2.5-flash-image",
                model="gemini-3-pro-image-preview",
                contents=[prompt, normal_img, anomalous_img],
                # contents=[prompt, anomalous_img],
            )

            if (
                not response.candidates
                or not response.candidates[0].content
                or not response.candidates[0].content.parts
            ):
                print("      No valid response parts found")
                return None

            # Look for image data in response
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    # API sometimes returns text along with the image
                    pass
                elif part.inline_data is not None:
                    mask_image = Image.open(BytesIO(part.inline_data.data))
                    return mask_image

            print("      No inline_data found in response")
            return None

        except google.genai.errors.ClientError as e:
            # Parse retry delay from error
            match = re.search(r"'retryDelay':\s*'(\d+)s'", str(e.details))
            time_to_sleep = int(match.group(1)) if match else 60

            print(f"\n      Quota exceeded, sleeping for {time_to_sleep} seconds...")
            for i in trange(time_to_sleep, desc="      Waiting"):
                time.sleep(1)

            if attempt < max_retries - 1:
                print(f"      Retrying (attempt {attempt + 2}/{max_retries})...")

        except Exception as e:
            print(f"\n      Error during mask generation: {e}")
            if attempt < max_retries - 1:
                print(f"      Retrying (attempt {attempt + 2}/{max_retries})...")
                time.sleep(5)

        finally:
            spinner_active = False
            spinner_thread.join()
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()
            spinner_active = True

    # All retries exhausted
    return None


def construct_mask_path(anomalous_img_rel_path: str, category: str) -> Path:
    """
    Construct the mask file path from anomalous image path.

    Args:
        anomalous_img_rel_path: Relative path of anomalous image
        category: Object category

    Returns:
        Absolute path where mask should be saved
    """
    # Parse the path: {category}/test/{defect}/{filename}
    path_parts = Path(anomalous_img_rel_path).parts
    filename = Path(anomalous_img_rel_path).stem

    # Construct mask directory path
    mask_dir = generated_dataset_dir / category / "test" / path_parts[2] / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Mask filename: {original_name}_mask.png
    mask_filename = f"{filename}_mask.png"

    return mask_dir / mask_filename


def apply_fixed_threshold(mask: Image.Image, threshold: int = 250) -> Image.Image:
    """
    Apply fixed thresholding to clean up mask.
    Anything above threshold becomes white (255), rest becomes black (0).

    Args:
        mask: Input mask image (grayscale)
        threshold: Threshold value (default: 250)

    Returns:
        Thresholded binary mask
    """
    # Convert to grayscale if needed
    if mask.mode != "L":
        mask = mask.convert("L")

    mask_array = np.array(mask)

    # Apply fixed thresholding: values > threshold become 255, rest become 0
    _, mask_thresholded = cv2.threshold(mask_array, threshold, 255, cv2.THRESH_BINARY)

    return Image.fromarray(mask_thresholded)


def validate_mask(
    mask: Image.Image, reference_img: Image.Image, relaxed: bool = True
) -> tuple[bool, str]:
    """
    Validate generated mask.

    Args:
        mask: Generated mask image
        reference_img: Reference anomalous image
        relaxed: If True, use relaxed validation (shape matters more than perfect binary)

    Returns:
        (is_valid, message) tuple
    """
    issues = []

    # Check size - auto-resize if needed
    if mask.size != reference_img.size:
        issues.append(
            f"Size mismatch: mask {mask.size} vs reference {reference_img.size} - will auto-resize"
        )

    # Check mode - convert to grayscale if needed
    if mask.mode not in ["L", "1"]:
        if mask.mode in ["RGB", "RGBA"]:
            issues.append(f"Mode is {mask.mode}, will convert to grayscale")
        else:
            issues.append(f"Unexpected mode: {mask.mode}")

    # Convert to array for pixel analysis
    if mask.mode == "RGB":
        mask_gray = mask.convert("L")
    else:
        mask_gray = mask

    mask_array = np.array(mask_gray)

    # Check if mask is not all-black
    if mask_array.max() == 0:
        return False, "Mask is completely black (no anomalous regions detected)"

    # Check if mask is not all-white
    if mask_array.min() == 255:
        return False, "Mask is completely white (entire image marked as anomalous)"

    if relaxed:
        # Relaxed validation: just check reasonable anomalous area
        anomalous_ratio = (mask_array > 127).sum() / mask_array.size

        if anomalous_ratio < 0.001:
            issues.append(
                f"Warning: Very small anomalous area ({anomalous_ratio * 100:.2f}%)"
            )
        elif anomalous_ratio > 0.95:
            issues.append(
                f"Warning: Very large anomalous area ({anomalous_ratio * 100:.2f}%)"
            )

    if issues:
        return True, "; ".join(issues)

    return True, "Valid"


def load_or_create_mask_tracking_csv() -> pd.DataFrame:
    """Load existing mask tracking CSV or create a new one."""
    if mask_tracking_csv_path.exists():
        try:
            return pd.read_csv(mask_tracking_csv_path)
        except pd.errors.EmptyDataError:
            print(
                f"Warning: tracking file '{mask_tracking_csv_path}' is empty. Recreating it."
            )
        except Exception as e:
            print(f"Warning: failed to read tracking file: {e}. Recreating it.")

    # Create new DataFrame with required columns
    df = pd.DataFrame(
        columns=[
            "object",
            "defect",
            "img_path",
            "mask_path",
            "base_img_path",
            "timestamp",
            "validation_status",
        ]
    )

    mask_tracking_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(mask_tracking_csv_path, index=False)
    return df


def save_mask_tracking_csv(df: pd.DataFrame):
    """Save the mask tracking CSV."""
    df.to_csv(mask_tracking_csv_path, index=False)


def add_mask_to_tracking(
    object_name: str,
    defect: str,
    img_path: str,
    mask_path: str,
    base_img_path: str,
    validation_status: str,
) -> pd.DataFrame:
    """
    Add a newly generated mask to the tracking CSV and save immediately.

    Returns:
        Updated DataFrame
    """
    df = load_or_create_mask_tracking_csv()

    new_row = pd.DataFrame(
        [
            {
                "object": object_name,
                "defect": defect,
                "img_path": img_path,
                "mask_path": mask_path,
                "base_img_path": base_img_path,
                "timestamp": datetime.now().isoformat(),
                "validation_status": validation_status,
            }
        ]
    )

    df = pd.concat([df, new_row], ignore_index=True)
    save_mask_tracking_csv(df)
    return df


def mask_generation_pipeline(
    dataset_name: str,
    resume: bool = True,
    categories: list = None,
    max_gen: int = None,
    num_target: int = None,
    dry_run: bool = False,
):
    """
    Main mask generation pipeline.

    Args:
        dataset_name: Name of dataset ("mvtec", "visa", "realiad")
        resume: If True, skip existing masks
        categories: List of categories to process (None = all)
        max_gen: Maximum number of masks to generate in this session
        num_target: Target number of total masks per category (existing + new)
        dry_run: If True, don't actually generate masks
    """
    global generated_dataset_dir, mask_tracking_csv_path

    # Set dynamic paths based on dataset
    config = DATASET_CONFIGS[dataset_name]
    dataset_base = config["base_path"]
    generated_dataset_dir = _project_root / "datasets" / config["output_path"]
    mask_tracking_csv_path = generated_dataset_dir / "mask_generation_tracking.csv"

    print(f"\n{'=' * 80}")
    print(f"MASK GENERATION PIPELINE - {dataset_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"Generated images dir: {generated_dataset_dir}")
    print(f"Mask tracking CSV: {mask_tracking_csv_path}")
    print(f"Resume mode: {resume}")
    print(f"Max generation: {max_gen if max_gen else 'unlimited'}")
    print(f"Num target per category: {num_target if num_target else 'unlimited'}")
    print(f"Dry run: {dry_run}")

    # Load API client
    load_dotenv()
    client = genai.Client()
    print("✓ API client initialized")

    # Load image generation tracking CSV
    image_tracking_csv = generated_dataset_dir / "generation_tracking.csv"
    print(f"\n✓ Loading image tracking CSV: {image_tracking_csv}")
    df_images = pd.read_csv(image_tracking_csv)
    print(f"✓ Loaded {len(df_images)} generated images")

    # Filter by categories if specified
    if categories:
        df_images = df_images[df_images["object"].isin(categories)]
        print(f"✓ Filtered to {len(df_images)} images in categories: {categories}")

    # Load mask tracking CSV
    df_masks = load_or_create_mask_tracking_csv()
    print(f"✓ Loaded {len(df_masks)} existing mask entries")

    # Build set of already generated masks for quick lookup
    existing_masks = set()
    if resume and not df_masks.empty:
        existing_masks = set(df_masks["img_path"].values)
        print(f"✓ Resume mode: {len(existing_masks)} masks already generated")

    # Group by category for round-robin processing
    categories_list = df_images["object"].unique()
    print(f"\n✓ Processing {len(categories_list)} categories: {list(categories_list)}")

    # Organize images by category
    category_queues = {}
    for category in categories_list:
        category_df = df_images[df_images["object"] == category]
        category_queues[category] = category_df.to_dict("records")

    # Count existing masks per category for --num-target
    category_mask_counts = {}
    if num_target is not None:
        for category in categories_list:
            if not df_masks.empty and "object" in df_masks.columns:
                category_mask_counts[category] = int(
                    (df_masks["object"] == category).sum()
                )
            else:
                category_mask_counts[category] = 0
            print(f"  {category}: {category_mask_counts[category]}/{num_target} masks")
        # Track which categories already met the target
        categories_done = {
            cat for cat, count in category_mask_counts.items() if count >= num_target
        }
        if categories_done:
            print(f"✓ Categories already at target: {categories_done}")

    # Round-robin processing
    session_count = 0
    total_generated = 0
    total_skipped = 0
    total_failed = 0

    # Create round-robin iterator
    max_len = max(len(queue) for queue in category_queues.values())

    with tqdm(total=len(df_images), desc="Overall progress") as pbar:
        for idx in range(max_len):
            # Check session limit
            if max_gen and session_count >= max_gen:
                print(f"\n✓ Reached session limit ({max_gen} masks generated)")
                break

            # Check if all categories met their target
            if num_target is not None and all(
                category_mask_counts.get(cat, 0) >= num_target
                for cat in categories_list
            ):
                print(
                    f"\n✓ All categories reached target ({num_target} masks per category)"
                )
                break

            # Process one image from each category
            for category in categories_list:
                queue = category_queues[category]
                if idx >= len(queue):
                    continue  # This category exhausted

                # Check per-category target
                if (
                    num_target is not None
                    and category_mask_counts.get(category, 0) >= num_target
                ):
                    pbar.update(1)
                    continue

                # Check session limit
                if max_gen and session_count >= max_gen:
                    break

                row = queue[idx]

                # Skip if mask already exists (resume mode)
                if resume and row["img_path"] in existing_masks:
                    total_skipped += 1
                    pbar.update(1)
                    continue

                # Construct paths
                anomalous_img_path = generated_dataset_dir / row["img_path"]
                normal_img_path = dataset_base / row["base_img_path"]
                mask_path = construct_mask_path(row["img_path"], row["object"])

                # Check if mask file exists (file system check)
                if resume and mask_path.exists():
                    total_skipped += 1
                    existing_masks.add(row["img_path"])
                    pbar.update(1)
                    continue

                # Display progress
                print(f"\n{'-' * 80}")
                print(
                    f"[{session_count + 1}/{max_gen if max_gen else '∞'}] "
                    f"{row['object']}/{row['defect']} - {Path(row['img_path']).name}"
                )
                print(f"  Normal: {normal_img_path.relative_to(dataset_base)}")
                print(
                    f"  Anomalous: {anomalous_img_path.relative_to(generated_dataset_dir)}"
                )
                print(f"  Mask: {mask_path.relative_to(generated_dataset_dir)}")

                if dry_run:
                    print(f"  [DRY RUN] Would generate mask here")
                    total_generated += 1
                    session_count += 1
                    pbar.update(1)
                    continue

                # Load images
                try:
                    normal_img = Image.open(normal_img_path)
                    anomalous_img = Image.open(anomalous_img_path)
                except Exception as e:
                    print(f"  ✗ Failed to load images: {e}")
                    total_failed += 1
                    pbar.update(1)
                    continue

                # Create prompt
                prompt = create_mask_prompt(row["object"], row["defect_synonym"])

                # Generate mask
                print(f"  Generating mask...")
                mask = generate_mask(client, normal_img, anomalous_img, prompt)

                if mask is None:
                    print(f"  ✗ Mask generation failed")
                    total_failed += 1
                    pbar.update(1)
                    continue

                # Validate and process mask
                is_valid, message = validate_mask(mask, anomalous_img)

                if not is_valid:
                    print(f"  ✗ Validation failed: {message}")
                    total_failed += 1
                    pbar.update(1)
                    continue

                if message != "Valid":
                    print(f"  ⚠ {message}")

                # Resize if needed
                if mask.size != anomalous_img.size:
                    mask = mask.resize(anomalous_img.size, Image.Resampling.LANCZOS)

                # Convert to grayscale if needed
                if mask.mode != "L":
                    mask = mask.convert("L")

                # Apply fixed thresholding to clean up the mask (>250 = white, rest = black)
                # mask_thresholded = apply_fixed_threshold(mask, threshold=250)
                # print(f"  Applied fixed threshold: 250")

                # don't apply the threshold
                mask_thresholded = mask

                # Save thresholded mask
                try:
                    mask_thresholded.save(mask_path)
                    print(f"  ✓ Saved thresholded mask to {mask_path.name}")
                except Exception as e:
                    print(f"  ✗ Failed to save mask: {e}")
                    total_failed += 1
                    pbar.update(1)
                    continue

                # Add to tracking CSV immediately
                mask_rel_path = str(mask_path.relative_to(generated_dataset_dir))
                df_masks = add_mask_to_tracking(
                    row["object"],
                    row["defect"],
                    row["img_path"],
                    mask_rel_path,
                    row["base_img_path"],
                    message,
                )

                total_generated += 1
                session_count += 1
                existing_masks.add(row["img_path"])
                if num_target is not None:
                    category_mask_counts[row["object"]] = (
                        category_mask_counts.get(row["object"], 0) + 1
                    )
                pbar.update(1)

                # also generate a matplotlib subplot with 3 images: normal, anomalous, mask, overlayed
                # for visual inspection on the root of the generated dataset

                overlayed_img = anomalous_img.copy().convert("RGBA")
                mask_rgba = mask_thresholded.convert("L").point(
                    lambda p: 255 if p > 127 else 0
                )
                red_mask = Image.new("RGBA", mask_rgba.size, (255, 0, 0, 128))
                overlayed_img.paste(red_mask, (0, 0), mask_rgba)

                import matplotlib.pyplot as plt

                fig, axs = plt.subplots(1, 4, figsize=(15, 5))
                axs[0].imshow(normal_img)
                axs[0].set_title("Normal Image")
                axs[0].axis("off")
                axs[1].imshow(anomalous_img)
                axs[1].set_title("Anomalous Image")
                axs[1].axis("off")
                axs[2].imshow(mask_thresholded, cmap="gray")
                axs[2].set_title("Generated Mask")
                axs[2].axis("off")
                axs[3].imshow(overlayed_img)
                axs[3].set_title("Overlayed")
                axs[3].axis("off")
                viz_path = generated_dataset_dir / "current_mask_viz.png"
                fig.savefig(viz_path)
                plt.close(fig)
                print(f"  ✓ Saved visualization to {viz_path.name}")
                # Clean up
                normal_img.close()
                anomalous_img.close()

    # Final statistics
    print(f"\n{'=' * 80}")
    print("MASK GENERATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total images processed: {len(df_images)}")
    print(f"Masks generated this session: {total_generated}")
    print(f"Masks skipped (already exist): {total_skipped}")
    print(f"Failed generations: {total_failed}")
    print(f"Total masks now available: {len(df_masks)}")
    print(f"Tracking CSV: {mask_tracking_csv_path}")
    print(f"{'=' * 80}")


def validate_existing_masks(dataset_name: str, visualize: int = 0):
    """
    Validate all existing masks in the dataset.

    Args:
        dataset_name: Name of dataset
        visualize: Number of random samples to visualize (0 = none)
    """
    config = DATASET_CONFIGS[dataset_name]
    generated_dataset_dir = _project_root / "datasets" / config["output_path"]
    mask_tracking_csv_path = generated_dataset_dir / "mask_generation_tracking.csv"

    print(f"\n{'=' * 80}")
    print(f"MASK VALIDATION - {dataset_name.upper()}")
    print(f"{'=' * 80}")

    # Load tracking CSV
    df = pd.read_csv(mask_tracking_csv_path)
    print(f"✓ Loaded {len(df)} mask entries")

    issues = []
    missing = []

    for idx, row in df.iterrows():
        mask_path = generated_dataset_dir / row["mask_path"]
        img_path = generated_dataset_dir / row["img_path"]

        # Check if files exist
        if not mask_path.exists():
            missing.append(f"Mask missing: {row['mask_path']}")
            continue

        if not img_path.exists():
            issues.append(f"Anomalous image missing: {row['img_path']}")
            continue

        # Load and validate
        try:
            mask = Image.open(mask_path)
            img = Image.open(img_path)

            is_valid, message = validate_mask(mask, img, relaxed=True)

            if not is_valid:
                issues.append(f"{row['mask_path']}: {message}")
            elif message != "Valid":
                # Just warnings, not critical
                pass

        except Exception as e:
            issues.append(f"{row['mask_path']}: Failed to validate - {e}")

    # Report
    print(f"\n{'=' * 80}")
    print("VALIDATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Total masks: {len(df)}")
    print(f"Missing masks: {len(missing)}")
    print(f"Validation issues: {len(issues)}")

    if missing:
        print(f"\nMissing masks (first 10):")
        for item in missing[:10]:
            print(f"  - {item}")

    if issues:
        print(f"\nValidation issues (first 10):")
        for item in issues[:10]:
            print(f"  - {item}")

    # Visualization
    if visualize > 0:
        print(f"\n✓ Generating {visualize} visualization samples...")
        print("  (Visualization not yet implemented)")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate binary segmentation masks for anomalous images"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mvtec", "visa", "realiad"],
        required=True,
        help="Dataset to process",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Skip existing masks (default: enabled)"
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate all masks (overwrite existing)",
    )

    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="Specific categories to process (default: all)",
    )

    parser.add_argument(
        "--max-gen",
        type=int,
        help="Maximum number of masks to generate in this session",
    )

    parser.add_argument(
        "--num-target",
        type=int,
        help="Target number of total masks per category (existing + new). "
        "Categories that already have this many masks will be skipped.",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without actual generation"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing masks, don't generate new ones",
    )

    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="Number of samples to visualize (used with --validate-only)",
    )

    args = parser.parse_args()

    try:
        if args.validate_only:
            validate_existing_masks(args.dataset, args.visualize)
        else:
            resume = not args.no_resume if args.no_resume else True
            mask_generation_pipeline(
                dataset_name=args.dataset,
                resume=resume,
                categories=args.categories,
                max_gen=args.max_gen,
                num_target=args.num_target,
                dry_run=args.dry_run,
            )
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
