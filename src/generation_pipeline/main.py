import re
import sys
import threading
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from google import genai
from pathlib import Path
from borders import frame
import google.genai.errors
from tqdm import trange
from datetime import datetime
from dotenv import load_dotenv
import PIL.ImageChops as ImageChops

# Dataset configurations
DATASET_CONFIGS = {
    "mvtec": {
        "base_path": Path("datasets/mvtec"),
        "output_path": "all_generated_mvtec",
        "image_extension": "*.png",
    },
    "visa": {
        "base_path": Path("datasets/visa"),
        "output_path": "all_generated_visa",
        "image_extension": "*.JPG",
    },
    "realiad": {
        "base_path": Path("datasets/realiad/realiad_256"),
        "output_path": "all_generated_realiad",
        "image_extension": "*.jpg",
    },
}

# Project paths (relative to project root, two levels up from src/generation_pipeline)
_project_root = Path(__file__).parent.parent.parent
json_dir = Path(__file__).parent / "json_files"
generated_dataset_dir = _project_root / "datasets" / "all_generated_mvtec"
csv_tracking_file = generated_dataset_dir / "generation_tracking.csv"


def get_base_image_dir(dataset_name: str, dataset_path: Path, category: str) -> Path:
    """Get the directory containing normal/base images for a category."""
    if dataset_name == "mvtec":
        return dataset_path / category / "train" / "good"
    elif dataset_name == "visa":
        return dataset_path / category / "Data" / "Images" / "Normal"
    elif dataset_name == "realiad":
        return dataset_path / category / "OK"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_base_images(dataset_name: str, base_dir: Path, image_extension: str) -> list:
    """Get list of base images, handling Real-IAD's subfolder structure."""
    if dataset_name == "realiad":
        # Real-IAD has subfolders S0001, S0002, etc. Collect from all subfolders
        images = []
        for subfolder in base_dir.iterdir():
            if subfolder.is_dir():
                images.extend(list(subfolder.glob(image_extension)))
        return images
    else:
        return list(base_dir.glob(image_extension))


def create_image_iterator(dataset_name: str, base_images: list, base_dir: Path):
    """Create an image iterator appropriate for the dataset."""
    if dataset_name == "realiad":
        # For Real-IAD: randomly sample from random subfolder each time
        import random

        def realiad_random_sampler():
            subfolders = [
                f for f in base_dir.iterdir() if f.is_dir() and list(f.glob("*.jpg"))
            ]
            while True:
                folder = random.choice(subfolders)
                images = list(folder.glob("*.jpg"))
                yield random.choice(images)

        return realiad_random_sampler()
    else:
        # For MVTec/Visa: cycle through images with replacement
        def looping_iterator(images):
            idx = 0
            while True:
                yield images[idx % len(images)]
                idx += 1

        return looping_iterator(base_images)


def create_prompt(object_name, defect_description):
    if defect_description.find("OVERRIDE PROMPT") != -1:
        return defect_description.replace("OVERRIDE PROMPT", "").strip()

    return (
        f"This is an intact {object_name} image. Please generate a damaged {object_name} image based on this image. "
        f"The description of the damaged {object_name} is as follows: {defect_description}"
    )
    #     f"Modify this {object_name} by adding a defect which can be described as '{defect_description}', "
    #     "limit the changes to only the defect area, the defect must be visually evident."


def review():
    ans = input("Accept generated image? \033[1;33m(Y/n)\033[0m: ")
    return ans.lower() in ["", "y", "yes"]


def load_or_create_tracking_csv():
    """Load existing tracking CSV or create a new one."""
    if csv_tracking_file.exists():
        try:
            return pd.read_csv(csv_tracking_file)
        except pd.errors.EmptyDataError:
            # File exists but is empty or has no parsable columns : recreate it
            print(
                f"Warning: tracking file '{csv_tracking_file}' is empty or invalid. Recreating it."
            )
        except Exception as e:
            # If any other read error occurs, warn and recreate the CSV
            print(
                f"Warning: failed to read tracking file '{csv_tracking_file}': {e}. Recreating it."
            )

    # Create new DataFrame with required columns
    df = pd.DataFrame(
        columns=[
            "object",
            "defect",
            "defect_synonym",
            "img_path",
            "base_img_path",
            "full_prompt",
        ]
    )
    # Ensure the directory exists
    csv_tracking_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_tracking_file, index=False)
    return df


def save_tracking_csv(df):
    """Save the tracking CSV."""
    df.to_csv(csv_tracking_file, index=False)


def get_existing_images_info(object_name, defect_name):
    """
    Get information about existing images for a specific object/defect combination.
    Returns a dict with image numbers and their full paths.
    """
    defect_dir = generated_dataset_dir / object_name / "test" / defect_name
    if not defect_dir.exists():
        return {}

    # Look for images matching the pattern: {defect_name}_{number}.png
    existing_images = {}
    for img_path in defect_dir.glob(f"{defect_name}_*.png"):
        # Extract the number from filename
        try:
            num = int(img_path.stem.split("_")[-1])
            existing_images[num] = img_path
        except (ValueError, IndexError):
            # Skip files that don't match the expected pattern
            continue

    return existing_images


def sync_csv_with_filesystem():
    """
    Synchronize the CSV with the actual filesystem.
    Remove entries from CSV if their corresponding images don't exist.
    """
    df = load_or_create_tracking_csv()
    if df.empty:
        return df

    # Check each row and keep only those where the image file exists
    valid_rows = []
    for idx, row in df.iterrows():
        img_full_path = generated_dataset_dir / row["img_path"]
        if img_full_path.exists():
            valid_rows.append(row)

    df_cleaned = pd.DataFrame(valid_rows)
    if not df_cleaned.empty:
        df_cleaned.reset_index(drop=True, inplace=True)

    save_tracking_csv(df_cleaned)
    return df_cleaned


def get_next_image_number(object_name, defect_name, existing_images):
    """
    Determine the next image number to use.
    Fill gaps if images were deleted, otherwise use max+1.
    """
    if not existing_images:
        return 0

    # Find the first gap in the sequence
    sorted_nums = sorted(existing_images.keys())
    for i in range(len(sorted_nums)):
        if i not in existing_images:
            return i

    # No gaps found, return next number after the max
    return max(sorted_nums) + 1


def add_image_to_tracking(
    object_name, defect_name, defect_synonym, img_number, prompt, base_img_path
):
    """Add a newly generated image to the tracking CSV."""
    df = load_or_create_tracking_csv()

    # Create relative path
    rel_path = f"{object_name}/test/{defect_name}/{defect_name}_{img_number:04d}.png"

    # Create new row
    new_row = pd.DataFrame(
        [
            {
                "object": object_name,
                "defect": defect_name,
                "defect_synonym": defect_synonym,
                "img_path": rel_path,
                "base_img_path": base_img_path,
                "full_prompt": prompt,
            }
        ]
    )

    df = pd.concat([df, new_row], ignore_index=True)
    save_tracking_csv(df)
    return rel_path


def generate_synonyms(client, defect, num_synonyms):
    # ask client to generate a list of synonyms or variations of the defect
    prompt = f"Generate {num_synonyms} variations or synonyms of the defect '{defect}'"
    while True:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": {"type": "array", "items": {"type": "string"}},
                },
            )
            break
        except google.genai.errors.ClientError as e:
            match = re.search(r"'retryDelay':\s*'(\d+)s'", str(e.details))
            time_to_sleep = int(match.group(1)) if match else 60
            print(
                f"Quota exceeded, sleeping for {time_to_sleep} seconds before retrying..."
            )
            for i in trange(time_to_sleep, desc="Sleeping"):
                time.sleep(1)

    response = json.loads(response.text)
    # add the defect itself to the list as the first item
    response.insert(0, defect)
    return response


def generate_image(client, base_image, prompt):
    # Spinner animation
    spinner_active = True

    def spinner():
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        while spinner_active:
            sys.stdout.write(f"\r      Generating image... {chars[idx % len(chars)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1

    spinner_thread = threading.Thread(target=spinner)
    spinner_thread.start()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            # model="gemini-3-pro-image-preview",
            contents=[prompt, base_image],
        )
        if (
            not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            print("      No valid response parts found")
            return None
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print("Response output:", part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                return image
        return None
    finally:
        spinner_active = False
        spinner_thread.join()
        sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the spinner line
        sys.stdout.flush()


def create_object_defects_synonyms(client, defects, num_synonyms=9):
    # loop through each defect and create synonyms
    synonyms_dict = {}
    for defect in defects:
        variations = generate_synonyms(client, defect, num_synonyms)
        synonyms_dict[defect] = variations

    return synonyms_dict


def image_generation_pipeline(
    synonyms_filepath: Path,
    desired_num_generation_filepath: Path,
    objects_to_include: list,
    dataset_path: Path = _project_root / "datasets" / "mvtec",
    dataset_name: str = "mvtec",
    dry_run: bool = False,
    use_unique_base_images: bool = False,
    human_review: bool = False,
):
    # Set dynamic paths based on dataset
    global generated_dataset_dir, csv_tracking_file
    generated_dataset_dir = (
        _project_root / "datasets" / DATASET_CONFIGS[dataset_name]["output_path"]
    )
    csv_tracking_file = generated_dataset_dir / "generation_tracking.csv"

    # Sync CSV with filesystem first (remove entries for deleted images)
    print("Synchronizing tracking CSV with filesystem...")
    df = sync_csv_with_filesystem()
    print(f"Tracking CSV loaded with {len(df)} existing entries")

    # load the client
    load_dotenv()
    client = genai.Client()

    synonyms_json = json.load(open(synonyms_filepath))
    desired_num_generation = json.load(open(desired_num_generation_filepath))

    for object_name in objects_to_include:
        assert object_name in synonyms_json, f"{object_name} not in synonyms_json"

        # Get base image directory and images based on dataset
        config = DATASET_CONFIGS[dataset_name]
        normal_image_dir_path = get_base_image_dir(
            dataset_name, Path(dataset_path), object_name
        )
        base_images = get_base_images(
            dataset_name, normal_image_dir_path, config["image_extension"]
        )

        # Get already used base images for this object from CSV
        used_base_images = set()
        if not df.empty:
            object_entries = df[df["object"] == object_name]
            if "base_img_path" in object_entries.columns:
                # Convert to absolute paths for comparison
                used_base_images = set(
                    (dataset_path / img).resolve()
                    for img in object_entries["base_img_path"].dropna()
                    if pd.notna(img)
                )

        print(f"\n{'=' * 60}")
        print(f"Processing {object_name}")
        print(f"  Total base images available: {len(base_images)}")
        if use_unique_base_images:
            print(f"  Already used base images: {len(used_base_images)}")

            # Remove already used images from the pool
            available_base_images = [
                img for img in base_images if img.resolve() not in used_base_images
            ]
        else:
            available_base_images = base_images
        print(
            f"  Available base images for new generation: {len(available_base_images)}"
        )

        # Calculate total images needed based on existing images
        total_images_needed = 0
        for defect, target_num in desired_num_generation[object_name].items():
            if defect == "good":
                continue
            existing_images = get_existing_images_info(object_name, defect)
            num_existing = len(existing_images)
            num_needed = target_num - num_existing
            total_images_needed += max(0, num_needed)

        print(f"  Total images needed: {total_images_needed}")

        # check we have enough available base images
        if use_unique_base_images and len(available_base_images) < total_images_needed:
            raise ValueError(
                f"Not enough available base images for {object_name}: "
                f"have {len(available_base_images)} available (out of {len(base_images)} total), "
                f"need {total_images_needed}"
            )

            # Shuffle available base images for random sampling
            sampled_images = available_base_images.copy()
            np.random.shuffle(sampled_images)
            sampled_images = sampled_images[:total_images_needed]
            image_iter = iter(sampled_images)
        elif not use_unique_base_images:
            # Create image iterator based on dataset type
            image_iter = create_image_iterator(
                dataset_name, base_images, normal_image_dir_path
            )

        for defect, num_to_generate in desired_num_generation[object_name].items():
            # avoid good which is not a defect
            if defect == "good":
                continue

            print(f"\n  Processing defect: {defect} (target: {num_to_generate} images)")

            # Get existing images info
            existing_images = get_existing_images_info(object_name, defect)
            num_existing = len(existing_images)
            num_needed = num_to_generate - num_existing

            print(f"    Existing: {num_existing}, Need to generate: {num_needed}")

            if num_needed <= 0:
                print(f"    ✓ Already have enough images for {defect}, skipping")
                continue

            # Create output directory
            defect_dir = generated_dataset_dir / object_name / "test" / defect
            defect_dir.mkdir(parents=True, exist_ok=True)

            synonyms = synonyms_json[object_name][defect]

            # Generate the needed images
            successful_generations = 0
            # Start synonym index from number of existing images to continue the cycle
            synonym_idx = num_existing

            while successful_generations < num_needed:
                base_image = None
                img_path = None
                if base_image is None:
                    try:
                        img_path = next(image_iter)
                        print(f"  Using base image: {img_path}")
                    except StopIteration:
                        print(f"    Warning: Ran out of base images for {object_name}")
                        break
                # Get the next available image number
                img_number = get_next_image_number(object_name, defect, existing_images)

                # Sample a synonym variation (cycle through them)
                variation = synonyms[synonym_idx % len(synonyms)]
                variation = variation.replace("_", " ")

                # Create the prompt
                prompt = create_prompt(object_name, variation)
                # display the prompt
                print("=" * 40)
                print(
                    f"\033[1mGenerating image {successful_generations + 1} of {num_needed} remaining for defect '{defect}' - Prompt: \033[0m"
                )
                print(f"Synonym: \033[0;32m{variation}\033[0m")
                if human_review:
                    frame(
                        prompt,
                        spacing=0,
                        min_width=60,
                        max_width=100,
                        frame_style="single",
                    )
                    a = input(
                        "Press '\033[1;42mEnter\033[0m' to continue, 'q' to quit, or 'e' to edit prompt: "
                    )
                    if a.lower() == "q":
                        print("Exiting generation loop.")
                        return
                    elif a.lower() == "e":
                        prompt = input("Enter new prompt: ")
                        print(f"Using edited prompt: {prompt}")
                # Load base image (only if we don't have one or the previous was accepted)
                if base_image is None:
                    try:
                        img_path = next(image_iter)
                    except StopIteration:
                        print(f"    Warning: Ran out of base images for {object_name}")
                        break

                    base_image = Image.open(img_path)
                    # Get relative path of base image from dataset root
                    base_img_rel_path = img_path.relative_to(dataset_path)

                # Output filename
                output_filename = f"{defect}_{img_number:04d}.png"
                output_path = defect_dir / output_filename

                # -- GENERATION --
                if dry_run:
                    print(f"    [DRY RUN] Would generate: {output_filename}")
                    ans = review() if human_review else "y"
                    if ans:
                        print(f"      ✓ [DRY RUN] Would save to {output_path}")
                        successful_generations += 1
                        synonym_idx += 1
                        # Move to next base image
                        if base_image is not None:
                            base_image.close()
                            base_image = None
                    else:
                        print(
                            "      ✗ [DRY RUN] Discarded by user - will retry with same base image"
                        )
                else:
                    print(f">>> Generating: {output_filename}")
                    print(f"      Using base image: {base_img_rel_path}")
                    try:
                        generated_image = generate_image(client, base_image, prompt)
                        # generated_image = base_image.copy() # dummy generation for testing
                    except Exception as e:
                        print(f"      Error during image generation: {e}")
                        generated_image = None

                    if generated_image is None:
                        print(
                            "! ! ! ! ! ! ! Generation failed, waiting 1 minute before retrying..."
                        )
                        [time.sleep(1) for _ in trange(60, desc="Waiting")]
                    else:
                        # Ensure generated image matches base image size for proper comparison
                        if generated_image.size != base_image.size:
                            print(
                                f"      Resizing generated image from {generated_image.size} to {base_image.size}"
                            )
                            generated_image = generated_image.resize(
                                base_image.size, Image.Resampling.LANCZOS
                            )

                        # generated_image.save("current_generated_image.png")
                        generated_image_resized = generated_image.resize(
                            (256, 256), Image.Resampling.LANCZOS
                        )
                        generated_image_resized.save("current_generated_image.png")
                        # base_image.save("current_base_image.png")
                        base_image_resized = base_image.resize(
                            (256, 256), Image.Resampling.LANCZOS
                        )
                        base_image_resized.save("current_base_image.png")
                        # save the diff as the absolute difference
                        # Convert both images to RGB to ensure they have the same mode
                        generated_image_rgb = generated_image.convert("RGB")
                        base_image_rgb = base_image.convert("RGB")
                        diff = ImageChops.difference(
                            generated_image_rgb, base_image_rgb
                        )
                        # Convert to grayscale to show intensity of differences (white = more different)
                        # diff_gray = diff.convert("L")
                        # Downsample the diff image to 256x256 for speed
                        diff_resized = diff.resize((256, 256), Image.Resampling.LANCZOS)
                        diff_resized.save("current_diff_image.png")

                        ans = review() if human_review else "y"
                        if ans:
                            # Save the generated image
                            generated_image.save(output_path)
                            # Add to tracking CSV
                            add_image_to_tracking(
                                object_name,
                                defect,
                                variation,
                                img_number,
                                prompt,
                                str(base_img_rel_path),
                            )
                            # Update existing_images dict
                            existing_images[img_number] = output_path
                            print(f"      ✓ Saved to {output_path}")
                            successful_generations += 1
                            synonym_idx += 1
                            # Move to next base image
                            base_image.close()
                            base_image = None
                        else:
                            print(
                                "      ✗ Discarded by user - will retry with same base image"
                            )
                            # save to bad generated
                            bad_generated_dir = (
                                dataset_path.parent
                                / f"bad_generated_{dataset_path.name}"
                                / object_name
                                / "test"
                                / defect
                            )
                            bad_generated_dir.mkdir(parents=True, exist_ok=True)
                            bad_output_path = bad_generated_dir / output_filename
                            generated_image.save(bad_output_path)

            # Clean up any remaining open base image
            if base_image is not None:
                base_image.close()

    print(f"\n{'=' * 60}")
    print("Image generation pipeline completed!")
    print(f"Tracking CSV saved to: {csv_tracking_file}")


if __name__ == "__main__":
    # -- ARGUMENTS --
    parser = argparse.ArgumentParser()
    h = "Generate augmented images"
    parser.add_argument("--generate_images", action="store_true", help=h)
    h = "Run the script without making any changes"
    parser.add_argument("--dry_run", action="store_true", help=h)
    h = "Dataset to generate images for"
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mvtec", "visa", "realiad"],
        default="mvtec",
        help=h,
    )
    h = "Enable human review of generated images"
    parser.add_argument("--human_review", action="store_true", help=h)
    args = parser.parse_args()

    # Load dataset configuration
    dataset_name = args.dataset
    config = DATASET_CONFIGS[dataset_name]

    # Load synonyms and num_images files
    synonyms_file = json_dir / f"{dataset_name}_final_defect_synonyms.json"
    num_images_file = json_dir / f"{dataset_name}_num_images_per_category.json"

    # Load objects from JSON
    with open(num_images_file) as f:
        objects_to_include = list(json.load(f).keys())

    # -- MAIN --
    try:
        if args.generate_images:
            image_generation_pipeline(
                synonyms_file,
                num_images_file,
                objects_to_include,
                dataset_path=config["base_path"],
                dataset_name=dataset_name,
                dry_run=args.dry_run,
                use_unique_base_images=False,
                human_review=args.human_review,
            )
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
