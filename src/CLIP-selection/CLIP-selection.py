import os
import argparse
import random
import csv

import numpy as np
import torch
import open_clip
from PIL import Image


def setup_model(model_name="ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=None, seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer, device


def clip_score(image1_path, image2_path, prompt1, prompt2, model, preprocess, tokenizer, device):
    """
    Returns (probs_per_image, probs_per_image1, similarity) for two images vs two prompts.
    probs_per_image : softmax over images (dim=0), shape (2,2)
    probs_per_image1: softmax over texts  (dim=1), shape (2,2)
    """
    img1 = preprocess(Image.open(image1_path)).unsqueeze(0).to(device)
    img2 = preprocess(Image.open(image2_path)).unsqueeze(0).to(device)
    images = torch.cat([img1, img2], dim=0)

    text_tokens = tokenizer([prompt1, prompt2]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features  = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        logits = similarity * model.logit_scale.exp()
        probs_per_image  = logits.softmax(dim=0)
        probs_per_image1 = logits.softmax(dim=1)

    return probs_per_image, probs_per_image1, similarity


def is_bad_generation(image1_path, image2_path, prompt1, prompt2, model, preprocess, tokenizer, device):
    """
    Returns True if anomaly image (image1) is NOT correctly generated.
    Criteria: anomaly image is less similar to anomaly prompt than normal image.
    """
    p, p1, _ = clip_score(image1_path, image2_path, prompt1, prompt2,
                           model, preprocess, tokenizer, device)
    return (p[0][0] < p[1][0]) and (p[1][1] < p[0][1]) and (p1[0][0] < p1[0][1])


def run_csv(csv_path, anomaly_base, normal_base, output_log,
            model, preprocess, tokenizer, device):
    with open(csv_path, 'r') as f:
        data = list(csv.reader(f))

    with open(output_log, 'a') as log:
        for row in data[1:]:
            category      = row[0]
            an_image_path = row[3]
            no_image_path = row[4]
            anomaly_prompt = row[2]

            idx1 = anomaly_prompt.find(':')
            left_idx = anomaly_prompt.rfind('.', 0, idx1)
            if left_idx == -1:
                left_idx = 0
            anomaly_prompt = anomaly_prompt[left_idx:idx1].strip()

            a_path = os.path.join(anomaly_base, an_image_path)
            n_path = os.path.join(normal_base, no_image_path)

            prompt_anomaly = f"This is a damaged {category} image with {anomaly_prompt}."
            prompt_normal  = f"This is an intact {category} image without any damage."

            p, p1, _ = clip_score(a_path, n_path, prompt_anomaly, prompt_normal,
                                   model, preprocess, tokenizer, device)

            bad = (p[0][0] < p[1][0]) and (p[1][1] < p[0][1]) and (p1[0][0] < p1[0][1])
            if bad:
                log.write(
                    f"category:{category}, '{a_path}' not correctly generated. "
                    f"prompt='{anomaly_prompt}'. "
                    f"p[0][0]={p[0][0]:.4f}, p[0][1]={p[0][1]:.4f}, "
                    f"p[1][0]={p[1][0]:.4f}, p[1][1]={p[1][1]:.4f}\n"
                )
                print(f"[BAD] {os.path.basename(a_path)}")
            else:
                print(f"[OK ] {os.path.basename(a_path)}")


def main():
    parser = argparse.ArgumentParser(description="CLIP-based generation quality filter")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # single-pair mode
    pair = subparsers.add_parser("pair", help="Score a single image pair")
    pair.add_argument("image1",   help="Anomaly image path")
    pair.add_argument("image2",   help="Normal image path")
    pair.add_argument("prompt1",  help="Anomaly text prompt")
    pair.add_argument("prompt2",  help="Normal text prompt")
    pair.add_argument("--model",  default="ViT-bigG-14")
    pair.add_argument("--pretrained", default="laion2b_s39b_b160k")
    pair.add_argument("--device", default=None)

    # CSV batch mode
    batch = subparsers.add_parser("batch", help="Filter from tracking CSV")
    batch.add_argument("csv_path",      help="Path to generation_tracking.csv")
    batch.add_argument("anomaly_base",  help="Base dir for anomaly images")
    batch.add_argument("normal_base",   help="Base dir for normal images")
    batch.add_argument("output_log",    help="Path to write bad-generation log")
    batch.add_argument("--model",       default="ViT-bigG-14")
    batch.add_argument("--pretrained",  default="laion2b_s39b_b160k")
    batch.add_argument("--device",      default=None)

    args = parser.parse_args()

    model, preprocess, tokenizer, device = setup_model(
        model_name=args.model, pretrained=args.pretrained, device=args.device
    )

    if args.mode == "pair":
        p, p1, sim = clip_score(args.image1, args.image2, args.prompt1, args.prompt2, model, preprocess, tokenizer, device)
        print(f"probs_per_image (softmax over images):\n{p.cpu().numpy()}")
        print(f"probs_per_image1 (softmax over texts):\n{p1.cpu().numpy()}")
        bad = (p[0][0] < p[1][0]) and (p[1][1] < p[0][1]) and (p1[0][0] < p1[0][1])
        print("Result: BAD generation" if bad else "Result: OK")

    elif args.mode == "batch":
        run_csv(args.csv_path, args.anomaly_base, args.normal_base, args.output_log,
                model, preprocess, tokenizer, device)


if __name__ == "__main__":
    main()
