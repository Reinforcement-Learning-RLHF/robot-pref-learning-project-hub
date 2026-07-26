"""
VLM preference labeling using Google Gemini API.

Importable interface:
    label_pair(video_path_1, video_path_2, api_key, model, prompt, prompt_path) -> dict
    generate_vlm_pairs(registry_csv, output_csv, api_key, model, prompt_path, max_pairs) -> int
"""

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

FIELDNAMES = [
    "traj_A", "traj_B", "label", "evaluator_type", "timestamp", "model_version", "prompt_version",
    "score_ab", "score_ba_flipped", "comment_ab", "comment_ba", "consistent",
]
PROMPT_VERSION = "gemini_pref_v1"


class PreferenceLabel(BaseModel):
    score: float = Field(
        description=(
            "A number from 0.0 to 1.0 indicating which video performed better. "
            "Use 1.0 if Video 1 is better. "
            "Use 0.0 if Video 2 is better. "
            "Use 0.5 if they are equally good."
        )
    )
    comment: str = Field(description="Brief qualitative explanation of the preference decision")


def _wait_for_active(client, video_file):
    while video_file.state.name != "ACTIVE":
        print(f"  Waiting for {video_file.name} to become ACTIVE...")
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    return video_file


def label_pair(
    video_path_1: str,
    video_path_2: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    prompt: str | None = None,
    prompt_path: str | None = None,
) -> dict:
    """
    Label one (A, B) video pair using Gemini. Runs both orderings to reduce position bias.

    Returns dict: score, score_ab, score_ba_flipped, comment_ab, comment_ba.
    score=1 → A is better; score=0 → B is better; score=0.5 → tie.
    """
    if prompt is None:
        path = Path(prompt_path) if prompt_path else PROMPTS_DIR / "gemini_pref.txt"
        with open(path) as f:
            prompt = f.read()

    upload_client = genai.Client(api_key=api_key)
    print(f"  Uploading {video_path_1} ...")
    vf1 = upload_client.files.upload(file=video_path_1)
    print(f"  Uploading {video_path_2} ...")
    vf2 = upload_client.files.upload(file=video_path_2)

    vf1 = _wait_for_active(upload_client, vf1)
    vf2 = _wait_for_active(upload_client, vf2)

    def _query(first, second) -> PreferenceLabel:
        response = upload_client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_text(text="Video 1:"),
                types.Part.from_uri(file_uri=first.uri, mime_type="video/mp4"),
                types.Part.from_text(text="Video 2:"),
                types.Part.from_uri(file_uri=second.uri, mime_type="video/mp4"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PreferenceLabel,
            ),
        )
        return response.parsed

    result_ab = _query(vf1, vf2)
    result_ba = _query(vf2, vf1)

    score_ba_flipped = 1 - result_ba.score
    combined_score = (result_ab.score + score_ba_flipped) / 2

    # Consistency check: both orderings must agree on the winner AND have similar confidence.
    # A split like AB=0.40 / BA_flipped=0.00 is flagged inconsistent even though both are < 0.5,
    # because the 0.40 spread means one ordering was uncertain while the other was decisive.
    same_direction = (result_ab.score > 0.5) == (score_ba_flipped > 0.5)
    close_scores = abs(result_ab.score - score_ba_flipped) <= 0.25
    consistent = same_direction and close_scores

    print("\n--- label_pair scores ---")
    print(f"  AB ordering  (video_1 shown as Video 1): raw score = {result_ab.score}")
    print(f"    -> {'video_1 wins' if result_ab.score > 0.5 else 'video_2 wins' if result_ab.score < 0.5 else 'tie'}")
    print(f"  BA ordering  (video_2 shown as Video 1): raw score = {result_ba.score}")
    print(f"  score_ba_flipped = 1 - {result_ba.score} = {score_ba_flipped}")
    print(f"    -> {'video_1 wins' if score_ba_flipped > 0.5 else 'video_2 wins' if score_ba_flipped < 0.5 else 'tie'}")
    print(f"  combined = ({result_ab.score} + {score_ba_flipped}) / 2 = {combined_score}")
    if consistent:
        print(f"  [OK] Both orderings agree.")
    else:
        print(f"  [WARN] Orderings inconsistent (spread={abs(result_ab.score - score_ba_flipped):.2f} > 0.25).")
    print(f"  comment_ab: {result_ab.comment}")
    print(f"  comment_ba: {result_ba.comment}")
    print("-------------------------\n")

    return {
        "score": combined_score,
        "score_ab": result_ab.score,
        "score_ba_flipped": score_ba_flipped,
        "comment_ab": result_ab.comment,
        "comment_ba": result_ba.comment,
        "consistent": consistent,
    }


def generate_vlm_pairs(
    registry_csv: str,
    output_csv: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    prompt_path: str | None = None,
    max_pairs: int | None = None,
) -> int:
    """
    Read rollout registry, generate pairwise Gemini preference labels, write to output_csv.

    Pairs successes against failures. Returns number of pairs labeled.
    """
    registry_csv = Path(registry_csv)
    output_csv = Path(output_csv)

    if prompt_path is None:
        prompt_path = str(PROMPTS_DIR / "gemini_pref.txt")

    with open(prompt_path) as f:
        prompt = f.read()

    with open(registry_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    successes = [r for r in rows if r.get("success_label", "").lower() == "true"]
    failures = [r for r in rows if r.get("success_label", "").lower() == "false"]
    pairs = [(s, fail) for s in successes for fail in failures]

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        session_start = time.time()

        for i, (a, b) in enumerate(pairs):
            started_at = datetime.now().strftime("%H:%M:%S")
            print(f"Labeling pair {i + 1}/{len(pairs)}: {a['rollout_id']} vs {b['rollout_id']}  [started {started_at}]")
            pair_start = time.time()

            result = label_pair(a["video_path"], b["video_path"], api_key, model, prompt=prompt)
            labeled_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            pair_elapsed = time.time() - pair_start
            total_elapsed = time.time() - session_start
            print(f"  Pair done in {pair_elapsed:.1f}s  |  total elapsed {total_elapsed:.1f}s")

            writer.writerow({
                "traj_A": a["rollout_id"],
                "traj_B": b["rollout_id"],
                "label": result["score"],
                "evaluator_type": "vlm_gemini",
                "timestamp": labeled_at,
                "model_version": model,
                "prompt_version": PROMPT_VERSION,
                "score_ab": result["score_ab"],
                "score_ba_flipped": result["score_ba_flipped"],
                "comment_ab": result["comment_ab"],
                "comment_ba": result["comment_ba"],
                "consistent": result["consistent"],
            })

    total_secs = time.time() - session_start
    total_mins, secs = divmod(int(total_secs), 60)
    avg = total_secs / len(pairs) if pairs else 0
    print(f"\nFinished {len(pairs)} pairs in {total_mins}m {secs:02d}s  |  avg {avg:.1f}s per pair")
    print(f"Wrote {len(pairs)} labeled pairs to {output_csv}")
    return len(pairs)
