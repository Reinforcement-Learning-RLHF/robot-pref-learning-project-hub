from transformers import AutoProcessor
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Google AI inference")

    parser.add_argument("--family", type=str, default="qwen3",
                        help="which model family to use (e.g. qwen3)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="which model to use (e.g. Qwen/Qwen2-VL-2B)")
    parser.add_argument("--video_paths", type=list, default=["demos/bad_shaky_agentview.mp4", "demos/good_clean_slow_agentview.mp4"],
                        help="where your video(s) at")
    parser.add_argument("--prompt", type=str, default=None,
                        help="text prompt to help the vlm (if unspecified, will use prompt in the txt file)")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu or cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="dtype to load model parameters in")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for deterministic generation")
    args = parser.parse_args()

    if args.seed is not None:
        import numpy as np
        import random
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.family == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration as VLM
    elif args.family == "qwen2":
        from transformers import Qwen2VLForConditionalGeneration as VLM
        from qwen_vl_utils import process_vision_info
    else:
        raise ValueError(f"Unsupported model family: {args.family}")

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = VLM.from_pretrained(
        args.model, dtype=args.dtype, device_map=args.device
    )
    model = model.to(args.device).eval()

    # load processer
    processor = AutoProcessor.from_pretrained(args.model)

    # build prompt
    prompts_dir = os.path.abspath(os.path.join(__file__, "..", "..", "prompts"))

    videos = args.video_paths
    system_prompt = args.prompt
    user_prompt = []
    for i in range(len(videos)):
        user_prompt.append({"type": "text", "text": f"Video {i+1}"})
        user_prompt.append({"type": "video", "video": videos[i]})

    if len(videos) == 0:
        raise ValueError("No videos provided")
    elif len(videos) == 1:
        # Scoring mode
        if system_prompt is None:
            with open(os.path.join(prompts_dir, "system_score.txt"), 'r') as f:
                system_prompt = f.read()
    elif len(videos) == 2:
        # Preference mode
        if system_prompt is None:
            with open(os.path.join(prompts_dir, "system_pref.txt"), 'r') as f:
                system_prompt = f.read()
    else:
        # Ranking mode
        if system_prompt is None:
            with open(os.path.join(prompts_dir, "system_rank.txt"), 'r') as f:
                system_prompt = f.read()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_prompt}
    ]

    # chat template and tokenize
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt'
    )
    inputs = inputs.to(model.device)

    # generate output
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

if __name__ == '__main__':
    main()
