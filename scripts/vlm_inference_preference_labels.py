from google import genai
import time
import argparse
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# This script uses the Google Gemini API to generate preference labels between two robot videos.
# Put videos in vlm-pref/demos
# SPECIFICATIONS:
# Run the script in the vlm-pref folder
# If you want a default API key, please create a .env file in the vlm-pref/scripts folder and add your key as GOOGLE_API_KEY=your_api_key


class PreferenceLabel(BaseModel):
    score: float = Field(description="Score for Video 1 relative to Video 2: 0 if Video 1 is worse, 0.5 if equally good, 1 if Video 1 is better")
    comment: str = Field(description="Brief qualitative explanation of the preference decision")


def wait_for_active(client, video_file):
    while video_file.state.name != "ACTIVE":
        print(f"Waiting for {video_file.name} to become ACTIVE...")
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    return video_file


def main(api_key):
    parser = argparse.ArgumentParser(description="Google AI preference label inference between two videos")

    # Add args
    parser.add_argument("--api_key", type=str, default=api_key,
                        help="put your api key")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="which model to use (e.g. gemini-2.5-flash)")
    parser.add_argument("--video_path_1", type=str, default="demos/bad_shaky_agentview.mp4",
                        help="path to the first video")
    parser.add_argument("--video_path_2", type=str, default="demos/good_clean_medium_agentview.mp4",
                        help="path to the second video")
    parser.add_argument("--prompt", type=str, default=None,
                        help="text prompt to help the vlm (if unspecified, will use prompt in the txt file)")
    args = parser.parse_args()

    # Upload files using google-genai client
    upload_client = genai.Client(api_key=args.api_key)
    print("Uploading videos...")
    video_file_1 = upload_client.files.upload(file=args.video_path_1)
    video_file_2 = upload_client.files.upload(file=args.video_path_2)

    prompt = args.prompt
    if prompt is None:
        with open("scripts/vlm_prompt_preference_labels.txt", 'r') as f:
            prompt = f.read()

    video_file_1 = wait_for_active(upload_client, video_file_1)
    video_file_2 = wait_for_active(upload_client, video_file_2)

    # Use langchain for structured output
    llm = ChatGoogleGenerativeAI(model=args.model, google_api_key=args.api_key)
    structured_llm = llm.with_structured_output(PreferenceLabel)

    def query(first, second):
        message = HumanMessage(content=[
            {"type": "text", "text": "Video 1:"},
            {"type": "media", "mime_type": "video/mp4", "file_uri": first.uri},
            {"type": "text", "text": "Video 2:"},
            {"type": "media", "mime_type": "video/mp4", "file_uri": second.uri},
            {"type": "text", "text": prompt},
        ])
        return structured_llm.invoke([message])

    # Run both orderings to mitigate position bias
    result_ab = query(video_file_1, video_file_2)
    result_ba = query(video_file_2, video_file_1)

    # Combine: score_ab is "how good is video 1", score_ba is "how good is video 2"
    # so we flip score_ba to get it back in terms of video 1
    combined_score = (result_ab.score + (1 - result_ba.score)) / 2

    print({
        "score": combined_score,
        "score_ab": result_ab.score,
        "score_ba_flipped": 1 - result_ba.score,
        "comment_ab": result_ab.comment,
        "comment_ba": result_ba.comment,
    })


if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    main(api_key)
