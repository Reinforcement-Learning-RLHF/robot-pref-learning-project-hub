import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

from src.preferences.vlm_gemini import label_pair

result = label_pair(
    video_path_1="data/test/good_clean_slow_agentview.mp4",
    video_path_2="data/test/bad_shaky_agentview.mp4",
    api_key=api_key,
    model="gemini-2.5-flash",
)

print(result)
