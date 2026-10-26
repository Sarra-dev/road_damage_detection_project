import os
os.environ["HF_HUB_DISABLE_XET"] = "1"   

import time
from huggingface_hub import snapshot_download

COUNTRIES = ["Czech", "India"]         

patterns = ["data/data.yaml"]
for c in COUNTRIES:
    patterns += [
        f"data/images/*/*/{c}_*.jpg",
        f"data/labels/*/*/{c}_*.txt",
    ]

for attempt in range(1, 11):
    try:
        path = snapshot_download(
            repo_id="dronefreak/RDD2022",
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir="data/raw/rdd2022",
            max_workers=8,
        )
        print("Done:", path)
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        time.sleep(10)
else:
    print("Gave up after 10 attempts. Run the script again to resume.")