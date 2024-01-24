import os
import requests

REPO_ROOT = os.path.realpath(os.path.join(os.path.split(__file__)[0], ".."))


def fetch_files():
    s3_url = "https://sdaia-kaust-public.s3.us-east-2.amazonaws.com/diabetic-retinopathy-detection/"
    flist_path = os.path.join(REPO_ROOT, "file_list.txt")
    with open(flist_path, "r") as file:
        for line in file:
            subpath = line.strip()
            url = f"{s3_url}{subpath}"
            target_path = os.path.join(REPO_ROOT, subpath)
            if os.path.exists(target_path):
                print(f"File already in place: {target_path}")
                continue
            print(f"Downloading {url}...")
            req = requests.get(url)
            if not req.ok:
                print(f"Failed to download {url}")
                continue
            print(f"Downloaded {url}")
            target_dir = os.path.split(target_path)[0]
            os.makedirs(target_dir, exist_ok=True)
            with open(target_path, "wb") as fdst:
                fdst.write(req.content)
            print(f"Saved to {target_path}")
    print("File fetching done")
    return


if __name__ == "__main__":
    fetch_files()
