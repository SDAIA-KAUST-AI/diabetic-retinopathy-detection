import os
import requests
import multiprocessing

REPO_ROOT = os.path.realpath(os.path.join(os.path.split(__file__)[0], ".."))


def fetch_one(subpath: str) -> None:
    s3_url = "https://sdaia-kaust-public.s3.us-east-2.amazonaws.com/diabetic-retinopathy-detection/"

    url = f"{s3_url}{subpath}"
    target_path = os.path.join(REPO_ROOT, subpath)
    if os.path.exists(target_path):
        print(f"File already in place: {target_path}")
        return
    print(f"Downloading {url}...")
    req = requests.get(url)
    if not req.ok:
        print(f"Failed to download {url}")
        return
    print(f"Downloaded {url}")
    target_dir = os.path.split(target_path)[0]
    os.makedirs(target_dir, exist_ok=True)
    with open(target_path, "wb") as fdst:
        fdst.write(req.content)
    print(f"Saved to {target_path}")


def fetch_files():
    flist_path = os.path.join(REPO_ROOT, "file_list.txt")
    subpaths = []
    with open(flist_path, "r") as file:
        for line in file:
            subpath = line.strip()
            subpaths.append(subpath)

    with multiprocessing.Pool(8) as pool:
        pool.map(fetch_one, subpaths)

    print("File fetching done")


if __name__ == "__main__":
    fetch_files()
