import os
import requests
from tqdm import tqdm


def download_file_from_google_drive(file_url, destination):
    file_id = file_url.split("/d/")[1].split("/")[0]
    url = f"https://drive.usercontent.google.com/download?export=download&confirm=t&id={file_id}"

    # Use gdown to get the final URL and headers
    session = requests.Session()
    response = session.get(url, stream=True)

    # Sometimes gdown handles confirmation pages (for large files)
    confirm_token = get_confirm_token(response)
    if confirm_token:
        url = f"{url}&confirm={confirm_token}"
        response = session.get(url, stream=True)

    # Get the total size from headers
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"File downloaded to: {destination}")


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


# Example usage
shareable_link = [
    "https://drive.google.com/file/d/162jvy2ZxodG4AGaDF1y_FLOUxoJK5tfy/view?usp=sharing",
    "https://drive.google.com/file/d/18RHw7o5UB1wSJ6QBcJfhr_bnljpcrmtd/view?usp=sharing",
]
output_file = ["data/train_truncated.csv", "data/train_truncated_transposed.csv"]

for link, file in zip(shareable_link, output_file):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(file), exist_ok=True)
    download_file_from_google_drive(link, file)
