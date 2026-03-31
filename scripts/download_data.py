import argparse
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

CHUNK_SIZE = 8192


def download_with_progress(url: str, output_path: Path):
    """Downloads a file from url to output_path with a progress bar."""
    with urllib.request.urlopen(url) as response:
        total_size = int(response.getheader("Content-Length", 0))

        with (
            open(output_path, "wb") as f,
            tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar,
        ):
            while chunk := response.read(CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_with_progress(archive_path: Path, dest_dir: Path):
    """Extracts .tgz archive with a progress bar."""
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, unit="file", desc=f"Extracting {archive_path.name}"):
            tar.extract(member, path=dest_dir, filter="data")


def download_data(url: str, dir_name: str = "data"):
    """
    Downloads and extracts the dataset.
    Skips if the dataset is already downloaded.
    """
    dir_path = Path(dir_name)
    archive_name = url.split("/")[-1]
    archive_path = dir_path / archive_name
    data_name = archive_name.split(".")[-2]

    if (dir_path / data_name).exists():
        print(f"Data already exists in '{dir_path}'. Skipping.")
        return

    dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data from: {url} ...")
    try:
        download_with_progress(url, archive_path)
        print("Extracting archive...")
        extract_with_progress(archive_path, dir_path)
    except Exception as e:
        print(f"Error: {e}")
        return
    finally:
        if archive_path.exists():
            archive_path.unlink()
            print("Archive removed.")

    print("Succesfully downloaded data!")


def main():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--full", action="store_true", help="Download full dataset (2.9GB)")
    args = parser.parse_args()

    if args.full:
        url = "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
    else:
        url = "https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz"

    download_data(url=url)


if __name__ == "__main__":
    main()
