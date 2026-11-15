"""Download the UCI Online Retail dataset."""
from pathlib import Path
import zipfile
import io
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Dataset"
DATA_DIR.mkdir(exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def download_dataset():
    """Download the official UCI workbook; use ucimlrepo as a fallback."""
    xlsx_path = DATA_DIR / "Online Retail.xlsx"
    csv_path = DATA_DIR / "Online_Retail.csv"
    if xlsx_path.exists() or csv_path.exists():
        print("Dataset already exists. Skipping download.")
        return xlsx_path if xlsx_path.exists() else csv_path

    print("Downloading Online Retail dataset from UCI...")
    try:
        response = requests.get(UCI_ZIP_URL, timeout=180)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            excel_members = [n for n in zf.namelist() if n.lower().endswith(('.xlsx', '.xls'))]
            if not excel_members:
                raise FileNotFoundError("No Excel workbook found in the downloaded UCI archive.")
            member = excel_members[0]
            xlsx_path.write_bytes(zf.read(member))
        print(f"Dataset saved to: {xlsx_path}")
        return xlsx_path
    except Exception as exc:
        print(f"Direct UCI download failed: {exc}")
        print("Trying the UCI Python import package...")
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=352)
            parts = [dataset.data.ids, dataset.data.features, dataset.data.targets]
            frames = [part for part in parts if part is not None and not part.empty]
            raw = pd.concat(frames, axis=1)
            raw.to_csv(csv_path, index=False)
            print(f"Dataset saved to: {csv_path}")
            return csv_path
        except Exception as fallback_exc:
            raise RuntimeError(
                "Could not download the dataset. Download 'Online Retail.xlsx' manually "
                "from https://archive.ics.uci.edu/dataset/352/online+retail and place it in Dataset/. "
                f"Details: {fallback_exc}"
            ) from fallback_exc


if __name__ == "__main__":
    download_dataset()
