from pathlib import Path

MODEL_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = MODEL_DIRECTORY / "data"


def reference_directory():
    return str(MODEL_DIRECTORY)


def data_path(name):
    return DATA_DIRECTORY / name
