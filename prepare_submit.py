"""Create the PART-I submission archive required by the assignment."""

from pathlib import Path
import zipfile


OUTPUT_PATH = Path("nlp2026-final-outputs.zip")

REQUIRED_FILES = [
    Path("modules/attention.py"),
    Path("modules/gpt2_layer.py"),
    Path("models/base_gpt.py"),
    Path("models/gpt2.py"),
    Path("classifier.py"),
    Path("optimizer.py"),
    Path("config.py"),
    Path("datasets.py"),
    Path("evaluation.py"),
    Path("utils.py"),
    Path("sanity_check.py"),
    Path("optimizer_test.py"),
    Path("optimizer_test.npy"),
    Path("prepare_submit.py"),
    Path("predictions/last-linear-layer-sst-dev-out.csv"),
    Path("predictions/last-linear-layer-sst-test-out.csv"),
    Path("predictions/full-model-sst-dev-out.csv"),
    Path("predictions/full-model-sst-test-out.csv"),
    Path("predictions/last-linear-layer-cfimdb-dev-out.csv"),
    Path("predictions/last-linear-layer-cfimdb-test-out.csv"),
    Path("predictions/full-model-cfimdb-dev-out.csv"),
    Path("predictions/full-model-cfimdb-test-out.csv"),
]


def main():
    missing = [str(path) for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot create submission archive; missing files:\n- "
            + "\n- ".join(missing)
        )

    with zipfile.ZipFile(
        OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in REQUIRED_FILES:
            archive.write(path, path.as_posix())

    print(f"Submission zip file created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
