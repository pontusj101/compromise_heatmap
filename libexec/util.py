import logging
import pickle
import random
from pathlib import Path

from libexec.storage import GcsStorageBucket

l = logging.getLogger(__name__)


def load_samples(file_paths: list[Path], bucket: str, data_folder, log_window):
    ret = []

    for path in file_paths:
        local_path = data_folder / path

        if bucket:
            l.debug("Downloading file %s", local_path)
            GcsStorageBucket(bucket).download(path, local_path)

        with local_path.open("rb") as f:
            l.debug("Reading file %s", local_path)
            sample = pickle.load(f)

        for snapshot in sample:
            snapshot["x"] = snapshot["x"][:, : log_window + 1]

        ret.append(sample)

    return ret


def get_sample_filenames(
    log_window,
    num_validation_samples,
    num_compromised,
    num_uncompromised,
    fp_rate,
    bucket: str = None,
    data_folder=None,
):
    if bucket:
        filenames = GcsStorageBucket(bucket).list_files("training_samples")
    elif data_folder:
        filenames = list_files_relative_to_cwd(f"{data_folder}/training_samples")
    else:
        raise ValueError("either a bucket or a local data_folder must be specified")

    filenames = [
        fn
        for fn in filenames
        if int(fn.split("/")[1].split("_")[2]) >= log_window
        and float(fn.split("/")[3].split("_")[1]) == fp_rate
    ]

    uncompromised_filenames = [f for f in filenames if "passive" in f]
    compromised_filenames = [f for f in filenames if "passive" not in f]

    l.debug("Found %d uncompromised samples.", len(uncompromised_filenames))
    l.debug("Found %d compromised samples.", len(compromised_filenames))

    if len(compromised_filenames) < num_compromised:
        raise ValueError(
            f"{num_compromised} compromised samples requested, but only "
            f"{len(compromised_filenames)} found"
        )

    if len(uncompromised_filenames) < num_uncompromised:
        raise ValueError(
            f"{num_uncompromised} uncompromised samples requested, but only "
            f"{len(uncompromised_filenames)} found"
        )

    if len(uncompromised_filenames) < num_uncompromised:
        raise ValueError(
            f"{num_uncompromised} uncompromised samples requested, but only "
            f"{len(uncompromised_filenames)} found"
        )

    if (
        len(uncompromised_filenames + compromised_filenames)
        < num_compromised + num_uncompromised + num_validation_samples
    ):
        diff = (
            num_compromised
            + num_uncompromised
            + num_validation_samples
            - len(uncompromised_filenames + compromised_filenames)
        )
        raise ValueError(f"not enough samples for validation, {diff} missing")
    try:
        filtered_filenames = (
            uncompromised_filenames[:num_uncompromised]
            + compromised_filenames[: num_compromised + num_validation_samples]
        )
    except IndexError:
        raise ValueError(
            f"{num_validation_samples} requested but only "
            f"{len(compromised_filenames) - num_compromised} compromised samples found"
        )

    random.shuffle(filtered_filenames)

    filtered_filenames = [Path(path) for path in filtered_filenames]

    return filtered_filenames[:-num_validation_samples], filtered_filenames[
        -num_validation_samples:
    ]


def list_files_relative_to_cwd(directory):
    paths = []
    for file in Path(directory).rglob("*"):
        if file.is_file():
            paths.append(Path(*file.parts[1:]).as_posix())  # remove data/
    return paths
