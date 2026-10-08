"""Bounded ownership of original CSV bytes; source files are never deleted."""

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tempfile


@dataclass(frozen=True)
class StagedSource:
    """Capability for a CSV snapshot owned by the import-preview flow."""

    owner: object
    path: Path
    size: int


def retain_source(source, max_bytes: int):
    owner = tempfile.TemporaryDirectory(prefix="analytico-source-")
    destination = Path(owner.name) / "source.csv"
    is_path = isinstance(source, (str, Path))
    handle = open(source, "rb") if is_path else source
    position = None if is_path else handle.tell()
    try:
        if not is_path:
            handle.seek(0)
        size = 0
        with destination.open("wb") as output:
            while block := handle.read(1024 * 1024):
                if isinstance(block, str):
                    block = block.encode("utf-8")
                size += len(block)
                if size > max_bytes:
                    raise ValueError("CSV exceeds the configured upload size limit.")
                output.write(block)
        # Staged uploads are immutable snapshots. Read-only mode is safe for
        # normal retained copies too, and allows a staged snapshot to be linked
        # into a dataset directory without changing permissions on link.
        destination.chmod(0o400)
        return owner, destination, size
    except Exception:
        owner.cleanup()
        raise
    finally:
        if is_path:
            handle.close()
        else:
            handle.seek(position)


def validate_staged_source(source: StagedSource, max_bytes: int) -> int:
    """Validate that a reuse capability names its own private source snapshot."""
    if not isinstance(source, StagedSource):
        raise TypeError("Staged source reuse requires an app-owned snapshot.")
    owner_name = getattr(source.owner, "name", None)
    if not owner_name:
        raise ValueError("Staged CSV source ownership is unavailable.")
    root = Path(owner_name).absolute()
    expected = root / "source.csv"
    if Path(source.path).absolute() != expected:
        raise ValueError("Staged CSV source is outside its owned directory.")
    try:
        root_stat = root.lstat()
        source_path = Path(source.path)
        source_stat = source_path.lstat()
    except OSError as error:
        raise ValueError("Staged CSV source is no longer available.") from error
    if (not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode) & 0o077
            or not stat.S_ISREG(source_stat.st_mode)):
        raise ValueError("Staged CSV source is not a regular owned file.")
    if stat.S_IMODE(source_stat.st_mode) & 0o222:
        raise ValueError("Staged CSV source must remain read-only.")
    if source_stat.st_size != source.size:
        raise ValueError("Staged CSV source size changed after preview.")
    if source.size > max_bytes:
        raise ValueError("CSV exceeds the configured upload size limit.")
    return source.size


def materialize_staged_source(source: StagedSource, destination: Path, max_bytes: int) -> bool:
    """Link an app-owned snapshot or fall back to a bounded independent copy.

    Returns True when a hardlink was created. External sources must continue to
    use the normal copy path and cannot enter through this capability.
    """
    expected_size = validate_staged_source(source, max_bytes)
    destination = Path(destination)
    try:
        os.link(source.path, destination, follow_symlinks=False)
        return True
    except (OSError, NotImplementedError, TypeError):
        # Cross-device, unsupported, or policy-restricted hardlinks retain the
        # same independent-copy behavior as before.
        pass

    total = 0
    created = False
    try:
        with source.path.open("rb") as input_file:
            with destination.open("xb") as output_file:
                created = True
                while block := input_file.read(1024 * 1024):
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError("CSV exceeds the configured upload size limit.")
                    output_file.write(block)
        if total != expected_size:
            raise ValueError("Staged CSV source size changed during copy.")
        return False
    except Exception:
        if created:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def retain_staged_source(source: StagedSource, max_bytes: int, *, prefix="analytico-source-"):
    """Create a dataset-owned source entry from an app-owned staged snapshot."""
    size = validate_staged_source(source, max_bytes)
    owner = tempfile.TemporaryDirectory(prefix=prefix)
    destination = Path(owner.name) / "source.csv"
    try:
        materialize_staged_source(source, destination, max_bytes)
        return owner, destination, size
    except Exception:
        owner.cleanup()
        raise
