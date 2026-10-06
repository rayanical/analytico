"""Bounded ownership of original CSV bytes; source files are never deleted."""

from pathlib import Path
import tempfile


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
        return owner, destination, size
    except Exception:
        owner.cleanup()
        raise
    finally:
        if is_path:
            handle.close()
        else:
            handle.seek(position)
