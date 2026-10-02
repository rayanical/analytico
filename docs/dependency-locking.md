# Dependency locking

Backend installations use `backend/requirements.txt` or `backend/requirements-dev.txt`. Both contain exact direct and transitive versions and SHA-256 distribution hashes. Editable dependency intent and audited minimum versions live in the corresponding `.in` files. The development lock is constrained by the runtime lock so the two cannot silently select different runtime versions.

The initial locks preserve the previously tested backend versions. Universal resolution includes platform markers, including Windows-only Colorama and excluding Uvloop on Windows/PyPy. The resolver targets Python 3.11 and newer. Validation covers Python 3.14 on macOS; universal resolution is not evidence of execution on every platform or future Python version.

Install into a fresh virtual environment:

```sh
cd backend
python -m venv venv
source venv/bin/activate
python -m pip install --require-hashes --only-binary=:all: -r requirements.txt
# For development, use requirements-dev.txt instead.
python -m pip check
```

On Windows, activate with `venv\Scripts\activate`. Binary-only installation avoids executing unpinned source-build dependencies. If a platform has no matching wheel, installation fails; review that platform explicitly rather than disabling hashes or accepting a source build without review. Use a new environment when verifying a release: `pip install` does not remove unrelated packages already installed.

The frontend already pins its resolved graph in `frontend/package-lock.json`; install with `npm ci`.

## Controlled updates

Locks were generated with uv 0.10.7. uv is maintenance tooling, not an application dependency. From the repository root, edit the `.in` sources and deliberately refresh both locks:

```sh
uv pip compile backend/requirements.in --universal --python-version 3.11 --generate-hashes --upgrade -o backend/requirements.txt
uv pip compile backend/requirements-dev.in -c backend/requirements.txt --universal --python-version 3.11 --generate-hashes --upgrade -o backend/requirements-dev.txt
```

Omit `--upgrade` to prefer existing pins. Commit both source and generated changes. Then install into a fresh environment with hashes, run `pip check`, audit both locks, and run the backend regression and interpretation benchmark suites. Check platforms intended for distribution before claiming support.

Hashes verify downloaded artifacts; they do not prove package safety. Locks freeze versions until the next deliberate security/compatibility update. The Python interpreter, pip and OS libraries are outside these application locks.

## Verification on October 1, 2026

A fresh Python 3.14 macOS environment installed the development lock with `--require-hashes --only-binary=:all:`. `pip check` passed, 53 backend tests and six benchmark tests passed, and both application locks returned no known vulnerability findings with pip-audit. Each lock has 36 exact package pins, including the Windows-only dependency; all pins applicable to the test machine match the previously validated environment. Runtime installation was separately verified in another fresh environment.

## Disk ingestion addition on October 2, 2026

Added exact `duckdb==1.5.6` to the runtime intent and regenerated both universal hash locks without upgrading the existing pins. A fresh Python 3.14 macOS environment installed the development lock with `--require-hashes --only-binary=:all:`; `uv pip check` reported all 36 installed packages compatible. Both runtime and development locks returned no known vulnerability findings with pip-audit. This does not establish executable/wheel compatibility on untested operating systems.
