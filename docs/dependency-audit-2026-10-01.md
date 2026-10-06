# Dependency Audit — 2026-10-01

## Scope and method

Audited the frontend lockfile with `npm audit` for the full dependency tree and production-only dependencies. Audited the installed backend virtual environment and freshly resolved `backend/requirements.txt` / `backend/requirements-dev.txt` with `pip-audit` 2.10.1. The installed environment uses Python 3.14.0. The Python requirements are ranges, so the fresh-resolution audit checks the versions a current install selects; the installed-environment audit also caught vulnerable versions already present in the local venv.

## Findings and changes

| Scope | Before | After |
| --- | --- | --- |
| Frontend production dependencies | 11 findings: 2 critical, 5 high, 4 moderate | 0 |
| Frontend development dependencies | 36 findings: 24 high, 11 moderate, 1 low | 0 (the full-tree audit is clear) |
| Frontend full tree | 47 findings: 2 critical, 29 high, 15 moderate, 1 low | 0 |
| Installed backend venv | 39 advisory records across 7 packages; 29 records across six application libraries and 10 in the venv's `pip` tool | 0 across 36 installed distributions |
| Fresh backend runtime requirements | No known findings in the initial latest-version resolution | 0 across 30 resolved distributions |
| Fresh backend development requirements | No known findings in the initial latest-version resolution | 0 across 32 resolved distributions |

`npm audit` found the frontend's critical runtime findings in Next.js and jsPDF, plus axios advisories. Next.js moved from 16.1.3 to 16.3.8, with `eslint-config-next` kept in step at 16.3.8. This stays on Next.js 16; 16.3.8 is the current Active LTS security release and includes the September 30 fixes, following the September 22 critical upstream fix in 16.3.6. [September release](https://nextjs.org/blog/september-2026-security-release), [September 22 release](https://nextjs.org/blog/nextjs-security-update-september-22-2026).

Axios moved from 1.13.2 to 1.20.0 within major version 1, which is outside the vulnerable ranges reported for the installed version. [Axios 1.20.0 release](https://github.com/axios/axios/releases/tag/v1.20.0), [Axios security advisories](https://github.com/axios/axios/security/advisories).

jsPDF moved from 3.0.4 to 4.2.1. This is an explicit major-version change because the path-traversal advisory affects versions through 3.0.4 and the current jsPDF advisories are fixed in 4.2.1. [Path-traversal advisory](https://github.com/parallax/jsPDF/security/advisories/GHSA-f8cm-6447-x5h2), [4.2.1 security release](https://github.com/parallax/jsPDF/releases/tag/v4.2.1), [HTML-injection advisory](https://github.com/parallax/jsPDF/security/advisories/GHSA-wfv2-pwc8-crg5).

`npm audit fix` updated compatible transitive dependencies in the lockfile without `--force` (56 changed, four added, one removed). The remaining direct manifest changes are the version floors above.

The initial installed backend venv contained vulnerable AnyIO 4.12.1, Click 8.3.1, idna 3.11, python-dotenv 1.2.1, python-multipart 0.0.21, Starlette 0.50.0, and pip 25.3. The initial fresh requirements resolution was already selecting safe recent versions, which is why the installed-vs-resolved audit produced different results. The requirements now enforce the relevant fixes on fresh installs:

- `fastapi>=0.135.0` and `starlette>=1.3.1`; FastAPI 0.135.0 permits Starlette 1.x, and 1.3.1 fixes the latest audited Starlette advisory. [FastAPI 0.135.0 metadata](https://pypi.org/pypi/fastapi/0.135.0/json), [Starlette advisory](https://github.com/Kludex/starlette/security/advisories/GHSA-82w8-qh3p-5jfq).
- `python-multipart>=0.0.31` and `python-dotenv>=1.2.2`. [python-multipart advisory](https://github.com/Kludex/python-multipart/security/advisories/GHSA-v9pg-7xvm-68hf), [python-dotenv advisory](https://github.com/theskumar/python-dotenv/security/advisories/GHSA-mf9w-mj56-hr94).
- `anyio>=4.14.2`, `click>=8.3.3`, and `idna>=3.15` for the patched transitive versions reported by pip-audit. [AnyIO advisory](https://github.com/agronholm/anyio/security/advisories/GHSA-5p39-cfhj-2xmp), [Click 8.3.3 release](https://github.com/pallets/click/releases/tag/8.3.3), [idna advisory](https://github.com/kjd/idna/security/advisories/GHSA-65pc-fj4g-8rjx).

The refreshed venv resolved FastAPI 0.142.2, Starlette 1.7.0, python-multipart 0.0.32, python-dotenv 1.2.4, AnyIO 4.15.1, Click 8.5.0, idna 3.20, Uvicorn 0.40.0, pandas 2.3.3, OpenAI 2.15.0, Pydantic 2.12.5, and HTTPX 0.28.1. Its local `pip` was upgraded to 26.2.1; pip is environment tooling and was not added to application requirements.

PyArrow was removed from `backend/requirements.txt` after a repository search found no direct use. Installing it activated a pandas CSV inference path that stripped leading zeroes from string values; removing the unused package avoids adding that optional engine to fresh installs. The backend now uses pandas' C CSV parser for the affected path.

## Validation

- `npm audit`: 0 findings in the full tree and 0 in production dependencies.
- `npm run lint`, `npm run test:helpers`, and `npm run build`: passed with Next.js 16.3.8.
- jsPDF 4.2.1 API smoke: generated a valid two-page PDF buffer using text, PNG image, page addition, and `output('arraybuffer')`.
- Browser export: the updated app loaded without browser errors and exported a one-page A4 PDF (63,401 bytes), which was inspected with PDF metadata and a rendered PNG. The export still shows layout issues: the bottom x-axis label is clipped and widget controls/scrollbars remain visible. Treat visual export fidelity as follow-up work, not as verified by the package compatibility check.
- `pip-audit` reported no known findings for the installed venv, the runtime requirements, or the runtime-plus-development requirements. `pip check` found no broken requirements.
- The backend suite passed 53 tests on the refreshed environment.

## Residual constraints

At the audit checkpoint, Python requirements used minimum-version ranges. The subsequent [dependency-locking follow-up](dependency-locking.md) replaces install requirements with exact, hashed runtime and development locks; editable ranges now live in `.in` files. The jsPDF major bump passed the build, API smoke, and browser export checks; the browser export layout issues listed above remain unresolved.
