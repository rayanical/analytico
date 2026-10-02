# Consolidated automatic import and optional review

Normal CSV uploads and demos now stage, validate and confirm automatically.
Charting is available as soon as preparation completes; no review opens itself.
The single **Review data** action combines file syntax/locale/null settings,
cleaning and missing-value details, editable column interpretations, optional AI
proposals and the current parsed sample. The separate cleaning drawer was removed.

If a file cannot be safely imported, the retained staged source remains available
through **Review data** to adjust settings, recheck and explicitly apply them.
Closing that recovery view retains the source; Cancel import discards it. Invalid
records are not silently skipped merely to achieve an automatic import.

Post-import edits rebuild from retained source bytes and atomically publish a new
version. Failed edits leave the active dataset intact. Delimiter/encoding edits
reset prior field overrides and must be applied separately from column edits,
because the resulting field identities can change. Existing chart/history version
invalidation remains in place. AI proposals still require acceptance and applying
changes before affecting data.

Validation: 207 backend tests, frontend lint/type checks, helper checks and the
webpack production build passed. Browser checks verified a demo imports and charts
without a modal, optional review opens, and a parsing-setting edit applies. HTTP
checks verified locale-file automatic confirmation and malformed-file staged
recovery. The browser file chooser did not open reliably in the automation, so
the malformed-file browser recovery click path was not verified end to end.

The benchmark-only native loader experiment is separate from this production UI
change. See [current ingestion patterns](ingestion-patterns-research-2026-10-02.md).
