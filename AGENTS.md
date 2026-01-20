# Working Style
- Always run `cargo test` after changes unless explicitly told not to.
- Run `cargo test` and `cargo bench` before committing/pushing unless changes are comment-only.
- Expect benchmarks to be run after non-structural or important structural changes.
- Keep docs short, clear, and include minimal math when it helps.
- Prefer deterministic, algorithmically generated data in tests and benches.
- Use table/logging helpers for readable iteration output.
