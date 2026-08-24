# Code Convention

- `env.viewer_text_overlay(...)` calls should stay on one line when the arguments are short, such as simple overlay label/text/loc values.
- Avoid adding temporary variables that only rename a fixed literal and reduce readability. If one concept controls multiple related keys or branches, expose one meaningful parameter such as `use_hand="Right"` and derive the obvious variants inside the helper.
- Do not make a variable just to pass one literal into one function parameter. Write the literal directly at the call site, such as `attach_seconds=2.5` or `attach_start_distance=0.2`, unless the value is reused, mutated, or shown as user-facing runtime state.
- Notebook cells should not expose routine package plumbing such as `import sys`, `from pathlib import Path`, vendor directory constants, `sys.path` edits, or direct vendor imports like `from retargeting.seq_retarget import ...`. Hide those inside the project package and expose a clear helper such as `create_dex_retargeter(...)`.
