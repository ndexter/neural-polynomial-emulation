import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")))
USER_PATH = re.compile(
    r"(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+)"
)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def test_saved_notebooks_do_not_expose_user_paths():
    offenders = []
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text())
        for value in _strings(notebook):
            match = USER_PATH.search(value)
            if match is not None:
                offenders.append(f"{path.name}: {match.group(0)}")

    assert not offenders, "saved notebooks expose user paths:\n" + "\n".join(offenders)


def test_saved_notebooks_have_no_errors_or_stderr():
    offenders = []
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text())
        for cell in notebook.get("cells", []):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    offenders.append(f"{path.name}:{cell.get('id', '<unknown>')}: error")
                if output.get("name") == "stderr":
                    offenders.append(f"{path.name}:{cell.get('id', '<unknown>')}: stderr")

    assert not offenders, "saved notebooks contain failures:\n" + "\n".join(offenders)
