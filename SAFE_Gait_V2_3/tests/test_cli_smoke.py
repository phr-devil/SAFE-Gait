from pathlib import Path
import json
import subprocess
import sys


def test_synthetic_demo_cli_runs_end_to_end(tmp_path):
    root=Path(__file__).resolve().parents[1]
    proc=subprocess.run(
        [sys.executable, str(root/'scripts'/'run_synthetic_demo.py')],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert 'Saved:' in proc.stdout
    report=json.loads((root/'reports'/'synthetic_demo.json').read_text())
    assert set(report['pairs'])=={'SP->SP','HP->HP','SP->HP','HP->SP'}
