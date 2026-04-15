from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Streamlit dashboard.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    _ = load_config(args.config)

    dashboard_script = Path(__file__).resolve().parents[1] / "src" / "suturing_pipeline" / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_script)]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
