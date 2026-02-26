import json
import subprocess
import sys
import shutil
from pathlib import Path

def get_coverage(report_path):
    if not Path(report_path).exists():
        return None
    with open(report_path) as f:
        data = json.load(f)
    # Support both float and int coverage totals
    return data["totals"]["percent_covered"]

def run_command(cmd, cwd=None):
    """Helper to run shell commands."""
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, cwd=cwd)

def generate_baseline(output_path: Path):
    """Uses git worktree to generate coverage for the main branch."""
    # Create temp dir inside current project to ensure parent-path relative logic works
    project_root = Path.cwd()
    temp_dir = project_root / ".baseline_temp"
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print("--- Generating baseline coverage from 'main' ---")
    try:
        # 1. Create a temporary worktree of the main branch
        run_command(f"git worktree add {temp_dir} main")

        # 2. Run tests. We point specifically to the absolute path of our output
        print("Running tests on main branch...")
        run_command("pdm install -dG test", cwd=temp_dir)
        # We output to a temp file first then move it to be safe
        run_command(f"pdm run pytest --cov=inference_perf --cov-report=json:{output_path.absolute()} tests/", cwd=temp_dir)
        
        print(f"✅ Baseline generated: {output_path.name}")
    finally:
        # 3. Cleanup: remove the worktree folder and the git reference
        run_command(f"git worktree remove {temp_dir} --force")

def main():
    parser = argparse.ArgumentParser(description="Check for coverage regression.")
    parser.add_argument("--force", action="store_true", help="Force regeneration of coverage reports")
    
    args = parser.parse_args()

    current_report = Path("coverage.json")
    baseline_report = Path("coverage_main.json")

    # If baseline doesn't exist, try to generate it
    if "--generate-baseline" in sys.argv or not baseline_report.exists():
        try:
            generate_baseline(baseline_report)
        except Exception as e:
            print(f"❌ Error generating baseline: {e}")
            sys.exit(1)

    current_val = get_coverage(current_report)
    baseline_val = get_coverage(baseline_report)

    # Safeguard against missing data
    if current_val is None or baseline_val is None:
        print("❌ Error: Could not read coverage values.")
        print(f"Current Report Exists: {current_report.exists()}")
        print(f"Baseline Report Exists: {baseline_report.exists()}")
        sys.exit(1)

    print("\n--- Coverage Results ---")
    print(f"Main Branch:    {baseline_val:.2f}%")
    print(f"Current Branch: {current_val:.2f}%")

    # Use a small epsilon (0.01) to handle floating point precision issues
    if current_val < (baseline_val - 0.01):
        diff = baseline_val - current_val
        print(f"❌ FAIL: Coverage decreased by {diff:.2f}%")
        sys.exit(1)
    
    print("✅ PASS: Coverage is maintained or improved.")

if __name__ == "__main__":
    main()