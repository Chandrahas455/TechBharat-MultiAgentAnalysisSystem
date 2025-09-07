#!/usr/bin/env python
import sys
import warnings
import os
import argparse
from pathlib import Path

from datetime import datetime

from cleanup_and_analysis.crew import CleanupAndAnalysis

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run_one(csv_file_path: str, user_context: str | None = None) -> str | None:
    """Run the data processing crew on a single CSV file and return the report text if available."""
    if not os.path.exists(csv_file_path):
        print(f"Error: File '{csv_file_path}' not found.")
        return None
    if not csv_file_path.lower().endswith('.csv'):
        print(f"Error: File '{csv_file_path}' is not a CSV file.")
        return None

    print(f"Starting data processing pipeline for: {csv_file_path}")
    print("=" * 60)

    inputs = {
        'csv_file_path': csv_file_path,
        'current_year': str(datetime.now().year),
        'timestamp': datetime.now().isoformat(),
        'user_context': user_context or "",
    }

    try:
        crew = CleanupAndAnalysis().crew()
        result = crew.kickoff(inputs=inputs)

        print("\n" + "=" * 60)
        print("Data processing pipeline completed successfully!")
        print("=" * 60)

        return result if isinstance(result, str) else None
    except Exception as e:
        print(f"An error occurred while running the crew: {e}")
        raise Exception(f"An error occurred while running the crew: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Cleanup and Analysis crew on a CSV in the inputs folder.")
    parser.add_argument("--file", help="CSV filename inside the inputs folder (e.g., data.csv)")
    parser.add_argument("--context", default="", help="Short description of what the data is about (for agent context)")
    args = parser.parse_args()

    # Determine the fixed inputs folder relative to this module
    project_root = Path(__file__).resolve().parents[2]  # .../cleanup_and_analysis
    input_dir = project_root / "input"
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: inputs folder not found at {input_dir}")
        sys.exit(1)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    # Resolve target file
    if args.file:
        target_path = input_dir / args.file
        if not target_path.exists():
            print(f"Error: File not found in inputs: {target_path}")
            # Show available files
            print("Available files:")
            for p in csv_files:
                print(f" - {p.name}")
            sys.exit(1)
    else:
        # Interactive selection by number
        print("Select a CSV file to process from inputs:")
        for idx, p in enumerate(csv_files, start=1):
            print(f" {idx}. {p.name}")
        try:
            choice = int(input("Enter number: ").strip())
            if choice < 1 or choice > len(csv_files):
                raise ValueError
            target_path = csv_files[choice - 1]
        except Exception:
            print("Invalid selection.")
            sys.exit(1)

    report_text = run_one(str(target_path), user_context=args.context)
    # Save per-file report alongside the project (one level above inputs)
    base_name = target_path.stem
    report_path = project_root / f"report_{base_name}.md"
    if report_text and isinstance(report_text, str):
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Saved report: {report_path}")
    else:
        # Fallback: copy default report.md if created by task
        default_report = project_root / "report.md"
        if default_report.exists():
            with open(default_report, "r", encoding="utf-8") as f:
                content = f.read()
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Saved report: {report_path}")

    # Cleanup: remove intermediate CSV files to keep only the final transformed CSVs
    intermediate_dir = project_root / "intermediate_csv"
    try:
        if intermediate_dir.exists() and intermediate_dir.is_dir():
            for p in intermediate_dir.glob("*.csv"):
                try:
                    p.unlink()
                except Exception:
                    pass
            print(f"Cleared intermediate CSVs in: {intermediate_dir}")
    except Exception:
        # Non-fatal cleanup errors
        pass


if __name__ == "__main__":
    main()

