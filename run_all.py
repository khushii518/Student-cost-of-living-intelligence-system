"""Run the whole Student Cost of Living Intelligence System end to end.
   python run_all.py   (set RAW_PATH=/path/to/actualdataset.ods if the file lives elsewhere)"""
import subprocess, sys, os
missing = []
for mod in ["pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn", "odf", "openpyxl", "tabulate"]:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit("Missing packages: " + ", ".join(missing) + "\nInstall with:  pip install -r requirements.txt")
here = os.path.dirname(os.path.abspath(__file__))
for part in ["part1_pipeline_cleaning.py", "part2_statistics_optimization.py", "part3_machine_learning.py"]:
    print(f"\n===== {part} =====")
    subprocess.run([sys.executable, os.path.join(here, part)], check=True, cwd=here)
print("\nAll done. See results/, figures/, data/, db/ and REPORT.md")
