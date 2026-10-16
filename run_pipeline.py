import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ['01_reproduce_original.py', '02_validation_strategies.py', '03_feature_analysis_grouped.py', '04_pca_integrated.py', '05_figures.py', '06_remaining_figures.py', '07_improved_models.py', '08_nested_tuning.py', '09_extended_zoo.py', '10_nested_selection.py', '11_arsenal_sweep.py', '12_hybrid_augmentation.py', '13_structured_statistics.py', '14_nested_master.py', '15_predictor_summaries.py', '16_selection_ladder.py', '17_final_diagnostics.py', '18_figures_arsenal.py', '19_data_audit.py', '20_within_season_rank.py', '21_same_configuration.py', '22_cells_vs_plants.py', '23_repeated_sequences.py']

env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
os.makedirs(os.path.join(HERE, "results", "logs"), exist_ok=True)
for name in ORDER:
    log = os.path.join(HERE, "results", "logs", name.replace(".py", ".log"))
    print("running", name, flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call([sys.executable, os.path.join(HERE, name)], cwd=HERE,
                             env=env, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        sys.exit("%s failed (exit code %d); see %s" % (name, rc, log))
print("all scripts finished")
