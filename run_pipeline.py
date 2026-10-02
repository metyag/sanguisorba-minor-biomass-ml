# -*- coding: utf-8 -*-
"""Run the analysis scripts in order; stop at the first failure."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ['01_reproduce_original.py', '02_validation_strategies.py', '03_feature_analysis_grouped.py', '04_pca_integrated.py', '05_figures.py', '05b_remaining_figures.py', '05c_pdp_summary.py', '09_improved_models.py', '10_nested_tuning.py', '13_extended_zoo.py', '14_nested_selection.py', '18_arsenal_sweep.py', '19_hybrid_augmentation.py', '20_structured_statistics.py', '21_nested_master.py', '22_predictor_summaries.py', '23_selection_ladder.py', '24_final_diagnostics.py', '25_figures_arsenal.py', '26_data_audit.py', '29_within_season_rank.py', '129_same_configuration.py', '130_cells_vs_plants.py', '131_repeated_sequences.py']

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
