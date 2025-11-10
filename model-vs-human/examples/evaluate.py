import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
# Ensure we use the local model-vs-human package, not an installed one
_script_dir = os.path.dirname(os.path.abspath(__file__))
_model_vs_human_dir = os.path.dirname(_script_dir)
if _model_vs_human_dir not in sys.path:
    sys.path.insert(0, _model_vs_human_dir)

from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template


def run_evaluation():
    models = ["resnet18_base", "resnet18_linecolor"]
    datasets = ["cue-conflict"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types =  ["shape-bias"] #c.DEFAULT_PLOT_TYPES
    plotting_def = plotting_definition_template
    figure_dirname = "shape-bias-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    run_plotting()
