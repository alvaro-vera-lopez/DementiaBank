import argparse
import os

from src.util import load_config_from_json
from src.train_all import run_exp

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', default='C:/Users/alvar/PycharmProjects/corilga_api')
    args = parser.parse_args()

    # Define important paths
    root_path = args.r
    data_path = os.path.join(root_path, 'dataset/')
    wav_path = os.path.join(data_path, 'Extracted_data/')
    csv_path = os.path.join(data_path, 'global_data.csv')
    results_path = os.path.join(root_path, 'results')

    # Define model parameters
    model_name = 'MLP'
    k_folds, seed = 4, 67

    # Define filters
    filters = {"audio_type": ["cough-heavy"]}

    # Define features
    feats_config = load_config_from_json(os.path.join(root_path, 'config', 'feature_config.json'))
    feats_config['feature_type'] = 'Spafe_lpcc'
    feats_config['extra_features'] = False

    run_exp(csv_path, wav_path, results_path, filters, feats_config, model_name, k_folds, seed)
