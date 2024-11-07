# Python base libraries
import os
import sys
import json
import pickle
import argparse
# ML libraries
import torch
import mlflow
import numpy as np
import pandas as pd
# Tools
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# Sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# Sklearn tools
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import permutation_test_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
# My libraries
from feats import FeatureExtractor
from util import load_config_from_json


def run_exp(path_data_: str, path_wav_: str, path_results_: str, filters: dict, feature_config_: dict,
            model_name: str, k_folds: int = 1, seed: int = 42):
    """
    Run an experiment with the given parameters
    :param path_data_: path to the metadata file
    :param path_wav_: path to the wav files
    :param path_results_: path to the results
    :param filters: filters to be applied to the metadata
    :param feature_config_: configuration of the features
    :param model_name: name of the model to be used
    :param k_folds: number of folds to be used
    :param seed: seed to be used
    :return: The trained model and the scores of the experiment
    """
    # Check and create the directory to save the experiment
    exp_name = f'results_{feature_config_["feature_type"]}_plus-{feature_config_["extra_features"]}_' \
               f'{filters["audio_type"][0].replace(r"/", "")}_seed-{seed}'
    exp_name = os.path.join(path_results_, exp_name)

    os.makedirs(exp_name, exist_ok=True)
    feature_config_['output_path'] = path_results_

    # Define the data to be used
    ADReSS_metadata = pd.read_csv(path_data_, decimal=',')
    # Make the metadata for the DICOPERIA dataset
    path_exp_metadata = os.path.join(path_results_, 'exp_metadata.csv')
    exp_metadata = make_ADReSS_metadata(path_exp_metadata, ADReSS_metadata, filters) # ya no haría falta en el caso de ADReSS

    # empezar aqui
    # Make the subsets
    sample_gain_testing = 0.3
    data_folds = make_train_test_subsets(exp_metadata, sample_gain_testing, k_folds, seed)

    for num_fold, (train, test, label_train, label_test) in enumerate(data_folds):
        train_path = f'train_feats_{feature_config_["feature_type"]}_{feature_config_["extra_features"]}_' \
                     f'{filters["audio_type"][0].replace(r"/", "")}_{num_fold}_{seed}.npy'
        # Make the features
        if not os.path.exists(os.path.join(path_results_, train_path)):
            train_feats, train_labels = make_feats(path_wav_, train, label_train, feature_config_)
            # Save the train features
            np.save(os.path.join(path_results_, train_path), train_feats)
            np.save(os.path.join(path_results_, train_path.replace('train_feats', 'train_labels')), train_labels)
            print(train_feats.shape)
            print(train_labels.shape)
        else:
            print("Loading training feats from disk...")
            train_feats = np.load(os.path.join(path_results_, train_path))
            train_labels = np.load(os.path.join(path_results_, train_path.replace('train_feats', 'train_labels')))
            print(train_feats.shape)
            print(train_labels.shape)

        if not os.path.exists(os.path.join(path_results_, train_path.replace('train_feats', 'test_feats'))):
            test_feats, test_label = make_test_feats(path_results_, path_wav_, [test, label_test], feature_config_)
            # Save the test features
            np.save(os.path.join(path_results_, train_path.replace('train_feats', 'test_feats')), test_feats)
            np.save(os.path.join(path_results_, train_path.replace('train_feats', 'test_labels')), test_label)
            #print(test_feats.shape)
            #print(test_label.shape)
        else:
            print("Loading testing feats from disk...")
            test_feats = np.load(os.path.join(path_results_, train_path.replace('train_feats', 'test_feats')),
                                 allow_pickle=True)
            test_label = np.load(os.path.join(path_results_, train_path.replace('train_feats', 'test_labels')),
                                 allow_pickle=True)
            #print(test_feats.shape)
            #print(test_label.shape)

        # Training phase
        path_model = os.path.join(exp_name, f'{model_name}_{num_fold}_results')
        # Create the directory to save the results
        os.makedirs(path_model, exist_ok=True)

        # Configure the model
        print("Configuring model...")
        model, x_train, y_train, test_feats, test_label = config_model(model_name, train_feats, train_labels,
                                                                       test_feats, test_label, seed)
        print("Model configured.")
        pikle_model = os.path.join(path_model, f'{model_name}_{num_fold}.pkl')
        if not os.path.exists(pikle_model):
            # Train the model
            print("Fitting model...")
            model.fit(x_train, y_train)
            # Save the model
            pickle.dump(model, open(pikle_model, 'wb'))
            print("Model fitted and dumped.")
        else:
            print("Loading model from disk...")
            model = pickle.load(open(pikle_model, 'rb'))

        # Calculate the performance metrics using sklearn
        score_path = os.path.join(path_model, f'{model_name}_{num_fold}scores.pkl')

        print("Making scores...")
        all_scores = score_sklearn(model, test_feats, test_label, score_path)
        print("Scores made.")

        # Log the results using mlflow
        mlflow_run(filters, feature_config_, model_name, num_fold, seed, all_scores)


def mlflow_run(filters: dict, feature_config: dict, model_name: str, num_fold: int, seed: int, all_scores: dict,
               port: str = '8080'):
    """
    Run the experiment using mlruns to track the results
    :param filters: A dictionary of filters to be used in the experiment
    :param feature_config: A dictionary of features to be used in the experiment
    :param model_name: Name of the model to be used
    :param num_fold: Number of folds to be used in the experiment
    :param seed: Random state to be used in the experiment
    :param all_scores: A dictionary of scores to be logged
    :param port: Port of mlFlow Server
    :return: None
    """
    # Define the experiment
    mlflow.set_tracking_uri(f'http://localhost:{port}')
    mlflow.set_experiment(f'{feature_config["feature_type"]}_{feature_config["extra_features"]}_'
                          f'{filters["audio_type"][0].replace(r"/", "")}_'
                          f'seed-{seed}')

    with mlflow.start_run(run_name=f'{model_name}_{num_fold}_seed-{seed}'):
        # Log the parameters
        mlflow.log_params(filters)
        mlflow.log_params(feature_config)
        mlflow.log_param('model', model_name)
        mlflow.log_param('model_arguments', f'{model_name}')
        mlflow.log_param("fold", num_fold + 1)
        mlflow.log_param('random_state', seed)

        # Log the metrics
        mlflow_metric = {k: v for k, v in all_scores.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(mlflow_metric)

        non_numeric_params = {k: v for k, v in all_scores.items() if not isinstance(v, (int, float))}
        for k, v in non_numeric_params.items():
            mlflow.log_param(k, v)


def score_sklearn(model_trained, test_feats: list, test_label: list, path_to_save: str) -> dict:
    """
    Calculate a set of performance metrics using sklearn
    :param model_trained: a model trained using for inference
    :param test_feats: a list of test feats
    :param test_label: a list of test labels
    :param path_to_save: Path to save the results
    :return: a set of performance metrics: confusion matrix, f1 score, f-beta score, precision, recall, and auc score
    """
    model_name = model_trained.__class__.__name__
    # Start testing
    y_feats, y_true = test_feats, test_label
    y_score = make_prediction(model_trained, model_name, y_feats)

    avg_importance, all_importances = make_permutations(model_trained, model_name, y_feats, y_true, n_repeats=30, random_state=42)
    print(f'avg_importance: {avg_importance}')
    print(f'avg_importance shape: {avg_importance.shape}')
    print(f'all_importances: {all_importances}')
    print(f'all_importances shape: {all_importances.shape}')

    # Calculate the auc_score, FP-rate, and TP-rate
    sklearn_roc_auc_score = roc_auc_score(y_true, y_score)
    sklear_fpr, sklearn_tpr, n_thresholds = roc_curve(y_true, y_score)

    # Make prediction using a threshold that maximizes the difference between TPR and FPR
    optimal_idx = np.argmax(sklearn_tpr - sklear_fpr)
    optimal_threshold = n_thresholds[optimal_idx]
    y_pred = [1 if scr > optimal_threshold else 0 for scr in y_score]

    # Calculate Precision, Recall, F1, and F-beta scores
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f_beta, support = precision_recall_fscore_support(y_true, y_pred)
    f1_scr = f1_score(y_true, y_pred)

    # Calculate Confusion Matrix
    confusion_mx = confusion_matrix(y_true, y_pred)

    # Calculate the specificity and sensitivity
    tn, fp, fn, tp = confusion_mx.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    dict_scores = {
        'model_name': model_name,
        'acc_score': float(acc),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'tpr': sklearn_tpr.tolist(),
        'tnr': sklear_fpr.tolist(),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'decision_thresholds': n_thresholds.tolist(),
        'optimal_threshold': float(optimal_threshold),
        'auc_score': float(sklearn_roc_auc_score),
        'confusion_matrix': confusion_mx.tolist(),
        'f1_scr': float(f1_scr),
        'f_beta': f_beta.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
    }

    # Plot useful metric graphs
    if path_to_save is not None:
        # Plot the ROC curve for the model
        plt.plot(sklear_fpr, sklearn_tpr, marker='.', label=model_name)
        plt.title('ROC Curve')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # Save the ROC curve plot
        plt.savefig(path_to_save.replace('.pkl', '_ROC.png'))
        plt.close()

        # Plot the Precision-Recall curve for the model
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(lr_precision, lr_recall, label=model_name)
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the title
        plt.title(f'Precision-Recall Curve of {model_name} (Threshold={optimal_threshold:.2f}')
        plt.savefig(path_to_save.replace('.pkl', '_precision_recall.png'))
        plt.close()

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mx, display_labels=['Control', 'Positive'])
        disp.plot()
        # Plot the Confusion Matrix for the threshold selected
        plt.title(f'CM of {model_name} (Threshold={optimal_threshold:.2f}')
        plt.savefig(path_to_save.replace('.pkl', '_confusion_matrix.png'))
        plt.close()

    print("--------------------------------------------")
    print("Scoring: Accuracy = {:.2f}, AUC = {:.2f}".format(acc, sklearn_roc_auc_score))
    print("Scoring: Sensitivity = {:.2f}, specificity = {:.2f}".format(sensitivity, specificity))
    print('============================================\n')
    return dict_scores


def make_prediction(model, model_name: str, y_feats: list) -> list:
    """
    Function to make prediction using a model
    @param model: A model to make prediction
    @param model_name: Type of the model
    @param y_feats: A list of features to make prediction
    @return: A list of prediction scores
    """
    # Predict the scores
    y_score = []
    for feat_ in tqdm(y_feats, total=len(y_feats)):
        # Predict
        if model_name == 'LSTMclassifier':
            with torch.no_grad():
                feat_ = feat_.to('cpu')
                output_score = model.predict_proba(feat_)
                output_score = sum(output_score)[0].item() / len(output_score)
        else:
            # Print a warning if the number of features is not the same
            if feat_.shape[1] != model.n_features_in_:
                print(f"Warning: The number of features is not the same. "
                      f"Expected {model.n_features_in_} but got {feat_.shape[1]}")
                if feat_.shape[1] < model.n_features_in_:
                    feat_ = np.concatenate((feat_, np.zeros((feat_.shape[0], model.n_features_in_ - feat_.shape[1]))),
                                           axis=1)
                elif feat_.shape[1] > model.n_features_in_:
                    feat_ = feat_[:, :model.n_features_in_]

            # Predict
            output_score = model.predict(feat_)
            output_score = float(np.mean(output_score))

        # Average the scores of all segments from the input file
        y_score.append(output_score)
    return y_score


def make_permutations(model, model_name: str, y_feats: list, y_true: list, n_repeats: int = 30,
                      random_state: int = None) -> tuple:
    """
    Function to calculate permutation importance for each patient's data.
    Similar to `make_prediction`, but computes permutation importance.

    @param model: Trained model to use for permutation importance.
    @param model_name: The name/type of model (e.g., 'LSTMclassifier').
    @param y_feats: List of features for each patient (each containing multiple segments).
    @param y_true: List of true labels for each patient.
    @param n_repeats: Number of times to permute a feature.
    @param random_state: Seed for reproducibility.
    @return: Average permutation importance and individual patient importances.
    """

    # Store all patient importances
    all_importances = []

    # Loop through each patient's features and calculate permutation importance
    for feat_, true_label in tqdm(y_feats, total=len(y_feats)):

        # Handle LSTM model case
        if model_name == 'LSTMclassifier':
            with torch.no_grad():
                feat_ = feat_.to('cpu')  # Ensure the feature is on CPU
                true_label = np.repeat(true_label, feat_.shape[0])  # Match label length to number of segments

                # Compute permutation importance using sklearn's function
                result = permutation_importance(model, feat_, true_label, n_repeats=n_repeats,
                                                random_state=random_state, scoring='accuracy')

        # Handle other model types
        else:
            # Ensure feature dimension compatibility with model
            if feat_.shape[1] != model.n_features_in_:
                if feat_.shape[1] < model.n_features_in_:
                    # Pad with zeros if fewer features than expected
                    feat_ = np.concatenate((feat_, np.zeros((feat_.shape[0], model.n_features_in_ - feat_.shape[1]))),
                                           axis=1)
                else:
                    # Trim extra features if there are more than expected
                    feat_ = feat_[:, :model.n_features_in_]

            true_label = np.repeat(true_label, feat_.shape[0])  # Repeat true label for all segments

            # Compute permutation importance
            result = permutation_importance(model, feat_, true_label, n_repeats=n_repeats, random_state=random_state,
                                            scoring='accuracy')

        # Append the mean importances for the patient
        all_importances.append(result.importances_mean)

    # Calculate average importance across all patients
    avg_importance = np.mean(all_importances, axis=0)

    return avg_importance, np.array(all_importances)

def config_model(model_name: str, training_feats, training_labels, test_feats, test_labels, seed: int = 42):
    """
    Function to configure a sklearn model
    @param model_name: Type of the model
    @param training_feats: Training features
    @param training_labels: Training labels
    @param test_feats: Test features
    @param test_labels: Testing labels
    @param seed: Random seed
    @return: Configured model and a processed features with its labels
    """
    models = {"LogisticRegression": {"c": 0.01,
                                     "max_iter": 40,
                                     "solver": "liblinear",
                                     "penalty": "l2",
                                     "class_weight": "balanced",
                                     "verbose": True
                                     },
              "RandomForest": {"n_estimators": 20,
                               "criterion": "gini",
                               "max_depth": None,
                               "min_samples_split": 2,
                               "min_samples_leaf": 1,
                               "max_features": "sqrt",
                               "class_weight": "balanced",
                               "verbose": True
                               },
              "MLP": {"learning_rate_init": 0.001,
                      "alpha": 0.001,
                      "solver": "adam",
                      "hidden_layer_sizes": [20, 20],
                      "max_iter": 500,
                      "activation": "tanh",
                      "class_weight": "balanced",
                      "verbose": True
                      },
              "LinearSVM": {"c": 0.1,
                            "tol": 0.001,
                            "max_iter": 1000,
                            "class_weight": None,
                            "verbose": True
                            }
              }

    model_args = models[model_name]

    if model_name == 'LogisticRegression':
        model = LogisticRegression(C=float(model_args['c']),
                                   max_iter=int(model_args['max_iter']),
                                   solver=model_args['solver'],
                                   penalty=model_args['penalty'],
                                   class_weight=model_args['class_weight'],
                                   random_state=seed,
                                   verbose=True)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=model_args['n_estimators'],
                                       criterion=model_args['criterion'],
                                       max_depth=model_args['max_depth'],
                                       min_samples_split=model_args['min_samples_split'],
                                       min_samples_leaf=model_args['min_samples_leaf'],
                                       max_features=model_args['max_features'],
                                       class_weight=model_args['class_weight'],
                                       random_state=seed)
    elif model_name == 'LinearSVM':
        model = SVC(C=model_args['c'],
                    tol=model_args['tol'],
                    max_iter=model_args['max_iter'],
                    verbose=model_args['verbose'],
                    random_state=seed)

        trans = StandardScaler()
        min_max = MinMaxScaler()
        training_feats = min_max.fit_transform(training_feats)
        training_feats = trans.fit_transform(training_feats)
        test_feats = np.array([min_max.transform(f) for f in test_feats])
        test_feats = np.array([trans.transform(f) for f in test_feats])

    elif model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=model_args['hidden_layer_sizes'],
                              solver=model_args['solver'], alpha=model_args['alpha'],
                              learning_rate_init=model_args['learning_rate_init'],
                              verbose=model_args['verbose'], activation=model_args['activation'],
                              max_iter=model_args['max_iter'], random_state=seed)

        if model_args['class_weight'] == 'balanced':
            train_data = np.concatenate((training_feats, training_labels.reshape(training_feats.shape[0], 1)), axis=1)
            ind = np.where(train_data[:, -1] == 1)[0]
            n_positives = len(ind)
            n_negatives = train_data.shape[0] - n_positives
            up_sample_factor = int(n_negatives / n_positives) - 1
            for i in range(up_sample_factor):
                train_data = np.concatenate((train_data, train_data[ind, :]), axis=0)
            np.random.shuffle(train_data)
            training_feats = train_data[:, :-1]
            training_labels = train_data[:, -1]
    else:
        raise ValueError("Not implementation of the model: " + model_name)
    return model, training_feats, training_labels, test_feats, test_labels


def make_test_feats(path_to_save: str, path_wav: str, test_data: np.array, feats_config_: dict) -> (np.array, np.array):
    """
    Make the test features and get the labels
    :param path_to_save: path to save the features
    :param path_wav: path to the wav files
    :param test_data: test data
    :param feats_config_: features configuration
    :return: test features and its labels
    """
    y_feats, y_true = [], []
    # Make the test features and get the labels
    if not os.path.exists(os.path.join(path_to_save, 'y_feats.npy')):
        for ind, audio_id in tqdm(test_data[0].items(), total=test_data[0].shape[0]):
            # Prepare features
            FE = FeatureExtractor(feats_config_)
            try:
                F = FE.extract(os.path.join(path_wav, audio_id), audio_id)
                if not (np.isnan(F).any() or np.isinf(F).any()):
                    y_feats.append(np.array(F))
                    y_true.append(test_data[1][ind])
            except (Warning, Exception) as e:
                continue
    return y_feats, y_true


def make_feats(path_to_wav: str, audio_id: pd.DataFrame, labels: pd.DataFrame, feats_config_: dict) -> np.array:
    """
    Extract features from audio files
    :param path_to_wav: path to audio files
    :param audio_id: audio ids as a pandas dataframe
    :param labels: labels as a pandas dataframe
    :param feats_config_: feature configuration as a dictionary
    :return: features and labels
    """
    # Prepare feature extractor
    FE = FeatureExtractor(feats_config_)

    egs = []

    data = pd.concat([audio_id, labels], axis=1)
    for row in tqdm(data.iterrows(), total=data.shape[0]):
        audio_id = row[1]['root']
        label = row[1]['AD_status']
        # Prepare features
        # print(f'Procesando: {os.path.join(path_to_wav, audio_id)}')
        try:
            #print('hola')
            F = FE.extract(os.path.join(path_to_wav, audio_id), audio_id)
                  
        
            print(F)
            print(f'F en train_all: {F.shape}')
            if not(np.isnan(F).any() or np.isinf(F).any()):
                egs.append(np.concatenate((np.array(F), np.array([label] * F.shape[0]).reshape(F.shape[0], 1)), axis=1))
            else:
                print(f'Problema con: {os.path.join(path_to_wav, audio_id)}')
        except (Warning, Exception) as e:
            continue
    egs = np.vstack(egs)
    print(f'egs shape: {egs.shape}')

    return np.array(egs[:, :-1], dtype=float), np.array(egs[:, -1], dtype=int)


def make_k_fold_subsets(exp_metadata, k_fold, seed):
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)
    # Iterate over the k-folds
    k_folds = []
    for ind, (train_index, test_index) in enumerate(kf.split(exp_metadata['ID'])):
        # Get the train and test data
        audio_train = exp_metadata.iloc[train_index]['root']
        audio_test = exp_metadata.iloc[test_index]['root']

        label_train = exp_metadata.iloc[train_index]['AD_status']
        label_test = exp_metadata.iloc[test_index]['AD_status']

        k_folds.append([audio_train, audio_test, label_train, label_test])
    return k_folds


def make_train_test_subsets(metadata: pd.DataFrame, test_size: float = 0.3, k_fold: int = 1, seed: int = 42):
    """
    Split the data into train and test subsets. The split is done by patients.
    :param metadata: metadata as a pandas dataframe
    :param test_size: test size in percentage
    :param k_fold: number of folds
    :param seed: random seed
    :return: train and test subsets of features and its labels
    """

    if k_fold <= 1:
        PATIENT_ID_COLUMN = 'ID'
        CLASS_COLUMN = 'AD_status'
        AUDIO_ID_COLUMN = 'root'  # ruta al audio
        TT_COLUMN = 'tt' # Train-Test column

        # Filtrar los datos de entrenamiento y prueba en base a la columna 'tt'
        audio_data_train = metadata[metadata[TT_COLUMN] == 'train']
        audio_train = audio_data_train[AUDIO_ID_COLUMN]
        audio_label_train = audio_data_train[CLASS_COLUMN]

        audio_data_test = metadata[metadata[TT_COLUMN] == 'test']
        audio_test = audio_data_test[AUDIO_ID_COLUMN]
        audio_label_test = audio_data_test[CLASS_COLUMN]

        # Obtener el número de pacientes únicos en train y test
        pat_train = audio_data_train[PATIENT_ID_COLUMN].nunique()
        pat_test = audio_data_test[PATIENT_ID_COLUMN].nunique()

        # Imprimir la longitud de cada conjunto
        print(f"\tTest-set: {pat_test} patients & {len(audio_data_test)} samples")
        print(f"\tTrain-set: {pat_train} patients & {len(audio_data_train)} samples")

        # Devolver los conjuntos
        return [[audio_train, audio_test, audio_label_train, audio_label_test]]

        # # Making the subsets by patients
        # PATIENT_ID_COLUMN = 'ID'
        # CLASS_COLUMN = 'AD_status'
        # AUDIO_ID_COLUMN = 'id_audio'
        #
        # # Prepare the metadata
        # patient_data = metadata[[PATIENT_ID_COLUMN, CLASS_COLUMN]].drop_duplicates()
        # patient_id = patient_data[PATIENT_ID_COLUMN]
        # patient_class = patient_data[CLASS_COLUMN]
        #
        # # Split the data
        # pat_train, pat_test, pat_labels_train, pat_labels_test = train_test_split(patient_id, patient_class,
        #                                                                           test_size=test_size,
        #                                                                           stratify=patient_class,
        #                                                                           random_state=seed,
        #                                                                           shuffle=True)
        # # Using the patient subsets to select the audio samples
        # audio_data_train = metadata[(metadata[PATIENT_ID_COLUMN].isin(pat_train))]
        # audio_train = audio_data_train[AUDIO_ID_COLUMN]
        # audio_label_train = audio_data_train[CLASS_COLUMN]
        #
        # audio_data_test = metadata[(metadata[PATIENT_ID_COLUMN].isin(pat_test))]
        # audio_test = audio_data_test[AUDIO_ID_COLUMN]
        # audio_label_test = audio_data_test[CLASS_COLUMN]
        #
        # # Print the final length of each subset
        # print(f"\tTest-set: {len(pat_test)} patients & {len(audio_data_test)} samples")
        # print(f"\tTrain-set: {len(pat_train):} patients & {len(audio_data_train)} samples")
        # return [[audio_train, audio_test, audio_label_train, audio_label_test]]
    else:
        k_folds = make_k_fold_subsets(metadata, k_fold, seed)
        return k_folds


def make_ADReSS_metadata(save_path: str, metadata: pd.DataFrame, filters_: dict = None, remove_samples: dict = None):
    """
    Make a metadata file for the ADReSS dataset filtering some columns
    :param save_path: path to save the metadata file
    :param metadata: a list with all the audio samples in COPERIA as an Audio class
    :param filters_: a dictionary with the columns and values to keep
    :param remove_samples: a dictionary with the columns and values to remove
    :return: a pandas dataframe with the metadata of the DICOPERIA dataset
    """
    print('=== Filtering the metadata... ===')
    df = metadata.copy()

    if filters_ is None:
        filters_ = {'AD_status': ['AD', 'non-AD']}
        # filters_ = {'covid_status': ['healthy', 'no_resp_illness_exposed', 'resp_illness_not_identified',
        #             'positive_moderate', 'recovered_full', 'positive_mild', 'positive_asymp', 'under_validation']}

    for key, values in filters_.items():
        if 'ALL' in values:
            values = list(df[key].unique())

        df = df[df[key].isin(values)]

    # df.replace(['healthy', 'no_resp_illness_exposed', 'resp_illness_not_identified',
    #             'positive_moderate', 'recovered_full', 'positive_mild',
    #             'positive_asymp', 'under_validation'], [0, 0, 1, 1, 0, 1, 1, 1], inplace=True)   # añadir los covid_status

    df.replace(['AD', 'non-AD'], [1, 0], inplace=True)  # añadir los AD_status

    df.to_csv(save_path, index=False, decimal=',')
    print('Metadata saved in: {}'.format(save_path))
    print('=== Filtering DONE!! ===')
    return df


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', default='/home/projects/avera/DementiaBank')
    args = parser.parse_args()

    # Define important paths
    root_path = args.r
    data_path = os.path.join(root_path, 'dataset/')
    wav_path = os.path.join(data_path, 'Extracted_data/')
    csv_path = os.path.join(data_path, 'ADReSS_db.csv')   # poner nombre DB
    results_path = os.path.join(root_path, 'results')
    # Data filters
    all_filters = load_config_from_json(os.path.join(root_path, 'config', 'filter_config.json'))
    # Feature configuration
    feats_config = load_config_from_json(os.path.join(root_path, 'config', 'feature_config.json'))
    # Models configurations
    MODELS = load_config_from_json(os.path.join(root_path, 'config', 'models_config.json'))
    # Features sizes
    feats_sizes = load_config_from_json(os.path.join(root_path, 'config', 'feature_size.json'))
    # Runs configuration
    run_config = load_config_from_json(os.path.join(root_path, 'config', 'run_config.json'))
    num_folds = run_config['k_folds']
    random_state = run_config['seed']

    # Run the experiments
    for exp_filter in all_filters:
        # Select the feats
        all_feats = ['BERT_embedding', 'XVector', 'wav2vec',
                     'embMFCC', 'embRasta', 'embBasicSpectral', 'embVoicing', 'embEnergy',
                     'bert_MFCC', 'bert_wav2vec', 'MFCC_spectral']
        extra_features = [True, False] # volver a poner true

        for feat in all_feats:
            feats_config['feature_type'] = feat
            if 'ComParE_2016' in feat:
                extra_features = [False, True]

            for extra in extra_features:
                feats_config['extra_features'] = extra

                # Variar MLP hidden_layer_sizes ---------------------
                feature_size = feats_sizes[feat][str(extra)][0]
                print(f'Feature size: {feature_size}')
                classes = 2
                hidden_layer_size = ((feature_size + classes) // 2,)  # Convertir a entero y tupla

                # Asignar el nuevo valor en el modelo MLP
                if 'MLP' in MODELS:
                    MODELS['MLP']['hidden_layer_sizes'] = hidden_layer_size
                # ---------------------------------------------------

                # Select the model
                for model_ in MODELS.keys():
                    print('================================================================'
                          f'Running experiment with: \n'
                          f'    Model   :{model_}  -  Seed:{random_state}\n'
                          f'    Filters :{exp_filter}\n'
                          f'    Features:{feat} extra:{extra}\n'
                          f'----------------------------------------------------------------')
                    run_exp(csv_path, wav_path, f'{results_path}_{random_state}',
                            exp_filter, feats_config, model_,
                            num_folds, random_state)
