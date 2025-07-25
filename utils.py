import yaml
import random
import numpy as np
import torch
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import subprocess
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def seed_everything(seed: int):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Function to save the confusion matrix as a PNG
def save_confusion_matrix(y_true, y_pred, result_dir, split):

    filepath = os.path.join(result_dir, f"confusion_matrix_{split}.png")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Class 0', 'Class 1']  # Class labels

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)  # Save the figure as a PNG file
    plt.close()  # Close the figure to avoid displaying it in interactive environments

def save_conda_yaml():
    # Set filename
    conda_yaml_path = "conda.yaml"

    # Run the command to export current conda environment to a YAML file
    try:
        with open(conda_yaml_path, "w") as f:
            subprocess.run(
                ["conda", "env", "export"],
                stdout=f,
                check=True,
            )
        print(f"Conda environment successfully exported to '{conda_yaml_path}'.")

    except subprocess.CalledProcessError as e:
        print("Failed to export conda environment:", e)

    return conda_yaml_path


def create_cv_splits(task: str, seed: int = 28) -> None:

    match task:
        case "sarcoma_t1_grading_binary":

            save_path = f"./data/sarcoma/{task}_folds.pt"
            if not os.path.exists(save_path):

                files = sorted(glob("./data/sarcoma/*/T1/*.nii.gz"))
                files = [file for file in files if not "label" in file]
                subjects = [file.split("/")[-1].replace(".nii.gz", "") for file in files]
                subjects = [subject.replace("T1", "") for subject in subjects]
                subjects = [subject.replace("_updated", "") for subject in subjects]

                labels = []
                df = pd.read_csv("./data/sarcoma/patient_metadata.csv")

                for subject in subjects:
                    grading = df[df["ID"] == subject].Grading.item()
                    labels.append(0 if grading == 1 else 1)

                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                dict_folds = {}
                for fold, (train_idx, test_idx) in enumerate(skfold.split(np.zeros((len(labels))), labels)):
                    train_subjects = [subjects[i] for i in train_idx]
                    train_labels = [labels[i] for i in train_idx]
                    test_subjects = [subjects[i] for i in test_idx]
                    test_labels = [labels[i] for i in test_idx]

                    dict_folds[fold] = {"train_subjects": train_subjects,
                                        "train_labels": train_labels,
                                        "test_subjects": test_subjects,
                                        "test_labels": test_labels
                                        }

                torch.save(dict_folds, save_path)
                print("CV Splits saved successfully!")
            else:
                print("CV Splits already exist.")
        
        case "sarcoma_t2_grading_binary":
            
            save_path = f"./data/sarcoma/{task}_folds.pt"
            if not os.path.exists(save_path):

                files = sorted(glob("./data/sarcoma/*/T2/*.nii.gz"))
                files = [file for file in files if not "label" in file]
                subjects = [file.split("/")[-1].replace(".nii.gz", "") for file in files]
                subjects = [subject.replace("STIR", "") for subject in subjects]
                subjects = [subject.replace("_updated", "") for subject in subjects]
                subjects = [subject.replace("_ax", "") for subject in subjects]
                subjects = [subject.replace("_sag", "") for subject in subjects]

                labels = []
                df = pd.read_csv("./data/sarcoma/patient_metadata.csv")

                for subject in subjects:
                    grading = df[df["ID"] == subject].Grading.item()
                    labels.append(0 if grading == 1 else 1)

                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                dict_folds = {}
                for fold, (train_idx, test_idx) in enumerate(skfold.split(subjects, labels)):
                    train_subjects = [subjects[i] for i in train_idx]
                    train_labels = [labels[i] for i in train_idx]
                    test_subjects = [subjects[i] for i in test_idx]
                    test_labels = [labels[i] for i in test_idx]

                    dict_folds[fold] = {"train_subjects": train_subjects,
                                        "train_labels": train_labels,
                                        "test_subjects": test_subjects,
                                        "test_labels": test_labels
                                        }

                torch.save(dict_folds, save_path)
                print("CV Splits saved successfully!")
            
            else:
                print("CV Splits already exist.")
        
        case "glioma_t1_grading_binary":
            save_path = f"./data/ucsf/{task}_folds.pt"
            if not os.path.exists(save_path):
                files = sorted(glob("./data/ucsf/glioma_four_sequences/*T1_bias.nii.gz"))
                subjects = [os.path.basename(subject).split("_")[0] for subject in files]
                subjects = [patient_id.split("-")[0] + "-" + patient_id.split("-")[1] + "-" + patient_id.split("-")[2][1:] for patient_id in subjects]

                labels = []
                df = pd.read_csv("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/UCSF-PDGM-metadata_v2.csv")

                for subject in subjects:
                    grading = df[df["ID"] == subject]["WHO CNS Grade"].item()
                    labels.append(0 if grading < 4 else 1)

                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                dict_folds = {}
                for fold, (train_idx, test_idx) in enumerate(skfold.split(subjects, labels)):
                    train_subjects = [subjects[i] for i in train_idx]
                    train_labels = [labels[i] for i in train_idx]
                    test_subjects = [subjects[i] for i in test_idx]
                    test_labels = [labels[i] for i in test_idx]

                    dict_folds[fold] = {"train_subjects": train_subjects,
                                        "train_labels": train_labels,
                                        "test_subjects": test_subjects,
                                        "test_labels": test_labels
                                        }

                torch.save(dict_folds, save_path)
                print("CV Splits saved successfully!")
            else:
                print("CV Splits already exist.")
        
        case "glioma_t2_grading_binary":
            save_path = f"./data/ucsf/{task}_folds.pt"
            if not os.path.exists(save_path):
                files = sorted(glob("./data/ucsf/glioma_four_sequences/*T2_bias.nii.gz"))
                subjects = [os.path.basename(subject).split("_")[0] for subject in files]
                subjects = [patient_id.split("-")[0] + "-" + patient_id.split("-")[1] + "-" + patient_id.split("-")[2][1:] for patient_id in subjects]

                labels = []
                df = pd.read_csv("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/UCSF-PDGM-metadata_v2.csv")

                for subject in subjects:
                    grading = df[df["ID"] == subject]["WHO CNS Grade"].item()
                    labels.append(0 if grading < 4 else 1)

                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                dict_folds = {}
                for fold, (train_idx, test_idx) in enumerate(skfold.split(subjects, labels)):
                    train_subjects = [subjects[i] for i in train_idx]
                    train_labels = [labels[i] for i in train_idx]
                    test_subjects = [subjects[i] for i in test_idx]
                    test_labels = [labels[i] for i in test_idx]

                    dict_folds[fold] = {"train_subjects": train_subjects,
                                        "train_labels": train_labels,
                                        "test_subjects": test_subjects,
                                        "test_labels": test_labels
                                        }

                torch.save(dict_folds, save_path)
                print("CV Splits saved successfully!")
            else:
                print("CV Splits already exist.")

        case "glioma_t1c_grading_binary":
            save_path = f"./data/ucsf/{task}_folds.pt"
            if not os.path.exists(save_path):
                files = sorted(glob("./data/ucsf/glioma_four_sequences/*T1c_bias.nii.gz"))
                subjects = [os.path.basename(subject).split("_")[0] for subject in files]
                subjects = [patient_id.split("-")[0] + "-" + patient_id.split("-")[1] + "-" + patient_id.split("-")[2][1:] for patient_id in subjects]

                labels = []
                df = pd.read_csv("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/UCSF-PDGM-metadata_v2.csv")

                for subject in subjects:
                    grading = df[df["ID"] == subject]["WHO CNS Grade"].item()
                    labels.append(0 if grading < 4 else 1)

                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                dict_folds = {}
                for fold, (train_idx, test_idx) in enumerate(skfold.split(subjects, labels)):
                    train_subjects = [subjects[i] for i in train_idx]
                    train_labels = [labels[i] for i in train_idx]
                    test_subjects = [subjects[i] for i in test_idx]
                    test_labels = [labels[i] for i in test_idx]

                    dict_folds[fold] = {"train_subjects": train_subjects,
                                        "train_labels": train_labels,
                                        "test_subjects": test_subjects,
                                        "test_labels": test_labels
                                        }

                torch.save(dict_folds, save_path)
                print("CV Splits saved successfully!")
            else:
                print("CV Splits already exist.")

        case "glioma_flair_grading_binary":
            save_path = f"./data/ucsf/{task}_folds.pt"
            if not os.path.exists(save_path):
                files = sorted(glob("./data/ucsf/glioma_four_sequences/*FLAIR_bias.nii.gz"))
                subjects = [os.path.basename(subject).split("_")[0] for subject in files]
                subjects = [patient_id.split("-")[0] + "-" + patient_id.split("-")[1] + "-" + patient_id.split("-")[2][1:] for patient_id in subjects]

                labels = []
                df = pd.read_csv("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/UCSF-PDGM-metadata_v2.csv")

                for subject in subjects:
                    grading = df[df["ID"] == subject]["WHO CNS Grade"].item()
                    labels.append(0 if grading < 4 else 1)

                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                dict_folds = {}
                for fold, (train_idx, test_idx) in enumerate(skfold.split(subjects, labels)):
                    train_subjects = [subjects[i] for i in train_idx]
                    train_labels = [labels[i] for i in train_idx]
                    test_subjects = [subjects[i] for i in test_idx]
                    test_labels = [labels[i] for i in test_idx]

                    dict_folds[fold] = {"train_subjects": train_subjects,
                                        "train_labels": train_labels,
                                        "test_subjects": test_subjects,
                                        "test_labels": test_labels
                                        }

                torch.save(dict_folds, save_path)
                print("CV Splits saved successfully!")
        
        case "headneck_ct_hpv_binary":
            save_path = f"./data/headneck/{task}_folds.pt"
            if not os.path.exists(save_path):
                dirs = sorted(glob("./data/headneck/converted_nii_merged/*"))
                subjects = [os.path.basename(d) for d in dirs]

                labels = []
                df = pd.read_csv("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/headneck/patient_metadata_filtered.csv")

                for subject in subjects:
                    hpv = df[df["id"] == subject]["hpv"].item()
                    labels.append(0 if hpv == "negative" else 1)

                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                dict_folds = {}
                for fold, (train_idx, test_idx) in enumerate(skfold.split(subjects, labels)):
                    train_subjects = [subjects[i] for i in train_idx]
                    train_labels = [labels[i] for i in train_idx]
                    test_subjects = [subjects[i] for i in test_idx]
                    test_labels = [labels[i] for i in test_idx]

                    dict_folds[fold] = {"train_subjects": train_subjects,
                                        "train_labels": train_labels,
                                        "test_subjects": test_subjects,
                                        "test_labels": test_labels
                                        }

                torch.save(dict_folds, save_path)
                print("CV Splits saved successfully!")
        
        case "breast_mri_grading_binary":
            save_path = f"./data/breast/{task}_folds.pt"
            if not os.path.exists(save_path):
                dirs = sorted(glob("./data/breast/duke_tumor_grading/*segmentation.nii.gz"))
                subjects = [os.path.basename(d).replace("_segmentation.nii.gz", "") for d in dirs]

                labels = []
                df = pd.read_excel("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/breast/clinical_and_imaging_info.xlsx")

                for subject in subjects:
                    grade = df[df["patient_id"] == subject]["nottingham_grade"].item()
                    labels.append(0 if grade == "high" else 1)

                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                dict_folds = {}
                for fold, (train_idx, test_idx) in enumerate(skfold.split(subjects, labels)):
                    train_subjects = [subjects[i] for i in train_idx]
                    train_labels = [labels[i] for i in train_idx]
                    test_subjects = [subjects[i] for i in test_idx]
                    test_labels = [labels[i] for i in test_idx]

                    dict_folds[fold] = {"train_subjects": train_subjects,
                                        "train_labels": train_labels,
                                        "test_subjects": test_subjects,
                                        "test_labels": test_labels
                                        }

                torch.save(dict_folds, save_path)
                print("CV Splits saved successfully!")
        
        case "kidney_ct_grading_binary":
            save_path = f"./data/kidney/{task}_folds.pt"
            if not os.path.exists(save_path):
                dirs = sorted(glob("./data/kidney/converted_nii/*segmentation.nii.gz"))
                subjects = [os.path.basename(d).replace("_segmentation.nii.gz", "") for d in dirs]

                labels = []
                df = pd.read_csv("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/kidney/C4KC-KiTS_Clinical-Data_Version-1.csv")
                df = df.dropna(subset=["tumor_isup_grade"])

                final_subjects = []
                for subject in subjects:
                    try:
                        grade = df[df["patient_id"] == subject]["tumor_isup_grade"].item()
                        labels.append(0 if grade <= 2 else 1)
                        final_subjects.append(subject)
                    except ValueError:
                        continue
                subjects = final_subjects

                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                dict_folds = {}
                for fold, (train_idx, test_idx) in enumerate(skfold.split(subjects, labels)):
                    train_subjects = [subjects[i] for i in train_idx]
                    train_labels = [labels[i] for i in train_idx]
                    test_subjects = [subjects[i] for i in test_idx]
                    test_labels = [labels[i] for i in test_idx]

                    dict_folds[fold] = {"train_subjects": train_subjects,
                                        "train_labels": train_labels,
                                        "test_subjects": test_subjects,
                                        "test_labels": test_labels
                                        }

                torch.save(dict_folds, save_path)
                print("CV Splits saved successfully!")
        
        case "liver_ct_riskscore_binary":
            save_path = f"./data/liver/{task}_folds.pt"
            if not os.path.exists(save_path):
                dirs = sorted(glob("./data/liver/converted_nii/*segmentation.nii.gz"))
                subjects = [os.path.basename(d).replace("_segmentation.nii.gz", "") for d in dirs]

                labels = []
                df = pd.read_excel("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/liver/Colorectal-Liver-Metastases-Clinical-data-April-2023.xlsx")
                df = df[df['clinrisk_stratified'] != -999]

                final_subjects = []
                for subject in subjects:
                    try:
                        risk_score = df[df["Patient-ID"] == subject]["clinrisk_stratified"].item()
                        labels.append(risk_score)
                        final_subjects.append(subject)
                    except ValueError:
                        continue
                subjects = final_subjects

                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                dict_folds = {}
                for fold, (train_idx, test_idx) in enumerate(skfold.split(subjects, labels)):
                    train_subjects = [subjects[i] for i in train_idx]
                    train_labels = [labels[i] for i in train_idx]
                    test_subjects = [subjects[i] for i in test_idx]
                    test_labels = [labels[i] for i in test_idx]

                    dict_folds[fold] = {"train_subjects": train_subjects,
                                        "train_labels": train_labels,
                                        "test_subjects": test_subjects,
                                        "test_labels": test_labels
                                        }

                torch.save(dict_folds, save_path)
                print("CV Splits saved successfully!")
            
            else:
                print("CV Splits already exist.")

        case _:
            raise ValueError("Given task unkown!")


def calculate_hidden_units(output_dim_enc: int, target_params: int = 100000) -> int:
    """
    Calculate the number of hidden units in the encoder based on the output dimension and target parameters.
    
    Args:
        output_dim_enc (int): The output dimension of the encoder.
        target_params (int): The target number of parameters for the encoder.
        
    Returns:
        int: The calculated number of hidden units.
    """
    
    num_params = (target_params // (output_dim_enc +  3))

    return num_params