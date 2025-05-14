import os
import csv
from radiomics import featureextractor
import SimpleITK as sitk
import pandas as pd
from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy as np
import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def extract_radiomics_features(image_paths, mask_paths, output_csv_path, params_path=None):
    assert len(image_paths) == len(mask_paths), "Image and mask lists must be of the same length"

    # Load feature extractor (with or without parameter YAML file)
    if params_path:
        extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()

    # To store feature dictionaries for all cases
    all_features = []

    for img_path, mask_path in tqdm(zip(image_paths, mask_paths)):
        try:
            print(f"Extracting features from:\n  Image: {img_path}\n  Mask:  {mask_path}")
            features = extractor.execute(img_path, mask_path)

            # Use image filename (without extension) as subject ID
            subject_id = os.path.splitext(os.path.basename(img_path))[0]
            if subject_id.endswith(".nii"):
                subject_id = subject_id[:-4]  # Remove .nii if double-extension like .nii.gz

            # Filter out non-feature keys (image info etc.)
            features_filtered = {k: v for k, v in features.items() if k.startswith('original') or k.startswith('diagnostics')}
            features_filtered['SubjectID'] = subject_id

            all_features.append(features_filtered)

        except Exception as e:
            print(f"Error processing {img_path} and {mask_path}:\n{e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    df = df.set_index("SubjectID")
    df.sort_index(inplace=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path)
    print(f"\nRadiomics feature matrix saved to: {output_csv_path}")

def train_svm_rf(radiomics_csv_path, model_type="rf", score="roc_auc"):

    df = pd.read_csv(radiomics_csv_path)
    X = df.iloc[:, 38:].values

    labels = []
    institution = []
    subjects = []
    temp = pd.read_csv('./data/sarcoma/patient_metadata.csv')
    for patient in df["SubjectID"]:        
        patient_id = patient[:6] if patient.startswith("Sar") else patient[:4]
        subjects.append(patient_id)
        grading = temp[temp["ID"] == patient_id].Grading.item()
        grading = 0 if grading == 1 else 1            
        labels.append(grading) 
        site = 0 if patient_id.startswith("Sar") else 1
        institution.append(site)

    y = np.array(labels)

    total_features = X.shape[1]
    n_components_list = [int(total_features * i) for i in np.arange(0.1, 1.1, 0.1)]    

    # ---- DEFINE MODEL PIPELINES ----
    if model_type == "rf":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'pca__n_components': n_components_list, 
            "classifier__n_estimators": [50, 100, 250],
            "classifier__max_depth": [None, 5, 10, 20]
        }
    elif model_type == "svm":
        model = SVC(probability=True, random_state=42)
        param_grid = {
            'pca__n_components': n_components_list,  
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"]
        }
    else:
        raise ValueError("Invalid model. Choose 'rf' or 'svm'.")

    pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', model)
    ]) 
    
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Inner CV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, n_jobs=-1, scoring=score)

    # Nested CV
    nested_score = cross_val_score(estimator=grid_search, X=X, y=y, cv=outer_cv, n_jobs=-1, scoring=score)
    print("Nested CV Score (mean): ", nested_score.mean())
    print("Nested CV Score (std) : ", nested_score.std())

    # cv_results = cross_validate(estimator=grid_search, X=X, y=y, cv=outer_cv, n_jobs=-1, scoring=score, return_estimator=True)
    # estimators = cv_results['estimator']
    # scores = cv_results['test_score']
    # best_idx = np.argmax(scores)
    # best_score = scores[best_idx]
    # best_score_mean = np.mean(best_score)
    # best_score_std = np.std(best_score)
    # best_model = estimators[best_idx]
    # best_model_params = best_model.get_params()    
    # mlflow.log_param("best_model_params", best_model_params)
    # mlflow.log_param(f"{score}_score_mean", best_score_mean)
    # mlflow.log_param(f"{score}_score_std", best_score_std)    

    mlflow.log_param("model_type", model_type)
    mlflow.log_param("param_grid", param_grid)
    mlflow.log_param(f"{score}_score_mean", nested_score.mean())
    mlflow.log_param(f"{score}_score_std", nested_score.std())

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        mlflow.log_param(f"fold_{fold}_train_idx", train_idx)
        mlflow.log_param(f"fold_{fold}_test_idx", test_idx)
        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]
        mlflow.log_param(f"fold_{fold}_train_subjects", train_subjects)
        mlflow.log_param(f"fold_{fold}_test_subjects", test_subjects)

if __name__ == "__main__":
    
    task = "sarcoma_t1_grading_binary"                                

    match task:
        case "sarcoma_t1_grading_binary":
            image_paths = glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/*/T1/*.nii.gz")
            image_paths = sorted([path for path in image_paths if not "label" in path])
            mask_paths = glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/*/T1/*.nii.gz")
            mask_paths = sorted([path for path in mask_paths if "label" in path])
            output_path = f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/radiomics_cache_dir/radiomics_features_{task}.csv"
        
        case "sarcoma_t2_grading_binary":
            image_paths = glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/*/T2/*.nii.gz")
            image_paths = sorted([path for path in image_paths if not "label" in path])
            mask_paths = glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/*/T2/*.nii.gz")
            mask_paths = sorted([path for path in mask_paths if "label" in path])
            output_path = f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/radiomics_cache_dir/radiomics_features_{task}.csv"

        case _:
            raise ValueError(f"Unknown task: {task}")        

    if os.path.exists(output_path):
        print(f"Radiomics features already extracted. Skipping extraction.")
    else:
        # Extract features
        print(f"Extracting radiomics features of {len(image_paths)} images for task: {task}")
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Extract features 
        extract_radiomics_features(image_paths, mask_paths, output_path, params_path="/home/johannes/Data/SSD_1.9TB/MultiViewGCN/pyradiomics_mri_params.yaml")

    for model in ["rf", "svm"]:
        mlflow.set_experiment(task)
        mlflow.start_run()    
        train_svm_rf(output_path, model_type=model, score="roc_auc")
        mlflow.end_run()
