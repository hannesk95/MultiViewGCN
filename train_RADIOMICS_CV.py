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
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from utils import create_cv_splits


def extract_radiomics_features(task, params_path=None):

    match task:
        case "sarcoma_t1_grading_binary":
            image_paths = glob("./data/sarcoma/*/T1/*.nii.gz")
            image_paths = sorted([path for path in image_paths if not "label" in path])
            mask_paths = glob("./data/sarcoma/*/T1/*.nii.gz")
            mask_paths = sorted([path for path in mask_paths if "label" in path])
            output_csv_path = f"./radiomics_cache_dir/radiomics_features_{task}.csv"
        
        case "sarcoma_t2_grading_binary":
            image_paths = glob("./data/sarcoma/*/T2/*.nii.gz")
            image_paths = sorted([path for path in image_paths if not "label" in path])
            mask_paths = glob("./data/sarcoma/*/T2/*.nii.gz")
            mask_paths = sorted([path for path in mask_paths if "label" in path])
            output_csv_path = f"./radiomics_cache_dir/radiomics_features_{task}.csv"

        case _:
            raise ValueError(f"Unknown task: {task}")
        
    if os.path.exists(output_csv_path):
        print(f"Radiomics features already extracted. Skipping extraction.")
    
    else:
        # Extract features
        print(f"Extracting radiomics features of {len(image_paths)} images for task: {task}")
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

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

def train_classifier(task, model_type="rf", score="roc_auc"):

    match task:
        case "sarcoma_t1_grading_binary":
            radiomics_csv_path = f"./radiomics_cache_dir/radiomics_features_{task}.csv"
            assert os.path.exists(radiomics_csv_path)
            df = pd.read_csv(radiomics_csv_path)
            df["SubjectID"] = df["SubjectID"].str.replace("T1", "", regex=True)
            df["SubjectID"] = df["SubjectID"].str.replace("_updated_T1", "", regex=True)
            df["SubjectID"] = df["SubjectID"].str.replace("_updated", "", regex=True)
            create_cv_splits(task=task)
            folds_dict = torch.load(f"./data/sarcoma/{task}_folds.pt", weights_only=False)
        
        case "sarcoma_t2_grading_binary":
            radiomics_csv_path = f"./radiomics_cache_dir/radiomics_features_{task}.csv"
            assert os.path.exists(radiomics_csv_path)
            df = pd.read_csv(radiomics_csv_path)
            
            df["SubjectID"] = df["SubjectID"].str.replace("STIR_ax", "", regex=True)
            df["SubjectID"] = df["SubjectID"].str.replace("STIR_sag", "", regex=True)
            df["SubjectID"] = df["SubjectID"].str.replace("_updatedSTIR", "", regex=True)
            df["SubjectID"] = df["SubjectID"].str.replace("STIR", "", regex=True)
            folds_dict = torch.load("./data/sarcoma/sarcoma_t2_grading_binary_folds.pt")

        case _:
            raise ValueError("Given task is unkonwn!")
    
    outer_results_auroc = []
    outer_results_mcc = []
    for current_fold in range(5):

        print(f"\nFold {current_fold}")

        train_subjects = folds_dict[current_fold]["train_subjects"]
        train_labels = folds_dict[current_fold]["train_labels"]
        test_subjects = folds_dict[current_fold]["test_subjects"]
        test_labels = folds_dict[current_fold]["test_labels"]

        train_df = df[df['SubjectID'].isin(train_subjects)]
        X_train = train_df.iloc[:, 38:].values
        y_train = np.array(train_labels)

        test_df = df[df['SubjectID'].isin(test_subjects)]
        X_test = test_df.iloc[:, 38:].values
        y_test = np.array(test_labels)        

        total_features = X_train.shape[1]
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

        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(pipeline, param_grid, scoring=score, cv=cv_inner, refit=True, n_jobs=-1)
        
        # execute search
        result = search.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        
        # evaluate model on the hold out dataset
        y_pred = best_model.predict(X_test)
        y_score  = best_model.predict_proba(X_test)[:, -1]
        
        # evaluate the model
        auroc = roc_auc_score(y_true=y_test, y_score=y_score)
        mcc = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
        
        # store the result
        outer_results_auroc.append(auroc)
        outer_results_mcc.append(mcc)
    
    # summarize the estimated performance of the model
    print(outer_results_auroc)
    print(f"AUROC | mean: {np.mean(outer_results_auroc)} | std: {np.std(outer_results_auroc)}")
    print(outer_results_mcc)
    print(f"MCC   | mean: {np.mean(outer_results_mcc)} | std: {np.std(outer_results_mcc)}")    

if __name__ == "__main__":

    # for task in ["sarcoma_t1_grading_binary", "sarcoma_t2_grading_binary"]:
    for task in ["sarcoma_t1_grading_binary"]:
        # for model in ["rf", "svm"]:
        for model in ["rf"]:
            mlflow.set_experiment(task)
            mlflow.start_run()            
            extract_radiomics_features(task=task, params_path="./pyradiomics_mri_params.yaml")
            train_classifier(task=task, model_type=model, score="roc_auc")
            mlflow.end_run()
