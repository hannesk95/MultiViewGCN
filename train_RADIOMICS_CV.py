from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from glob import glob
from radiomics import featureextractor
import os
from tqdm import tqdm
import mlflow

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


def train_classifier(task, model_type, score, current_fold):

    radiomics_csv_path = f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/radiomics_cache_dir/radiomics_features_{task}.csv"
    df = pd.read_csv(radiomics_csv_path)
    df_metadata = pd.read_csv("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/patient_metadata.csv")

    subjects = df["SubjectID"]
    subjects = [subject[:6] if subject.startswith("Sar") else subject[:4] for subject in subjects]

    labels = []
    for subject in subjects:
        grading = df_metadata[df_metadata["ID"] == subject].Grading.item()
        labels.append(0 if grading == 1 else 1)

    df["Label"] = labels

    X = df.iloc[:, 38:-1].values
    y = df.iloc[:, -1].values

    # cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auroc_list = list()
    mcc_list = list()
    f1_list = list()
    bacc_list = list()

    for fold in range(5):

        if fold != current_fold:
            continue

        print(f"Fold {fold}")   

        folds_dict = torch.load(f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/{task}_folds.pt", weights_only=False)
        train_subjects = folds_dict[fold]["train_subjects"]
        test_subjects = folds_dict[fold]["test_subjects"]

        if task == "sarcoma_t1_grading_binary":
            df["SubjectID"] = df["SubjectID"].str.replace("T1", "", regex=True)
            df["SubjectID"] = df["SubjectID"].str.replace("_updated_T1", "", regex=True)
            df["SubjectID"] = df["SubjectID"].str.replace("_updated", "", regex=True)

        elif task == "sarcoma_t2_grading_binary":
            df["SubjectID"] = df["SubjectID"].str.replace("STIR_ax", "", regex=True)
            df["SubjectID"] = df["SubjectID"].str.replace("STIR_sag", "", regex=True)
            df["SubjectID"] = df["SubjectID"].str.replace("_updatedSTIR", "", regex=True)
            df["SubjectID"] = df["SubjectID"].str.replace("STIR", "", regex=True)

        train_df = df[df['SubjectID'].isin(train_subjects)]
        test_df = df[df['SubjectID'].isin(test_subjects)]

        X_train = train_df.iloc[:, 38:-1].values
        y_train = np.array(train_df["Label"])
        X_test = test_df.iloc[:, 38:-1].values
        y_test = np.array(test_df["Label"])

        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        total_features = X_train.shape[1]
        n_components_list = [int(total_features * i) for i in np.arange(0.1, 1.1, 0.1)]    
        
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
        
        search = GridSearchCV(pipeline, param_grid, scoring=score, cv=cv_inner, refit=True)   
        result = search.fit(X_train, y_train)    
        best_model = result.best_estimator_   
        
        yhat = best_model.predict(X_test)
        y_score  = best_model.predict_proba(X_test)[:, -1]
        
        bacc = balanced_accuracy_score(y_test, yhat)
        auroc = roc_auc_score(y_true=y_test, y_score=y_score)
        mcc = matthews_corrcoef(y_test, yhat)
        f1 = f1_score(y_test, yhat)

        # Log metrics to MLflow
        mlflow.log_metric("test_bacc", bacc)
        mlflow.log_metric("test_auc", auroc)   
        mlflow.log_metric("test_mcc", mcc)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_param("architecture", model_type)        
        mlflow.log_param("fold", fold)

        bacc_list.append(bacc)
        auroc_list.append(auroc)
        mcc_list.append(mcc)
        f1_list.append(f1)
        
    # summarize the estimated performance of the model
    print('BACC: %.3f (%.3f)' % (np.mean(bacc_list), np.std(bacc_list)))
    print('AUROC: %.3f (%.3f)' % (np.mean(auroc_list), np.std(auroc_list)))
    print('MCC: %.3f (%.3f)' % (np.mean(mcc_list), np.std(mcc_list)))
    print('F1: %.3f (%.3f)' % (np.mean(f1_list), np.std(f1_list)))


if __name__ == "__main__":

    for task in ["sarcoma_t1_grading_binary", "sarcoma_t2_grading_binary"]:
        for model in ["rf", "svm"]:
            for fold in range(5):  
                mlflow.set_experiment(task)
                mlflow.start_run()            
                extract_radiomics_features(task=task, params_path="./pyradiomics_mri_params.yaml")
                train_classifier(task=task, model_type=model, score="roc_auc", current_fold=fold)
                mlflow.end_run()
    