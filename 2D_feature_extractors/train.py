import sys
sys.path.append("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/")
import mlflow
from utils import seed_everything, create_cv_splits, calculate_hidden_units
from model_FE import MLP
import torch
from glob import glob
from dataset_FE import FeatureExtractorDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import GradScaler
from torch import autocast
import os
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, f1_score
import uuid
import re

EPOCHS = 300
BATCH_SIZE = 16
WARMUP_EPOCHS = 100
INITIAL_LR = 0.0
TARGET_LR = 0.001
SEED = 42
FOLDS = 5
READOUT = "mean"

def train(task: str, method: str, fold: int, views: int ):

    identifier = str(uuid.uuid4())
    seed_everything(SEED)
    mlflow.log_param("task", task)
    mlflow.log_param("method", method)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("warmup_epochs", WARMUP_EPOCHS)
    mlflow.log_param("initial_lr", INITIAL_LR)
    mlflow.log_param("target_lr", TARGET_LR)
    mlflow.log_param("seed", SEED)
    mlflow.log_param("readout", READOUT)
    mlflow.log_param("folds", FOLDS)
    mlflow.log_param("views", views)

    match method:
        case "DINOv2":
            input_dim = 384 * views
        case _:
            raise ValueError(f"Given method '{method}' unknown!")    

    # create_cv_splits(task=task)

    match task:
        case "sarcoma_t1_grading_binary":
            data = [file for file in glob(f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/*/T1/*{method}_{str(views).zfill(2)}views_planar*.pt")]
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/sarcoma_t1_grading_binary_folds.pt")
            
        case "sarcoma_t2_grading_binary":
            data = [file for file in glob(f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/*/T2/*{method}_{str(views).zfill(2)}views_planar*.pt")]
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/sarcoma_t2_grading_binary_folds.pt")       
        
        case "glioma_t1c_grading_binary":
            data = [file for file in glob(f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*T1c*{method}_{str(views).zfill(2)}views_planar*.pt")]
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_t1c_grading_binary_folds.pt")
        
        case "glioma_flair_grading_binary":
            data = [file for file in glob(f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*FLAIR*{method}_{str(views).zfill(2)}views_planar*.pt")] 
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_flair_grading_binary_folds.pt")
        
        case "headneck_ct_hpv_binary":
            data = [file for file in glob(f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/headneck/converted_nii_merged/*/*{method}_{str(views).zfill(2)}views_planar*.pt")]
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/headneck/headneck_ct_hpv_binary_folds.pt")
        
        case "breast_mri_grading_binary":
            data = [file for file in glob(f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/breast/duke_tumor_grading/*{method}_{str(views).zfill(2)}views_planar*.pt")]
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/breast/breast_mri_grading_binary_folds.pt")
        
        case "kidney_ct_grading_binary":
            data = [file for file in glob(f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/kidney/converted_nii/*{method}_{str(views).zfill(2)}views_planar*.pt")]
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/kidney/kidney_ct_grading_binary_folds.pt")
        
        case "liver_ct_riskscore_binary":
            data = [file for file in glob(f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/liver/converted_nii/*{method}_{str(views).zfill(2)}views_planar*.pt")]
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/liver/liver_ct_riskscore_binary_folds.pt")
        
        case _:
            raise ValueError(f"Given task '{task}' unkown!")
        
    for current_fold in range(FOLDS):

        if current_fold != fold:
            continue

        print(f"\nFold {current_fold}")
        mlflow.log_param("fold", current_fold)

        current_fold_dict = folds_dict[current_fold]
        train_subjects = current_fold_dict["train_subjects"]
        train_labels = current_fold_dict["train_labels"]
        test_subjects = current_fold_dict["test_subjects"]
        test_labels = current_fold_dict["test_labels"]

        mlflow.log_param("train_subjects", train_subjects)
        mlflow.log_param("train_labels", train_labels)
        mlflow.log_param("test_subjects", test_subjects)
        mlflow.log_param("test_labels", test_labels)         

        if "glioma" in task:
                train_subjects = [re.sub(r'(\d+)$', lambda m: m.group(1).zfill(4), subject) for subject in train_subjects]  # Pad subject number with zeros to match the data filenames
                test_subjects = [re.sub(r'(\d+)$', lambda m: m.group(1).zfill(4), subject) for subject in test_subjects]  # Pad subject number with zeros to match the data filenames

        train_data = []
        train_labels_list = []
        test_data = []
        test_labels_list = []
        for subject in train_subjects:            

            item = [s for s in data if subject in s]
            # assert len(item) <2, f"Found multiple items for subject {subject} in data: {item}"
            index = [i for i, s in enumerate(train_subjects) if subject in s]
            # assert len(index) < 2, f"Found multiple indices for subject {subject} in data: {index}"

            train_data.append(item[0])
            train_labels_list.append(train_labels[index[0]])            
        
        for subject in test_subjects:

            item = [s for s in data if subject in s]
            # assert len(item) <2, f"Found multiple items for subject {subject} in data: {item}"
            index = [i for i, s in enumerate(test_subjects) if subject in s]
            # assert len(index) < 2, f"Found multiple indices for subject {subject} in data: {index}"

            test_data.append(item[0])
            test_labels_list.append(test_labels[index[0]])               

        train_val_data = FeatureExtractorDataset(train_data, train_labels_list)
        test_data = FeatureExtractorDataset(test_data, test_labels_list)  

        # Further split train_val into training and validation (80/20 split)
        train_size = int(0.8 * len(train_val_data))
        val_size = len(train_val_data) - train_size
        train_data, val_data = torch.utils.data.random_split(train_val_data, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

        print(f"Number of train samples: {str(len(train_data)).zfill(4)}")
        print(f"Number of val samples:   {str(len(val_data)).zfill(4)}")
        print(f"Number of test samples:  {str(len(test_data)).zfill(4)}")

        # Create DataLoader
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        num_hidden_units = calculate_hidden_units(output_dim_enc=input_dim)

        model = MLP(input_dim=input_dim, hidden_dim=num_hidden_units, num_layers=1, num_classes=2, readout=READOUT)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {pytorch_total_params}")
        mlflow.log_param("num_trainable_params", pytorch_total_params)

        train_labels = np.array(train_labels, dtype=np.int64)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels.flatten())
        class_weights = torch.tensor(class_weights).to(torch.float32)
        class_weights = class_weights.to("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-3)

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            mode='max',           # Minimize the monitored metric (e.g., validation loss)
                                                            factor=0.95,           # Reduce LR by a factor of 0.1
                                                            patience=5,           # Wait for n epochs without improvement
                                                            threshold=1e-4)       # Minimum change to qualify as an improvement                                    

        # Variables to track the best model
        best_val_loss = float("inf")
        best_val_loss_epoch = 0
        best_val_f1 = 0.0
        best_val_f1_epoch = 0
        best_val_mcc = 0.0
        best_val_mcc_epoch = 0
        best_val_auc = 0.0
        best_val_auc_epoch = 0
        best_val_bacc = 0.0
        best_val_bacc_epoch = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss_list = []
            train_true_list = []
            train_pred_list = []
            train_score_list = []
            for batch_data in train_loader:
                
                X = batch_data[0].to(torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
                X = torch.squeeze(X, dim=1)  # Remove the channel dimension if it exists
                y = batch_data[1].to(torch.long).to("cuda" if torch.cuda.is_available() else "cpu")
                
                optimizer.zero_grad()            

                with autocast(device_type='cuda', dtype=torch.float32):
                    output = model(X)
                    loss = loss_fn(output, y)

                scaler.scale(loss).backward()            
                scaler.step(optimizer)            
                scaler.update()

                train_loss_list.append(loss.item())
                train_pred = torch.argmax(output, dim=1)
                train_true_list.append(y.cpu().numpy())
                train_pred_list.append(train_pred.cpu().numpy())
                train_score_list.append(output[:,1].detach().cpu().numpy())                  

            train_loss = sum(train_loss_list) / len(train_loader)
            train_true = np.concatenate(train_true_list)
            train_pred = np.concatenate(train_pred_list)
            train_score = np.concatenate(train_score_list)
            train_f1 = f1_score(train_true, train_pred, average='weighted')
            train_mcc = matthews_corrcoef(train_true, train_pred)
            train_auc = roc_auc_score(train_true, train_score)
            train_bacc = balanced_accuracy_score(train_true, train_pred)
            print(f"[TRAINING]   Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}, F1: {train_f1:.4f}, MCC: {train_mcc:.4f}, AUC: {train_auc:.4f}, BACC: {train_bacc:.4f}")          

            model.eval()
            val_loss_list = []
            val_true_list = []
            val_pred_list = []
            val_score_list = []
            with torch.no_grad():           
                for val_data in val_loader:
                    
                    X_val = val_data[0].to(torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
                    X_val = torch.squeeze(X_val, dim=1)  # Remove the channel dimension if it exists
                    y_val = val_data[1].to(torch.long).to("cuda" if torch.cuda.is_available() else "cpu")

                    val_output = model(X_val)
                    loss = loss_fn(val_output, y_val)

                    val_output = val_output.detach().cpu()

                    val_loss_list.append(loss.item())
                    val_pred = torch.argmax(val_output, dim=1)
                    val_true_list.append(y_val.cpu().numpy())
                    val_pred_list.append(val_pred.cpu().numpy())
                    val_score_list.append(val_output[:,1].detach().cpu().numpy())
            val_loss = sum(val_loss_list) / len(val_loader)
            val_true = np.concatenate(val_true_list)
            val_pred = np.concatenate(val_pred_list)
            val_score = np.concatenate(val_score_list)
            val_f1 = f1_score(val_true, val_pred, average='weighted')
            val_mcc = matthews_corrcoef(val_true, val_pred)
            val_auc = roc_auc_score(val_true, val_score)
            val_bacc = balanced_accuracy_score(val_true, val_pred)               

            print(f"[VALIDATION] Epoch [{epoch+1}/{EPOCHS}], Loss: {val_loss:.4f}, F1: {val_f1:.4f}, MCC: {val_mcc:.4f}, AUC: {val_auc:.4f}, BACC: {val_bacc:.4f}")
            
            # Warm-up phase
            if epoch < WARMUP_EPOCHS:
                warmup_factor = (epoch + 1) / WARMUP_EPOCHS  # Linear warm-up
                lr = INITIAL_LR + warmup_factor * (TARGET_LR - INITIAL_LR)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                # Step the ReduceLROnPlateau scheduler after the warm-up phase
                scheduler.step(val_auc)
                lr = optimizer.param_groups[0]['lr']

            if epoch == 0:
                torch.save(model.state_dict(), f"model_mcc_{identifier}.pth")
                mlflow.log_artifact(f"model_mcc_{identifier}.pth")
                torch.save(model.state_dict(), f"model_auc_{identifier}.pth")
                mlflow.log_artifact(f"model_auc_{identifier}.pth")
                torch.save(model.state_dict(), f"model_bacc_{identifier}.pth")
                mlflow.log_artifact(f"model_bacc_{identifier}.pth")
                torch.save(model.state_dict(), f"model_f1_{identifier}.pth")
                mlflow.log_artifact(f"model_f1_{identifier}.pth")
                torch.save(model.state_dict(), f"model_loss_{identifier}.pth")
                mlflow.log_artifact(f"model_loss_{identifier}.pth")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                torch.save(model.state_dict(), f"model_loss_{identifier}.pth")
                mlflow.log_artifact(f"model_loss_{identifier}.pth")
                print(f"Best model saved at epoch {epoch+1} with loss {best_val_loss:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_f1_epoch = epoch
                torch.save(model.state_dict(), f"model_f1_{identifier}.pth")
                mlflow.log_artifact(f"model_f1_{identifier}.pth")
                print(f"Best model saved at epoch {epoch+1} with F1 {best_val_f1:.4f}")
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                best_val_mcc_epoch = epoch
                torch.save(model.state_dict(), f"model_mcc_{identifier}.pth")
                mlflow.log_artifact(f"model_mcc_{identifier}.pth")
                print(f"Best model saved at epoch {epoch+1} with MCC {best_val_mcc:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_auc_epoch = epoch
                torch.save(model.state_dict(), f"model_auc_{identifier}.pth")
                mlflow.log_artifact(f"model_auc_{identifier}.pth")
                print(f"Best model saved at epoch {epoch+1} with AUC {best_val_auc:.4f}")
            if val_bacc > best_val_bacc:
                best_val_bacc = val_bacc
                best_val_bacc_epoch = epoch
                torch.save(model.state_dict(), f"model_bacc_{identifier}.pth")
                mlflow.log_artifact(f"model_bacc_{identifier}.pth")
                print(f"Best model saved at epoch {epoch+1} with BACC {best_val_bacc:.4f}")       
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("train_mcc", train_mcc, step=epoch)
            mlflow.log_metric("train_auc", train_auc, step=epoch)
            mlflow.log_metric("train_bacc", train_bacc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            mlflow.log_metric("val_mcc", val_mcc, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)
            mlflow.log_metric("val_bacc", val_bacc, step=epoch)
            mlflow.log_metric("learning_rate", lr, step=epoch)
            torch.cuda.empty_cache()            

        # Log the best model parameters (only once)    
        mlflow.log_param("best_val_loss", best_val_loss)
        mlflow.log_param("best_val_loss_epoch", best_val_loss_epoch)
        mlflow.log_param("best_val_f1", best_val_f1)
        mlflow.log_param("best_val_f1_epoch", best_val_f1_epoch)
        mlflow.log_param("best_val_mcc", best_val_mcc)
        mlflow.log_param("best_val_mcc_epoch", best_val_mcc_epoch)
        mlflow.log_param("best_val_auc", best_val_auc)
        mlflow.log_param("best_val_auc_epoch", best_val_auc_epoch)
        mlflow.log_param("best_val_bacc", best_val_bacc)
        mlflow.log_param("best_val_bacc_epoch", best_val_bacc_epoch)        

        # Load the best model for evaluation  
        model.load_state_dict(torch.load(f"model_auc_{identifier}.pth"))

        model.eval()
        test_loss_list = []
        test_true_list = []
        test_pred_list = []
        test_score_list = []
        with torch.no_grad():           
            for test_data in test_loader:
                
                X_test = test_data[0].to(torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
                X_test = torch.squeeze(X_test, dim=1)  # Remove the channel dimension if it exists
                y_test = test_data[1].to(torch.long).to("cuda" if torch.cuda.is_available() else "cpu")

                test_output = model(X_test)
                loss = loss_fn(test_output, y_test)

                test_loss_list.append(loss.item())
                test_pred = torch.argmax(test_output, dim=1)
                test_true_list.append(y_test.cpu().numpy())
                test_pred_list.append(test_pred.cpu().numpy())
                test_score_list.append(test_output[:,1].detach().cpu().numpy())

        test_loss = sum(test_loss_list) / len(test_loader)
        test_true = np.concatenate(test_true_list)
        test_pred = np.concatenate(test_pred_list)
        test_score = np.concatenate(test_score_list)
        test_f1 = f1_score(test_true, test_pred, average='weighted')
        test_mcc = matthews_corrcoef(test_true, test_pred)
        test_auc = roc_auc_score(test_true, test_score)
        test_bacc = balanced_accuracy_score(test_true, test_pred)
        
        print(f"[TESTING]   Loss: {test_loss:.4f}, F1: {test_f1:.4f}, MCC: {test_mcc:.4f}, AUC: {test_auc:.4f}, BACC: {test_bacc:.4f}")
        
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_mcc", test_mcc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_bacc", test_bacc)   

        os.remove(f"model_auc_{identifier}.pth")
        os.remove(f"model_bacc_{identifier}.pth")
        os.remove(f"model_f1_{identifier}.pth")
        os.remove(f"model_loss_{identifier}.pth")
        os.remove(f"model_mcc_{identifier}.pth")


if __name__ == "__main__":
    
    for task in ["kidney_ct_grading_binary", "liver_ct_riskscore_binary",
                 "headneck_ct_hpv_binary", "breast_mri_grading_binary", 
                 "glioma_t1c_grading_binary", "glioma_flair_grading_binary", 
                 "sarcoma_t1_grading_binary", "sarcoma_t2_grading_binary"]:
        for method in ["DINOv2"]:
            for views in [1, 3, 8, 12, 16, 20, 24]:
                for fold in range(FOLDS):

                    mlflow.set_experiment(task+"_"+method+"_MLP_planar")
                    mlflow.start_run()    
                    train(task=task, method=method, fold=fold, views=views)
                    mlflow.end_run()

                