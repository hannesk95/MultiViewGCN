import torch
import mlflow
import numpy as np
from model import MLP
from torch.amp import GradScaler
from torch import autocast
import os

# from monai.data import PersistentDataset, DataLoader
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, f1_score
from utils import seed_everything, save_conda_yaml
from glob import glob
from utils import create_cv_splits
from dataset import MLPDataset

EPOCHS = 300
BATCH_SIZE = 16
ARCHITECTURE = "MLP"
WARMUP_EPOCHS = 100
INITIAL_LR = 0.0
TARGET_LR = 0.001
SEED = 42
FOLDS = 5
AGGREGATION = "mean"
VIEWS = 20

def main(fold, architecture, task, views, aggregation):

    ARCHITECTURE = architecture
    VIEWS = views
    AGGREGATION = aggregation
    
    # ----------------------------
    # Miscellaneous stuff
    # ----------------------------
    seed_everything(SEED)    
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mlflow.log_artifact("train_MLP_CV.py")
    mlflow.log_artifact("model.py")
    mlflow.log_artifact("dataset.py")
    mlflow.log_artifact("utils.py")
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("architecture", ARCHITECTURE)
    mlflow.log_param("seed", SEED)
    mlflow.log_param("warmup_epochs", WARMUP_EPOCHS)
    mlflow.log_param("initial_lr", INITIAL_LR)
    mlflow.log_param("target_lr", TARGET_LR)
    mlflow.log_param("fold", fold)
    mlflow.log_param("views", VIEWS)
    mlflow.log_param("aggregation", AGGREGATION)
    conda_yaml_path = save_conda_yaml()
    mlflow.log_artifact(conda_yaml_path)   

    # ----------------------------
    # Load data
    # ----------------------------

    create_cv_splits(task=task)

    match task:
        case "sarcoma_t1_grading_binary":
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/sarcoma_t1_grading_binary_folds.pt")
        case "sarcoma_t2_grading_binary":
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/sarcoma_t2_grading_binary_folds.pt")
        case "glioma_t1_grading_binary":
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_t1_grading_binary_folds.pt")
        case "glioma_t2_grading_binary":
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_t2_grading_binary_folds.pt")
        case "glioma_t1c_grading_binary":
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_t1c_grading_binary_folds.pt")
        case "glioma_flair_grading_binary":
            folds_dict = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_flair_grading_binary_folds.pt")
        case _:
            raise ValueError(f"Given task '{task}' unkown!")

    # ----------------------------
    # Start Cross Validation
    # ----------------------------

    for current_fold in range(FOLDS):

        if current_fold != fold:
            continue

        print(f"\nFold {current_fold}")

        current_fold_dict = folds_dict[current_fold]
        train_subjects = current_fold_dict["train_subjects"]
        train_labels = current_fold_dict["train_labels"]
        test_subjects = current_fold_dict["test_subjects"]
        test_labels = current_fold_dict["test_labels"]

        mlflow.log_param("train_subjects", train_subjects)
        mlflow.log_param("test_subjects", test_subjects)

        match task:
            case "sarcoma_t1_grading_binary":

                data = [file for file in glob(f"./data/sarcoma/*/T1/*graph-fibonacci-edge_attr_views{VIEWS}*.pt")]

                train_data = [file for file in data if any(subject in file for subject in train_subjects)]
                test_data = [file for file in data if any(subject in file for subject in test_subjects)]
        
                train_val_data = MLPDataset(train_data)
                test_data = MLPDataset(test_data)                

            case "sarcoma_t2_grading_binary":

                data = [file for file in glob(f"./data/sarcoma/*/T2/*graph-fibonacci-edge_attr_views{VIEWS}*.pt")]

                train_data = [file for file in data if any(subject in file for subject in train_subjects)]
                test_data = [file for file in data if any(subject in file for subject in test_subjects)]
        
                train_val_data = MLPDataset(train_data)
                test_data = MLPDataset(test_data)
            
            case "glioma_flair_grading_binary":

                data = [file for file in glob(f"./data/ucsf/glioma_four_sequences/*FLAIR_bias_graph-fibonacci-edge_attr_views{VIEWS}*.pt")]

                train_data = [file for file in data if any(subject in file for subject in train_subjects)]
                test_data = [file for file in data if any(subject in file for subject in test_subjects)]
        
                train_val_data = MLPDataset(train_data)
                test_data = MLPDataset(test_data)
            
            case "glioma_t1_grading_binary":

                data = [file for file in glob(f"./data/ucsf/glioma_four_sequences/*T1_bias_graph-fibonacci-edge_attr_views{VIEWS}*.pt")]

                train_data = [file for file in data if any(subject in file for subject in train_subjects)]
                test_data = [file for file in data if any(subject in file for subject in test_subjects)]
        
                train_val_data = MLPDataset(train_data)
                test_data = MLPDataset(test_data)
            
            case "glioma_t1c_grading_binary":

                data = [file for file in glob(f"./data/ucsf/glioma_four_sequences/*T1c_bias_graph-fibonacci-edge_attr_views{VIEWS}*.pt")]

                train_data = [file for file in data if any(subject in file for subject in train_subjects)]
                test_data = [file for file in data if any(subject in file for subject in test_subjects)]
        
                train_val_data = MLPDataset(train_data)
                test_data = MLPDataset(test_data)
            
            case "glioma_t2_grading_binary":

                data = [file for file in glob(f"./data/ucsf/glioma_four_sequences/*T2_bias_graph-fibonacci-edge_attr_views{VIEWS}*.pt")]

                train_data = [file for file in data if any(subject in file for subject in train_subjects)]
                test_data = [file for file in data if any(subject in file for subject in test_subjects)]
        
                train_val_data = MLPDataset(train_data)
                test_data = MLPDataset(test_data)

        # Further split train_val into training and validation (80/20 split)
        train_size = int(0.8 * len(train_val_data))
        val_size = len(train_val_data) - train_size
        train_data, val_data = torch.utils.data.random_split(train_val_data, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

        print(f"Number of train samples: {str(len(train_data)).zfill(4)}")
        print(f"Number of val samples:   {str(len(val_data)).zfill(4)}")
        print(f"Number of test samples:  {str(len(test_data)).zfill(4)}")

        # Create DataLoader
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        # Define the model, loss function, and optimizer
        model = MLP(num_classes=2, aggregation=AGGREGATION, n_views=VIEWS).to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {pytorch_total_params}")
        mlflow.log_param("num_trainable_params", pytorch_total_params)  
        train_labels = np.array(train_labels, dtype=np.int64)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels.flatten())
        class_weights = torch.tensor(class_weights).to(torch.float32).to(device)
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
            for (inputs, labels) in train_loader:
                
                #  = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()            

                with autocast(device_type='cuda', dtype=torch.float32):
                    output = model(inputs)
                    loss = loss_fn(output, labels)

                scaler.scale(loss).backward()            
                scaler.step(optimizer)            
                scaler.update()

                train_loss_list.append(loss.item())
                train_pred = torch.argmax(output, dim=1)
                train_true_list.append(labels.cpu().numpy())
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
                for (inputs, labels) in val_loader:
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    val_output = model(inputs)
                    val_loss = loss_fn(val_output, labels)

                    val_output = val_output.detach().cpu()

                    val_loss_list.append(val_loss.item())
                    val_pred = torch.argmax(val_output, dim=1)
                    val_true_list.append(labels.cpu().numpy())
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
                torch.save(model.state_dict(), "model_mcc.pth")
                mlflow.log_artifact("model_mcc.pth")
                torch.save(model.state_dict(), "model_auc.pth")
                mlflow.log_artifact("model_auc.pth")
                torch.save(model.state_dict(), "model_bacc.pth")
                mlflow.log_artifact("model_bacc.pth")
                torch.save(model.state_dict(), "model_f1.pth")
                mlflow.log_artifact("model_f1.pth")
                torch.save(model.state_dict(), "model_loss.pth")
                mlflow.log_artifact("model_loss.pth")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                torch.save(model.state_dict(), "model_loss.pth")
                mlflow.log_artifact("model_loss.pth")
                print(f"Best model saved at epoch {epoch+1} with loss {best_val_loss:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_f1_epoch = epoch
                torch.save(model.state_dict(), "model_f1.pth")
                mlflow.log_artifact("model_f1.pth")
                print(f"Best model saved at epoch {epoch+1} with F1 {best_val_f1:.4f}")
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                best_val_mcc_epoch = epoch
                torch.save(model.state_dict(), "model_mcc.pth")
                mlflow.log_artifact("model_mcc.pth")
                print(f"Best model saved at epoch {epoch+1} with MCC {best_val_mcc:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_auc_epoch = epoch
                torch.save(model.state_dict(), "model_auc.pth")
                mlflow.log_artifact("model_auc.pth")
                print(f"Best model saved at epoch {epoch+1} with AUC {best_val_auc:.4f}")
            if val_bacc > best_val_bacc:
                best_val_bacc = val_bacc
                best_val_bacc_epoch = epoch
                torch.save(model.state_dict(), "model_bacc.pth")
                mlflow.log_artifact("model_bacc.pth")
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
        model.load_state_dict(torch.load("model_auc.pth"))

        model.eval()
        test_loss_list = []
        test_true_list = []
        test_pred_list = []
        test_score_list = []
        with torch.no_grad():           
            for (inputs, labels) in test_loader:
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                test_output = model(inputs)
                test_loss = loss_fn(test_output, labels)

                test_loss_list.append(test_loss.item())
                test_pred = torch.argmax(test_output, dim=1)
                test_true_list.append(labels.cpu().numpy())
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

        os.remove("model_auc.pth")
        os.remove("model_bacc.pth")
        os.remove("model_f1.pth")
        os.remove("model_loss.pth")
        os.remove("model_mcc.pth")
        os.remove("conda.yaml")     

if __name__ == "__main__":    

    for views in [8, 12, 16, 20, 24]:
        for task in ["glioma_t1_grading_binary"]:#, "glioma_flair_grading_binary", "glioma_t1c_grading_binary", "glioma_t2_grading_binary"]:
        # for task in ["sarcoma_t1_grading_binary", "sarcoma_t2_grading_binary"]:
            for architecture in ["MLP"]:
                # for aggregation in ["mean", "max", "sum", "MLP"]:
                for aggregation in ["sum"]:
                    for fold in range(FOLDS):    
                        mlflow.set_experiment(task+"_MLP")
                        mlflow.start_run()    
                        main(fold, architecture, task, views, aggregation)
                        mlflow.end_run()
