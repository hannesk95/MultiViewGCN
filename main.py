import os
import sys
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from dataset import SarcomaDatasetCV, get_data
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef, f1_score
from model import PyGModel, ViewGNN, GNN, MLP
# from viewgcn_model import ViewGCN
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.backends import cudnn
from glob import glob
from utils import load_config, seed_everything, save_confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def train(loader, model, architecture, batch_size, criterion, optimizer, n_views):
    model.train()
    for data in loader:
        inputs = data[0].cuda()        
        labels = data[1].cuda()  

        if architecture == "MLP":
            out = model(inputs.x, batch_size) 
        else:
            out = model(inputs.x.to(torch.float32), inputs.edge_index.to(torch.long), inputs.batch)       
        
        loss = criterion(out, labels)
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad() 
        
def eval(loader, split, model, architecture, batch_size, criterion, n_views, result_dir):
    model.eval()

    loss_list = []
    true_list = []
    pred_list = []
    prob_list = []
    selected_nodes_list = []
    for data in loader: 
        inputs = data[0].cuda()        
        labels = data[1].cuda() 

        if architecture == "MLP":
            if split == "train":
                    out = model(inputs.x.to(torch.float32), batch_size)
            else:
                    out = model(inputs.x.to(torch.float32), 1)
        else:
            out = model(inputs.x.to(torch.float32), inputs.edge_index.to(torch.long), inputs.batch)         
               
        loss = criterion(out, labels)                
        pred = out.argmax(dim=1)
        
        loss_list.append(loss.item())
        pred_list.extend(pred.detach().cpu().numpy().tolist())    
        true_list.extend(labels.detach().cpu().numpy().tolist())
        prob_list.extend(torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy().tolist())       
     
    bacc = balanced_accuracy_score(y_true=true_list, y_pred=pred_list)
    auc = roc_auc_score(y_true=true_list, y_score=prob_list)
    mcc = matthews_corrcoef(y_true=true_list, y_pred=pred_list)
    f1 = f1_score(y_true=true_list, y_pred=pred_list,)
    loss = np.mean(loss_list)

    save_confusion_matrix(y_true=true_list, y_pred=pred_list, result_dir=result_dir, split=split)
    
    selected_nodes_list.extend(torch.ones(10))
    return bacc, auc, mcc, f1, loss, selected_nodes_list

def main(config):
 
    n_views = config["data"]["n_views"]
    dino_size = config["data"]["dino_size"]
    architecture = config["model"]["architecture"]
    readout = config["model"]["readout"]
    ratio = config["model"]["ratio"]
    dataset_name = config["data"]["dataset"]
    hidden_size = config["model"]["hidden_size"]
    n_folds = config["training"]["folds"]

    result_dir = f"./results/{dataset_name}_{n_folds}foldcv/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed_everything(config["training"]["seed"])
    cudnn.benchmark = True
    
    data, labels = get_data(datset_name=dataset_name, n_views=n_views, dino_size=dino_size)
    print(f"[INFO] Number of data samples: {len(data)}")

    dataset = SarcomaDatasetCV(data=data, labels=labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config["training"]["seed"]) 

    cv_test_bacc_list = []
    cv_test_auc_list = []
    cv_test_mcc_list = []
    cv_test_f1_list = []
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(data, labels)):
        print(f"\nFold {fold + 1}")

        # Split into train/val and test
        train_val_data = Subset(dataset, train_val_idx)
        test_data = Subset(dataset, test_idx)

        # Further split train_val into training and validation (80/20 split)
        train_size = int(0.8 * len(train_val_data))
        val_size = len(train_val_data) - train_size
        train_data, val_data = torch.utils.data.random_split(train_val_data, [train_size, val_size], generator=torch.Generator().manual_seed(config["training"]["seed"]))

        print(f"Number of train samples: {str(len(train_data)).zfill(4)}")
        print(f"Number of val samples:   {str(len(val_data)).zfill(4)}")
        print(f"Number of test samples:  {str(len(test_data)).zfill(4)}")

        # Get class weights for CE
        labels = np.array([graph[0].label for graph in train_data])
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels.flatten())
        class_weights = torch.tensor(class_weights).to(torch.float32).cuda()

        # Define dataloaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=False)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)

        assert architecture in ["MLP", "GCN", "GAT", "SAGE", "GIN"], "Given architecture is not implemented!"
        if architecture == "MLP":
            model = MLP(input_dim=data[0].x.shape[-1], hidden_dim=hidden_size, readout=readout).cuda()     
        else:
            model = GNN(input_dim=data[0].x.shape[-1], hidden_dim=hidden_size, readout=readout, ratio=ratio).cuda()
            # model = PyGModel(input_dim=data[0].x.shape[-1], hidden_dim=hidden_size, pool=pool, architecture=architecture).cuda() # PyGModel 
            # model = ViewGCN(name="ViewGCN", nclasses=2, num_views=24, hidden=128).cuda() # ViewGCN
            # model = ViewGNN(input_dim=data[0].x.shape[-1], hidden_dim=hidden_size, pool=pool, conv=architecture).cuda()   

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {pytorch_total_params}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode='max',           # Minimize the monitored metric (e.g., validation loss)
                                                               factor=0.9,           # Reduce LR by a factor of 0.1
                                                               patience=5,           # Wait for n epochs without improvement
                                                               threshold=1e-4,       # Minimum change to qualify as an improvement
                                                               verbose=True)          # Print messages when LR is reduced
                                                        

        train_loss_list = []
        val_loss_list = []
        train_auc_list = []
        val_auc_list = []
        train_bacc_list = []
        val_bacc_list = []
        train_mcc_list = []
        val_mcc_list = []
        train_f1_list = []
        val_f1_list = []
        lr_list = []
        best_metric = -1
        best_epoch = 0

        warmup_epochs = 100
        initial_lr = 0.000001  # Start LR during warm-up
        target_lr = 0.0001  # Final LR after warm-up
        
        for epoch in range(1, config["training"]["epochs"]+1):
            
            train(train_loader, model, architecture, config["training"]["batch_size"], criterion, optimizer, n_views)
            train_bacc, train_auc, train_mcc, train_f1, train_loss, _ = eval(train_loader, "train", model, architecture, config["training"]["batch_size"], criterion, n_views, result_dir)
            val_bacc, val_auc, val_mcc, val_f1, val_loss, _ = eval(val_loader, "val", model, architecture, config["training"]["batch_size"], criterion, n_views, result_dir)   
            
            # Warm-up phase
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs  # Linear warm-up
                lr = initial_lr + warmup_factor * (target_lr - initial_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                # Step the ReduceLROnPlateau scheduler after the warm-up phase
                scheduler.step(val_auc)
                lr = optimizer.param_groups[0]['lr']
      

            if epoch == 1:
                torch.save(model.state_dict(), f"./results/{dataset_name}_{n_folds}foldcv/fold{fold}_best_{architecture}_views{n_views}_ratio{ratio}_readout{readout}_dino{dino_size}_model.pt") 

            if val_auc > best_metric:
                best_metric = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), f"./results/{dataset_name}_{n_folds}foldcv/fold{fold}_best_{architecture}_views{n_views}_ratio{ratio}_readout{readout}_dino{dino_size}_model.pt") 

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)    
            train_auc_list.append(train_auc)
            val_auc_list.append(val_auc)
            train_bacc_list.append(train_bacc)
            val_bacc_list.append(val_bacc)
            train_mcc_list.append(train_mcc)
            val_mcc_list.append(val_mcc)
            train_f1_list.append(train_f1)
            val_f1_list.append(val_f1)
            lr_list.append(lr)

            plt.figure(figsize=(18, 18))
            plt.subplot(6, 1, 1)
            plt.plot(np.arange(len(train_loss_list)), train_loss_list, label="train")
            plt.plot(np.arange(len(val_loss_list)), val_loss_list, label="val")
            plt.title("Loss")
            plt.grid()
            plt.legend()

            plt.subplot(6, 1, 2)
            plt.plot(np.arange(len(train_auc_list)), train_auc_list, label="train")
            plt.plot(np.arange(len(val_auc_list)), val_auc_list, label="val")
            plt.title("AUROC")
            plt.grid()
            plt.legend()

            plt.subplot(6, 1, 3)
            plt.plot(np.arange(len(train_bacc_list)), train_bacc_list, label="train")
            plt.plot(np.arange(len(val_bacc_list)), val_bacc_list, label="val")
            plt.title("Balanced Accuracy")
            plt.grid()
            plt.legend()

            plt.subplot(6, 1, 4)
            plt.plot(np.arange(len(train_mcc_list)), train_mcc_list, label="train")
            plt.plot(np.arange(len(val_mcc_list)), val_mcc_list, label="val")
            plt.title("MCC")
            plt.grid()
            plt.legend()

            plt.subplot(6, 1, 5)
            plt.plot(np.arange(len(train_f1_list)), train_f1_list, label="train")
            plt.plot(np.arange(len(val_f1_list)), val_f1_list, label="val")
            plt.title("F1-Score")
            plt.grid()
            plt.legend()

            plt.subplot(6, 1, 6)
            plt.plot(np.arange(len(lr_list)), lr_list)
            plt.title("Learning Rate")
            plt.grid()

            plt.savefig(f"./results/{dataset_name}_{n_folds}foldcv/fold{fold}_performance_{architecture}_views{n_views}_ratio{ratio}_readout{readout}_dino{dino_size}.png")
            plt.close()

            print(f'Epoch: {epoch:03d} / {config["training"]["epochs"]}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {train_bacc:.4f}, Val Acc: {val_bacc:.4f}, AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

        # Testing
        print(f"Best model obtained at epoch {best_epoch} with performance of {best_metric}.")
        model.load_state_dict(torch.load(f"./results/{dataset_name}_{n_folds}foldcv/fold{fold}_best_{architecture}_views{n_views}_ratio{ratio}_readout{readout}_dino{dino_size}_model.pt"))
        test_bacc, test_auc, test_mcc, test_f1, test_loss, selected_nodes = eval(test_loader, "test", model, architecture, config["training"]["batch_size"], criterion, n_views, result_dir)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_bacc:.4f}, Test AUC: {test_auc:.4f}')
        cv_test_bacc_list.append(test_bacc)
        cv_test_auc_list.append(test_auc)
        cv_test_mcc_list.append(test_mcc)
        cv_test_f1_list.append(test_f1)

        save_path = f"./results/{dataset_name}_{n_folds}foldcv/fold{fold}_{architecture}_views{n_views}_ratio{ratio}_readout{readout}_dino{dino_size}_selected_nodes.pt"
        torch.save(selected_nodes, save_path)

    # Write final results to file
    with open(f"./results/{dataset_name}_{n_folds}foldcv/results_{architecture}_views{str(n_views).zfill(2)}_ratio{ratio}_readout{readout}_dino{dino_size}.txt", "w") as file:

        sys.stdout = file
        print(f"\nMean bacc {np.mean(cv_test_bacc_list):.4f}")
        print(f"Std bacc {np.std(cv_test_bacc_list):.4f}")
        
        print(f"Mean auc {np.mean(cv_test_auc_list):.4f}")
        print(f"Std auc {np.std(cv_test_auc_list):.4f}")

        print(f"Mean mcc {np.mean(cv_test_mcc_list):.4f}")
        print(f"Std mcc {np.std(cv_test_mcc_list):.4f}")

        print(f"Mean f1 {np.mean(cv_test_f1_list):.4f}")
        print(f"Std f1 {np.std(cv_test_f1_list):.4f}")
    sys.stdout = sys.__stdout__


if __name__ == "__main__":

    for dataset in ["nodule"]:                              # "sarcoma_t1", "sarcoma_t2", "headneck", "vessel", "adrenal", "synapse", "nodule"
        for architecture in ["GCN"]:                            # "GCN", "SAGE", "GAT", "MLP"
            for ratio in [0.25, 0.5, 0.75, 1.0]:                # 0.25, 0.5, 0.75, 1.0
                for readout in ["mean"]:                        # "sum", "mean"
                    for dino_size in ["small"]:                 # "small", "base", "large", "giant"
                        for n_views in [8, 12, 16, 20, 24]:     # 1, 3, 8, 12, 16, 20, 24

                            config = load_config(config_path="config.yaml")
                            config["data"]["dataset"] = dataset
                            config["model"]["architecture"] = architecture
                            config["model"]["readout"] = readout
                            config["model"]["ratio"] = ratio
                            config["data"]["dino_size"] = dino_size
                            config["data"]["n_views"] = n_views

                            main(config)

    
    # 00: sarcoma_t1 MLP
    # 01: sarcoma_t1 GCN
    # 02: sarcoma_t2 MLP
    # 03: sarcoma_t2 GCN
    # 04: headneck MLP
    # 05: headneck GCN
    # 06: vessel MLP
    # 07: vessel GCN
    # 08: adrenal MLP
    # 09: adrenal GCN
    # 10: synapse MLP
    # 11: synapse GCN
    # 12: nodule MLP
    # 13: nodule GCN
