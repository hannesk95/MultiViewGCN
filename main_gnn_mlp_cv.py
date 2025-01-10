import sys
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from dataset import SarcomaDataset, SarcomaDatasetCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef
from model import GCN, MLP, SAGE, GAT
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch_geometric import seed_everything
from torch.backends import cudnn
from _delete.model_paper import GCNHomConv
from glob import glob
import yaml

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train(loader, model, architecture, batch_size, criterion, optimizer):
    model.train()
    for data in loader:
        inputs = data[0].cuda()        
        labels = data[1].cuda()      
         
        match architecture:
            case "MLP":
                out = model(inputs.x, batch_size)
            # case "MLP_axial":
            #     out = model(inputs.x[::6, :], batch_size)               
            # case "MLP_acs":
            #     out = model(inputs.x[::2, :], batch_size)               
            case "GCN":               
                out, selected_nodes = model(inputs.x, inputs.edge_index.to(torch.long), inputs.batch)          
            case "SAGE":               
                out, selected_nodes = model(inputs.x, inputs.edge_index.to(torch.long), inputs.batch)          
            case "GAT":               
                out, selected_nodes = model(inputs.x, inputs.edge_index.to(torch.long), inputs.batch)          
            # case "GCN_fully":               
            #     out = model(inputs.x, inputs.edge_index, inputs.batch)          
            # case "GCNHomConv":             
            #     out = model(inputs)    
            #     out = out["graph_pred"]      
        loss = criterion(out, labels)  
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad() 
        

def eval(loader, split, model, architecture, batch_size, criterion):
    model.eval()

    loss_list = []
    true_list = []
    pred_list = []
    prob_list = []
    selected_nodes_list = []
    for data in loader: 
        inputs = data[0].cuda()        
        labels = data[1].cuda() 

        match architecture:           

            case "MLP":
                if split == "train":
                    out = model(inputs.x.to(torch.float32), batch_size)
                else:
                    out = model(inputs.x.to(torch.float32), 1)

            case "GCN":               
                out, selected_nodes = model(inputs.x.to(torch.float32), inputs.edge_index.to(torch.long), inputs.batch)
                selected_nodes_list.extend(selected_nodes.detach().cpu().numpy().tolist())
            
            case "SAGE":               
                out, selected_nodes = model(inputs.x.to(torch.float32), inputs.edge_index.to(torch.long), inputs.batch)
                selected_nodes_list.extend(selected_nodes.detach().cpu().numpy().tolist())
            
            case "GAT":               
                out, selected_nodes = model(inputs.x.to(torch.float32), inputs.edge_index.to(torch.long), inputs.batch)
                selected_nodes_list.extend(selected_nodes.detach().cpu().numpy().tolist())

            # case "MLP_axial":
            #     if split == "train":
            #         out = model(inputs.x[::6, :], batch_size)     
            #     else:
            #         out = model(inputs.x[::6, :], 1) 

            # case "MLP_acs":
            #     if split == "train":
            #         out = model(inputs.x[::2, :], batch_size)  
            #     else:
            #         out = model(inputs.x[::2, :], 1)              
            
            # case "GCN_fully":               
            #     out = model(inputs.x, inputs.edge_index, inputs.batch)
            
            # case "GCNHomConv":              
            #     out = model(inputs) 
            #     out = out["graph_pred"] 
        
        loss = criterion(out, labels)        
        pred = out.argmax(dim=1)
        
        loss_list.append(loss.item())
        pred_list.extend(pred.detach().cpu().numpy().tolist())    
        true_list.extend(labels.detach().cpu().numpy().tolist())
        prob_list.extend(torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy().tolist())       
     
    bacc = balanced_accuracy_score(y_true=true_list, y_pred=pred_list)
    auc = roc_auc_score(y_true=true_list, y_score=prob_list)
    mcc = matthews_corrcoef(y_true=true_list, y_pred=pred_list)
    loss = np.mean(loss_list)
    return bacc, auc, mcc, loss, selected_nodes_list

# epochs = 300
# batch_size = 16
# folds = 10
# seed = 42

def main(config):

    sequence = config["data"]["sequence"]
    n_views = config["data"]["n_views"]
    dino_size= config["data"]["dino_size"]
    fold = config["training"]["folds"]
    architecture = config["model"]["architecture"]
    pool = config["model"]["pool"]
    ratio = config["model"]["ratio"]


    # epochs = config["training"]["epochs"]
    # batch_size = config["training"]["batch_size"]
    # folds = config["training"]["folds"]
    # seed = config["training"]["seed"]

    # for dataset in ["sarcoma"]:                                 # "sarcoma", "headneck"
    #     for sequence in ["T1", "T2"]:                           # "T1", "T2"
    #         for architecture in ["MLP"]:                        # "GCN", "SAGE", "GAT", "MLP"
    #             for ratio in [1.00]:                            # 0.25, 0.50, 0.75, 1.00
    #                 for pool in ["sum", "mean"]:                # "sum", "mean"
    #                     for dino_size in ["small"]:             # "small", "base", "large", "giant"
    #                         for n_views in [1, 3, 6, 14, 26]:   # 1, 3, 6, 14 ,26, 42

    seed_everything(config["training"]["seed"])
    # cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True

    # data = []
    # data.extend(torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_{sequence}_train_dinov2-{dino_size}.pt", weights_only=False))
    # data.extend(torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_{sequence}_test_dinov2-{dino_size}.pt", weights_only=False))
    # labels = np.array([graph.label for graph in data])
    # labels = np.array([0 if item == 1 else 1 for item in labels])
    # labels = torch.from_numpy(labels).to(torch.long)
    
    if config["data"]["dataset"] == "sarcoma":        
        files = glob(f"/home/johannes/Code/MultiViewGCN/data/deep_learning/train/{sequence}/*graph_views{n_views}_dinov2-{dino_size}.pt")
        files.extend(glob(f"/home/johannes/Code/MultiViewGCN/data/deep_learning/test/{sequence}/*graph_views{n_views}_dinov2-{dino_size}.pt"))
        data = [torch.load(temp) for temp in files]
        labels = [graph.label.item() for graph in data]
        labels = [0 if item == 1 else 1 for item in labels]
        labels = torch.tensor(labels).to(torch.long)
    
    elif config["data"]["dataset"] == "headneck":
        raise NotImplementedError("Not yet implemented for head and neck cancer!")

    dataset = SarcomaDatasetCV(data=data, labels=labels)
    skf = StratifiedKFold(n_splits=config["training"]["folds"], shuffle=True, random_state=config["training"]["seed"]) 

    cv_test_bacc_list = []
    cv_test_auc_list = []
    cv_test_mcc_list = []
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(data, labels)):
        print(f"\nFold {fold + 1}")

        # Split into train/val and test
        train_val_data = Subset(dataset, train_val_idx)
        test_data = Subset(dataset, test_idx)

        # Further split train_val into training and validation (80/20 split)
        train_size = int(0.8 * len(train_val_data))
        val_size = len(train_val_data) - train_size
        train_data, val_data = torch.utils.data.random_split(train_val_data, [train_size, val_size], generator=torch.Generator().manual_seed(config["training"]["seed"]))

        # Define dataloaders
        train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        # Initialize Model
        match architecture:
            # case "MLP_axial":
            #     model = MLP(input_dim=data[0].x.shape[-1], pool=pool).cuda()        
            # case "MLP_acs":
            #     model = MLP(input_dim=data[0].x.shape[-1], pool=pool).cuda()        
            case "MLP":
                model = MLP(input_dim=data[0].x.shape[-1], pool=pool).cuda()        
            case "GCN":
                model = GCN(input_dim=data[0].x.shape[-1], pool=pool, prune=True, retention_ratio=ratio).cuda()
            case "SAGE":
                model = SAGE(input_dim=data[0].x.shape[-1], pool=pool, prune=True, retention_ratio=ratio).cuda()
            case "GAT":
                model = GAT(input_dim=data[0].x.shape[-1], pool=pool, prune=True, retention_ratio=ratio).cuda()
            # case "GCN_fully":
            #     model = GCN(input_dim=data[0].x.shape[-1], pool=pool, prune=True, retention_ratio=ratio, fully_connected_graph=True).cuda()
            # case "GCNHomConv":
            #     model = GCNHomConv(hidden_dim=64, total_number_of_gnn_layers=2, node_feature_dim=384, 
            #                     num_classes=2, retention_ratio=ratio).cuda()
            
            case _:
                raise ValueError("Given architecture is not available.")

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(pytorch_total_params)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        # criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.class_weights.cuda())
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)  

        train_loss_list = []
        val_loss_list = []
        train_auc_list = []
        val_auc_list = []
        train_bacc_list = []
        val_bacc_list = []
        train_mcc_list = []
        val_mcc_list = []
        lr_list = []
        best_metric = 0
        best_epoch = 0
        for epoch in range(1, config["training"]["epochs"]):
            lr = scheduler.get_last_lr()[0]
            train(train_loader, model, architecture, config["training"]["batch_size"], criterion, optimizer)
            train_bacc, train_auc, train_mcc, train_loss, _ = eval(train_loader, "train", model, architecture, config["training"]["batch_size"], criterion)
            val_bacc, val_auc, val_mcc, val_loss, _ = eval(val_loader, "val", model, architecture, config["training"]["batch_size"], criterion)   
            scheduler.step()

            if val_auc > best_metric:
                best_metric = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), f"./results/{sequence}_fold{fold}_best_{architecture}_views{n_views}_ratio{ratio}_pool{pool}_dino{dino_size}_model.pt") 

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)    
            train_auc_list.append(train_auc)
            val_auc_list.append(val_auc)
            train_bacc_list.append(train_bacc)
            val_bacc_list.append(val_bacc)
            train_mcc_list.append(train_mcc)
            val_mcc_list.append(val_mcc)
            lr_list.append(lr)

            plt.figure(figsize=(18, 10))
            plt.subplot(5, 1, 1)
            plt.plot(np.arange(len(train_loss_list)), train_loss_list, label="train")
            plt.plot(np.arange(len(val_loss_list)), val_loss_list, label="test")
            plt.title("Loss")
            plt.grid()
            plt.legend()

            plt.subplot(5, 1, 2)
            plt.plot(np.arange(len(train_auc_list)), train_auc_list, label="train")
            plt.plot(np.arange(len(val_auc_list)), val_auc_list, label="test")
            plt.title("AUROC")
            plt.grid()
            plt.legend()

            plt.subplot(5, 1, 3)
            plt.plot(np.arange(len(train_bacc_list)), train_bacc_list, label="train")
            plt.plot(np.arange(len(val_bacc_list)), val_bacc_list, label="test")
            plt.title("Balanced Accuracy")
            plt.grid()
            plt.legend()

            plt.subplot(5, 1, 4)
            plt.plot(np.arange(len(train_bacc_list)), train_bacc_list, label="train")
            plt.plot(np.arange(len(val_bacc_list)), val_bacc_list, label="test")
            plt.title("Balanced Accuracy")
            plt.grid()
            plt.legend()

            plt.subplot(5, 1, 5)
            plt.plot(np.arange(len(lr_list)), lr_list)
            plt.title("Learning Rate")
            plt.grid()

            plt.savefig(f"./results/{sequence}_fold{fold}_performance_{architecture}_views{n_views}_ratio{ratio}_pool{pool}_dino{dino_size}.png")
            plt.close()

            print(f'Epoch: {epoch:03d} / {config["training"]["epochs"]}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {train_bacc:.4f}, Val Acc: {val_bacc:.4f}, AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

        # Testing
        print(f"Best model obtained at epoch {best_epoch} with performance of {best_metric}.")
        model.load_state_dict(torch.load(f"./results/{sequence}_fold{fold}_best_{architecture}_views{n_views}_ratio{ratio}_pool{pool}_dino{dino_size}_model.pt"))
        test_bacc, test_auc, test_mcc, test_loss, selected_nodes = eval(test_loader, "test", model, architecture, config["training"]["batch_size"], criterion)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_bacc:.4f}, Test AUC: {test_auc:.4f}')
        cv_test_bacc_list.append(test_bacc)
        cv_test_auc_list.append(test_auc)
        cv_test_mcc_list.append(test_mcc)
        torch.save(selected_nodes, f"./results/{sequence}_fold{fold}_{architecture}_views{n_views}_ratio{ratio}_pool{pool}_dino{dino_size}_selected_nodes.pt")

    # Write final results to file
    with open(f"./results/{sequence}_results_{architecture}_views{n_views}_ratio{ratio}_pool{pool}_dino{dino_size}.txt", "w") as file:

        sys.stdout = file
        print(f"\nMean bacc {np.mean(cv_test_bacc_list):.4f}")
        print(f"Std bacc {np.std(cv_test_bacc_list):.4f}")
        print(f"Mean auc {np.mean(cv_test_auc_list):.4f}")
        print(f"Std auc {np.std(cv_test_auc_list):.4f}")
        print(f"Mean mcc {np.mean(cv_test_mcc_list):.4f}")
        print(f"Std mcc {np.std(cv_test_mcc_list):.4f}")
    sys.stdout = sys.__stdout__


if __name__ == "__main__":

    # epochs = config["training"]["epochs"]
    # batch_size = config["training"]["batch_size"]
    # folds = config["training"]["folds"]
    # seed = config["training"]["seed"]

    for dataset in ["sarcoma"]:                                 # "sarcoma", "headneck"
        for sequence in ["T1", "T2"]:                           # "T1", "T2"
            for architecture in ["MLP"]:                        # "GCN", "SAGE", "GAT", "MLP"
                for ratio in [1.00]:                            # 0.25, 0.50, 0.75, 1.00
                    for pool in ["sum", "mean"]:                # "sum", "mean"
                        for dino_size in ["small"]:             # "small", "base", "large", "giant"
                            for n_views in [1, 3, 6, 14, 26]:   # 1, 3, 6, 14 ,26, 42

                                config = load_config(config_path="config.yaml")
                                config["data"]["dataset"] = dataset
                                config["data"]["sequence"] = sequence
                                config["model"]["architecture"] = architecture
                                config["model"]["pool"] = pool
                                config["data"]["dino_size"] = dino_size
                                config["data"]["n_views"] = n_views
                                main(config)
