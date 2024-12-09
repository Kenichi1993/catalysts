#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class My_Dataset(Dataset):
    def __init__(
        self, train_features,train_labels,material_dictionary, batch_size
    ):
        super().__init__()
        self.train_features = train_features
        self.train_labels = train_labels
        self.material_dict = material_dictionary
        self.batch_size = batch_size

    def __getitem__(self, idx):
        data_array = self.train_features.iloc[idx]
        Support = self.material_dict[data_array["Support"]]
        M1 = self.material_dict[data_array["M1"]] if data_array["M1"] != "none" else None
        M2 = self.material_dict[data_array["M2"]] if data_array["M2"] != "none" else None
        M3 = self.material_dict[data_array["M3"]] if data_array["M3"] != "none" else None
        Temperature = data_array["Temperature"]
        Pch4 = data_array["Pch4"]
        Po2 = data_array["Po2"]
        Par = data_array["Par"]
        target_label = self.train_labels[idx]
        
        sample = {
            "Support": Support,
            "M1": M1,
            "M2": M2,
            "M3": M3,
            "Temperature": Temperature,
            "Pch4": Pch4,
            "Po2": Po2,
            "Par": Par,
            "target_label": target_label
        }
        return sample

    def __len__(self) -> int:
        return len(self.train_features)

    def reprocess(self, data, idxs):
        Support = [data[idx]["Support"] for idx in idxs]
        M1 = [data[idx]["M1"] for idx in idxs]
        M2 = [data[idx]["M2"] for idx in idxs]
        M3 = [data[idx]["M3"] for idx in idxs]
        Temperature = [data[idx]["Temperature"] for idx in idxs]
        Pch4 = [data[idx]["Pch4"] for idx in idxs]
        Po2 = [data[idx]["Po2"] for idx in idxs]
        Par = [data[idx]["Par"] for idx in idxs]
        target_label = [data[idx]["target_label"] for idx in idxs]
        return(
            Support,
            M1,
            M2,
            M3,
            Temperature,
            Pch4,
            Po2,
            Par,
            target_label
        )

    def collate_fn(self, data):
        data_size = len(data)
        idx_arr = np.arange(data_size)
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        output =list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output

            

def load_as_dict(path):
    with open(path,"r") as f:
        data = [l.split(",") for l in f.read().splitlines()]
        data = data[1:]
    dict ={}
    for x in data:
        id = x[1]
        v_list = (float(x[2]),int(x[3]),int(x[4]),float(x[5]))
        dict[id] = v_list
    return dict

def preprocess(args):
    ##Load data
    train_data_path =args.data_path
    data = pd.read_csv(train_data_path)
    material_dict = load_as_dict(args.material_index_path)
    

    train_data = data.drop("Type", axis=1).drop("C2y", axis=1)
    train_features = train_data
    train_labels = data["C2y"].values
    train_dataset = My_Dataset(train_features,train_labels,material_dict, args.batch_size)

    in_dim = train_features.shape[1]
    return train_dataset, in_dim


class SimpleModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.amount_max=260
        self.amount_min=1
        
        self.electronegativity_max=2.8
        self.electronegativity_min=0.9

        self.temp_max=920
        self.temp_min=600

        self.sub_net_0 = nn.Sequential(
                nn.Linear(12, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 16))
        
        self.sub_net_1 = nn.Sequential(
                nn.Linear(num_features, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 16))
        self.sub_net_2 = nn.Sequential(
                nn.Linear(num_features*2, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 16))
        self.sub_net_3 = nn.Sequential(
                nn.Linear(num_features*3, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 16))
        self.sub_layer_2 = nn.Linear(num_features*2, 64) 
        self.sub_layer_3 = nn.Linear(num_features*3, 64) 
        self.layer_1 = nn.Linear(32, 8)
        self.layer_2 = nn.Linear(8, 4)
        self.layer_out = nn.Linear(4, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)

        self.family_embeddings = nn.Embedding(19, 3) ##埋め込みfor族
        self.period_embeddings = nn.Embedding(8, 3) ##埋め込みfor周期

    def forward(self, x):        
        temperature = torch.tensor(x[4]).unsqueeze(1).to(device)
        temperature = (temperature-600)/(900-600) #normalization
        pch4 = torch.tensor(x[5]).unsqueeze(1).to(device)
        po2 = torch.tensor(x[6]).unsqueeze(1).to(device)
        par = torch.tensor(x[7]).unsqueeze(1).to(device)
        infos = torch.cat((temperature,pch4,po2,par),axis=1)
        support = torch.stack([self.material_info_to_array(tmp) for tmp in x[0]])
        base_feats = torch.cat((support, infos), axis=1).to(torch.float32)
        base_feats = self.sub_net_0(base_feats)

        m1 = torch.stack([self.material_info_to_array(tmp) for tmp in x[1]])
        front_processed_tensors=[]
        for i in range(len(x[0])):
            if x[2][i] is None and x[3][i] is None:
                tmp = m1[i]#.unsqueeze(0)
                front_processed = self.sub_net_1(tmp)
                front_processed_tensors.append(front_processed)
            elif x[2][i] is not None and x[3][i] is None:
                
                m2 = self.material_info_to_array(x[2][i])
                concat_tensor = torch.cat((m1[i],m2))
                front_processed = self.sub_net_2(concat_tensor)
                front_processed_tensors.append(front_processed)

            elif x[2][i] is not None and x[3][i] is not None:
                
                m2 = self.material_info_to_array(x[2][i])
                m3 = self.material_info_to_array(x[3][i])
                concat_tensor = torch.cat((m1[i],m2,m3))
                front_processed = self.sub_net_3(concat_tensor)
                front_processed_tensors.append(front_processed)

        front_processed_tensors = torch.stack((front_processed_tensors))

        x = torch.cat((front_processed_tensors, base_feats),axis=1)
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        return x
    
    def material_info_to_array(self,x):
        normalized_amount = torch.tensor((x[0]-self.amount_min)/(self.amount_max-self.amount_min)).reshape(1).to(device)
        family_emb = self.family_embeddings(torch.tensor([x[1]]).to(device))
        period_emb = self.period_embeddings(torch.tensor([x[2]]).to(device))
        normalized_elec = torch.tensor((x[3]-self.electronegativity_min)/(self.electronegativity_max-self.electronegativity_min)).reshape(1).to(device)        
        embedding = torch.cat((normalized_amount,family_emb[0],period_emb[0],normalized_elec))
        return embedding




def train(args,train_dataset, in_dim):
    # for cross validation
    n_splits = 7  
    kf = KFold(n_splits=n_splits)

    for train_index, valid_index in kf.split(train_dataset):
        train_subset = Subset(train_dataset, train_index)
        valid_subset = Subset(train_dataset, valid_index)

        train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        valid_dataloader = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)

        model = SimpleModel(8).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        max_epochs = 30

        for epoch in range(max_epochs):
            for samples in train_dataloader:
                ## 下の2行抜くとバグります(原因調査中)
                if samples == []:
                    continue
                inputs = samples[0][:8]
                labels = torch.tensor(samples[0][8],dtype=torch.float32).unsqueeze(1).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()    
        predictions = []
        true_labels = []
        model.eval()
        with torch.no_grad():
            
            for samples in valid_dataloader:
                ## 下の2行抜くとバグります(原因調査中)                
                if samples == []:
                    continue
                inputs = samples[0][:8]
                labels = torch.tensor(samples[0][8],dtype=torch.float32).unsqueeze(1).to(device)
                outputs = model(inputs)
                predictions.extend(outputs.tolist())
                true_labels.extend(labels.tolist())
        valid_score = criterion(torch.tensor(predictions), torch.tensor(true_labels))
        print("valid score: ", valid_score.item())

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_path', default="./data/adb27910-d0e5-4a22-9415-580bf597035a.csv",help='data_path') # for all data
    parser.add_argument('--data_path', default="./data/241209_list_HTP.csv",help='data_path')
    parser.add_argument('--material_index_path', default="./data/data_sheet.csv",help='data_path')
    parser.add_argument('--batch_size', default=32,help='data_path')
    #parser.add_argument('--support_index_path', default="./data/support_id.txt",help='data_path')
    args = parser.parse_args() 
    tran_data, in_dim = preprocess(args)

    train(args,tran_data,in_dim)

if __name__ == '__main__':
    main()