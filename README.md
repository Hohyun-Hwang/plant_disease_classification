# plant_disease_classification

----------------------

** Contributor    
JongHwan Park : bomebug15@ds.seoultech.ac.kr    
HoHyun Hwang : hhhwang94@ds.seoultech.ac.kr       
JuHee Han : fgtr153@ds.seoultech.ac.kr    

For Plant Diseases Classification , We use data from https://www.kaggle.com/vipoooool/new-plant-diseases-dataset    

We used two datasets, a small dataset consisting only of apple disease, and the entire dataset did not name it separately. And we compared the performance of the three models of plant disease classification. CNN Model is based on LeNet5, and other models is supposed by torch or other package(efficientnet is not supposed by pytorch official).    


If you want to train or demo on small dataset, try    
```
small_main.py, small_model.py, small_dataloader.py
```

If you want to use all dataset, try    

```
main.py, model.py, dataloader.py
```


## Requirements    
-------------------    
```
pip install torch, argparse, wandb(if you want to monitor on wandb), seaborn, tqdm, torchvision
pip install efficientnet_pytorch
```
---------------    
## How to Use    
-----------------    
```
python small_main.py or main.py --batch_size(int) --model[efficientnet,resnet, CNN] --device[cpu, gpu] --epoch(int) --mode[train,visualization]
```
--------------    

## Model Description    
------------    
```
python model_summary.py
```
-----------    
