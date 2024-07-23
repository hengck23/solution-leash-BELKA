# Kaggle Competition Solution

# NeurIPS 2024 - Predict New Medicines with BELKA (5-th)
https://www.kaggle.com/competitions/leash-BELKA

For discussion, please refer to:  
[https://www.kaggle.com/competitions/leash-BELKA/discussion/456084](https://www.kaggle.com/competitions/leash-BELKA/discussion/521894)


## 1. Hardware  
- GPU: 2x Nvidia Ada A6000 (Ampere), each with VRAM 48 GB
- CPU: Intel® Xeon(R) w7-3455 CPU @ 2.5GHz, 24 cores, 48 threads
- Memory: 256 GB RAM

## 2. OS 
- ubuntu 22.04.4 LTS


## 3. Set Up Environment
- Install Python >=3.10.9
- Install requirements.txt in the python environment
- Set up the directory structure as shown below.
``` 
└── <solution_dir>
    ├── src 
    ├── result
    ├── data
    |   ├── processed
    |   |     ├──all_buildingblock.csv
    |   ├── kaggle 
    |         ├── leash-BELKA
    |               ├── sample_submission.csv
    │               ├── train.parquet
    │               ├── test.parquet
    ├── LICENSE 
    ├── README.md 
```
- Modify the path setting by editing  "/src/third_party/\_current_dir_.py"

```
# please use full path 
KAGGLE_DATA_DIR = '<solution_dir>/data/kaggle'
PROCESSED_DATA_DIR = '<solution_dir>/data/processed'
RESULT_DIR = '<solution_dir>/result'
```

- Download kaggle dataset "leash-BELKA" from:  
https://www.kaggle.com/competitions/leash-BELKA/data

- Create processed data by run the python script:
```
python "/src/process-data-01/run_make_data.py"  
```
  There are 98 millions molecules in the train data. Hence processing the data can take very long time.  
  Alternatively, you can download processed data from the share google drive at :  
  <google-drive>/leash-BELKA-solution/data/processed  
  https://drive.google.com/drive/folders/1bEBGtTJrQlYc_MQRYceBp0Kb9zGYue9H?usp=drive_link  

## 4. Training the model

### Warning !!! training output will be overwritten to the "/result" folder
Please run the following python scripts to learn the model files

```  
python "/src/cnn1d-nonshare-05-mean-layer5-bn/run_train.py"
output model:
- /result/cnn1d-mean-pool-ly5-bn-01/fold-0/checkpoint/00400000.pth
- /result/cnn1d-mean-pool-ly5-bn-01/fold-1/checkpoint/00550000.pth
- /result/cnn1d-mean-pool-ly5-bn-01/fold-3/checkpoint/00415000.pth

python "/src/transformer-fa-03/run_train.py"
output model:
- /result/transfomer-fa-03/fold-2/checkpoint/00264000.pth
- /result/transfomer-fa-03/fold-4/checkpoint/00264000.pth

python "/src/mamba-03/run_train.py"
output model:
- /result/mamba-03/checkpoint/00255000.pth

``` 

## 5. Submission csv 

Please run the following script:

```
python "/src/cnn1d-nonshare-05-mean-layer5-bn/run_submit.py"
python "/src/transformer-fa-03/run_submit.py"
python "/src/mamba-03/run_submit.py"
python "/src/run_ensemble.py"
output file:
- /result/final-3fold-tx2a-mamba-fix.submit.csv
```

![alt text](https://github.com/hengck23/solution-leash-BELKA/blob/main/doc/Selection_164.png)  


## 7. Reference trained models and validation results
- Reference results can also be found in the share google drive at :  
  <google-drive>/leash-BELKA-solution/result  
  https://drive.google.com/drive/folders/1bEBGtTJrQlYc_MQRYceBp0Kb9zGYue9H?usp=drive_link  

- It includes the weight files, train/validation logs.
  

## Authors

- https://www.kaggle.com/hengck23

## License

- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement

"We extend our thanks to HP for providing the Z8 Fury-G5 Data Science Workstation, which empowered our deep learning experiments. The high computational power and large GPU memory enabled us to design our models swiftly."
