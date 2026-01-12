# Evaluation of uniGradICON

## Installation
- Install unigradicon
- Install surface-distance from https://github.com/google-deepmind/surface-distance.git
- Install Pandas
- Install Nibabel

## Download the Weight
Download the weight from https://github.com/uncbiag/uniGradICON/releases/tag/unigradicon_weights to evaluations/network_weights/unigradicon1.0/Step_2_final.trch

## Reproduce Steps
### COPDGene
Change the pathes to the image and landmark folder in COPDGene_eval.py and then run
```
python COPDGene_eval.py --weights_path ./network_weights/unigradicon1.0/Step_2_final.trch
```

### OAI
Change the path to the data_list in OAI_eval.py and then run
```
python OAI_eval.py --weights_path ./network_weights/unigradicon1.0/Step_2_final.trch
```

### HCP
Update the pathes in HCP_segs.py and then run
```
python HCP_eval.py --weights_path ./network_weights/unigradicon1.0/Step_2_final.trch
```

### Learn2Reg datasets
The learn2Reg evaluation scripts share the same interface
```
python l2r_[task]_eval.py --weights_path ./network_weights/unigradicon1.0/Step_2_final.trch --device 0 --exp [exp_name] --io_steps 0 --data_folder [path_to_dataset_folder]
```
The evaluation script will produce the ready-to-submit zip file under the evaluation_results/[exp_name] folder. You can submit the zip file to Learn2Reg website for evaluation. We include the metrics computed by Learn2Reg website in evaluations/evaluation_results/l2r_evaluation_results.

### IXI
Run
```
python IXI_eval.py --weights_path ./network_weights/unigradicon1.0/Step_2_final.trch --data_folder [path_to_dataset_folder]
```

### ACDCMM
Run
```
python ACDCMM_eval.py --weights_path ./network_weights/unigradicon1.0/Step_2_final.trch --bidirection 1 --ACDC_path [path_to_ACDC_dataset_folder] --MM_path [path_to_MM_dataset_folder] 
```
One can pass --ACDC_path, --MM_path, or both. 

## Our Evaluation Results
We also included the logs of our evaluation under evaluations/evaluations_results, exp_finetune, and exp_generalization.


