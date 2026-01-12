declare -A model_weights=(
    ["brain"]="/playpen-raid2/lin.tian/projects/icon_lung/ICON/network_weights/brain_model/brain_model_weights.trch"
    ["lung"]="/playpen-raid2/lin.tian/projects/icon_lung/ICON/network_weights/lung_model/lung_model_weights.trch"
    ["knee"]="/playpen-raid2/lin.tian/projects/icon_lung/ICON/network_weights/knee_model/knee_model_weights.trch"
    ["unigradicon"]="/playpen-raid2/lin.tian/projects/uniGradICON/pretrained_unigradicon/Step_2_final.trch"
    ["unigradicon(three_dataset)"]="/playpen-raid2/lin.tian/projects/uniGradICON/pretrained_unigradicon/on_threedatasets/Step_2_final.trch"
)


for model in "${!model_weights[@]}"; do
    echo "Evaluating model $model on COPDGene"
    python /playpen-raid2/lin.tian/projects/uniGradICON/evaluations/generalization/scripts/COPDGene_eval.py --weights_path "${model_weights[$model]}"  --io_steps 0 --device 0 --exp generalization_test_v2/"$model"_model/eval_on_COPDGene_wo_IO --model_type "$model"

    echo "Evaluating model $model on abdomen"
    python /playpen-raid2/lin.tian/projects/uniGradICON/evaluations/generalization/scripts/l2r_abdomen_eval.py --weights_path "${model_weights[$model]}"  --io_steps 0 --device 0 --data_folder /playpen-raid2/lin.tian/data/learn2reg/L2R_Task3_AbdominalCT --exp generalization_test_v2/"$model"_model/eval_on_l2r_abdomenCTCT_wo_IO --model_type "$model"

    echo "Evaluating model $model on HCP"
    python /playpen-raid2/lin.tian/projects/uniGradICON/evaluations/generalization/scripts/HCP_eval.py --weights_path "${model_weights[$model]}"  --io_steps 0 --device 0 --exp generalization_test_v2/"$model"_model/eval_on_HCP_wo_IO --model_type "$model"

    echo "Evaluating model $model on OAI"
    python /playpen-raid2/lin.tian/projects/uniGradICON/evaluations/generalization/scripts/OAI_eval.py --weights_path "${model_weights[$model]}"  --io_steps 0 --device 0 --exp generalization_test_v2/"$model"_model/eval_on_OAI_wo_IO --model_type "$model"

    echo "Evaluating model $model on ACDC"
    python /playpen-raid2/lin.tian/projects/uniGradICON/evaluations/generalization/scripts/ACDC_eval.py --weights_path "${model_weights[$model]}"  --io_steps 0 --device 0 --exp generalization_test_v2/"$model"_model/eval_on_ACDC_wo_IO --model_type "$model" --bidirection 1
done

# for model in "${!model_weights[@]}"; do
#     # echo "Evaluating model $model on COPDGene"
#     # python /playpen-raid2/lin.tian/projects/uniGradICON/evaluations/generalization/scripts/COPDGene_eval.py --weights_path "${model_weights[$model]}"  --io_steps 50 --device 0 --exp generalization_test_v2/"$model"_model/eval_on_COPDGene_w_IO --model_type "$model"

#     echo "Evaluating model $model on abdomen"
#     python /playpen-raid2/lin.tian/projects/uniGradICON/evaluations/generalization/scripts/l2r_abdomen_eval.py --weights_path "${model_weights[$model]}"  --io_steps 50 --device 0 --data_folder /playpen-raid2/lin.tian/data/learn2reg/L2R_Task3_AbdominalCT --exp generalization_test_v2/"$model"_model/eval_on_l2r_abdomenCTCT_w_IO

#     # echo "Evaluating model $model on HCP"
#     # python /playpen-raid2/lin.tian/projects/uniGradICON/evaluations/generalization/scripts/HCP_eval.py --weights_path "${model_weights[$model]}"  --io_steps 50 --device 0 --exp generalization_test_v2/"$model"_model/eval_on_HCP_w_IO --model_type "$model"

#     # echo "Evaluating model $model on OAI"
#     # python /playpen-raid2/lin.tian/projects/uniGradICON/evaluations/generalization/scripts/OAI_eval.py --weights_path "${model_weights[$model]}"  --io_steps 50 --device 0 --exp generalization_test_v2/"$model"_model/eval_on_OAI_w_IO --model_type "$model"
# done
