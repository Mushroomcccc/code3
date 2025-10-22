
# output path
KR_FLAG='Pure'
mkdir -p output
mkdir -p output/Kuairand/
mkdir -p output/Kuairand/env
mkdir -p output/Kuairand/env/log

output_path=output/Kuairand/

# data source path
#  dataset/Kuairand-Pure\log_standard_4_08_to_4_21_pure.csv
 #'dataset/Kuairand-Pure/log_session_4_08_to_5_08_Pure.csv'
#C:\Users\Lenovo\Desktop\KuaiSim-main\code\dataset\kuairand-Pure
data_path=dataset/Kuairand/

MODEL='KRMBUserResponse'
# MODEL='KRMBUserResponseWithBias'

for LR in 0.0001 #0.0001  0.00001
do
    for REG in 0
    do
        for N_LAYER in 2
        do
        
            file_key=user_${MODEL}_lr${LR}_reg${REG}_nlayer${N_LAYER}
            
            python train_multibehavior.py\
                --epoch 10\
                --seed 619607\
                --lr ${LR}\
                --batch_size 512\
                --val_batch_size 512\
                --cuda 0\
                --reader KRMBSeqReader\
                --train_file ${data_path}log_session_4_08_to_5_08_${KR_FLAG}.csv\
                --user_meta_file ${data_path}user_features_${KR_FLAG}_fillna.csv\
                --item_meta_file ${data_path}video_features_basic_${KR_FLAG}_fillna.csv\
                --max_hist_seq_len 100\
                --data_separator ','\
                --meta_file_separator ','\
                --n_worker 0\
                --val_holdout_per_user 5\
                --test_holdout_per_user 5\
                --model ${MODEL}\
                --dataset 'Kuairand'\
                --loss 'bce'\
                --l2_coef ${REG}\
                --model_path ${output_path}env/${file_key}.model\
                --user_latent_dim 32\
                --item_latent_dim 32\
                --enc_dim 64\
                --n_ensemble 2\
                --attn_n_head 4\
                --transformer_d_forward 64\
                --transformer_n_layer ${N_LAYER}\
                --state_hidden_dims 128\
                --scorer_hidden_dims 128 32\
                --dropout_rate 0.1\
                > ${output_path}env/log/${file_key}.model.log
        done
    done
done



# MODEL='KRMBUserResponse_MaxOut'
# MODEL='KRMBUserResponse'

# for LR in 0.0001 # 0.00001 0.001
# do
#     for REG in 0.01 0
#     do
#         for N_LAYER in 2
#         do
#             python train_multibehavior.py\
#                 --epoch 10\
#                 --seed 619607\
#                 --lr ${LR}\
#                 --batch_size 128\
#                 --val_batch_size 128\
#                 --cuda 0\
#                 --reader KRMBSeqReader\
#                 --train_file ${data_path}log_session_4_08_to_5_08_${KR_FLAG}.csv\
#                 --user_meta_file ${data_path}user_features_${KR_FLAG}_fillna.csv\
#                 --item_meta_file ${data_path}video_features_basic_${KR_FLAG}_fillna.csv\
#                 --max_hist_seq_len 100\
#                 --data_separator ','\
#                 --meta_file_separator ','\
#                 --n_worker 4\
#                 --val_holdout_per_user 5\
#                 --test_holdout_per_user 5\
#                 --model ${MODEL}\
#                 --loss 'bce'\
#                 --l2_coef ${REG}\
#                 --model_path ${output_path}env/user_${MODEL}_lr${LR}_reg${REG}_nlayer${N_LAYER}.model\
#                 --user_latent_dim 32\
#                 --item_latent_dim 32\
#                 --enc_dim 64\
#                 --n_ensemble 2\
#                 --attn_n_head 4\
#                 --transformer_d_forward 64\
#                 --transformer_n_layer ${N_LAYER}\
#                 --state_hidden_dims 128\
#                 --scorer_hidden_dims 128 32\
#                 --dropout_rate 0.1\
#                 > ${output_path}env/log/user_${MODEL}_lr${LR}_reg${REG}_nlayer${N_LAYER}.model.log
#         done
#     done
# done