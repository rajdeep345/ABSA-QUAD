python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest14 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 \
            > aste_rest14.txt

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest14 \
            --model_name_or_path models/contraste_model_after_8_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 \
            > contraste_rest14_epoch8.txt

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest14 \
            --model_name_or_path models/contraste_model_after_12_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 \
            > contraste_rest14_epoch12.txt

'''
python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest15 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > aste_rest15.txt

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest16 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > aste_rest16.txt

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset lap15 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > aste_lap14.txt


python main_ckpt1.py --task aste \
            --target_mode para \
            --dataset rest14 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20

python main_ckpt1.py --task aste \
            --target_mode para \
            --dataset rest15 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20

python main_ckpt1.py --task aste \
            --target_mode para \
            --dataset rest16 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20

python main_ckpt1.py --task aste \
            --target_mode para \
            --dataset lap15 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20
'''