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
            --num_train_epochs 20 > res14_aste.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest14 \
            --model_name_or_path models/contraste_model_after_4_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res14_contraste_epoch4.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest14 \
            --model_name_or_path models/contraste_model_after_6_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res14_contraste_epoch6.txt \

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
            --num_train_epochs 20 > res14_contraste_epoch8.txt \

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
            --num_train_epochs 20 > res14_contraste_epoch12.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest14 \
            --model_name_or_path models/baseline_model_after_6_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res14_baseline_epoch6.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest14 \
            --model_name_or_path models/baseline_model_after_8_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res14_baseline_epoch8.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest14 \
            --model_name_or_path models/baseline_model_after_12_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res14_baseline_epoch12.txt \


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
            --num_train_epochs 20 > res15_aste.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest15 \
            --model_name_or_path models/contraste_model_after_4_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res15_contraste_epoch4.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest15 \
            --model_name_or_path models/contraste_model_after_6_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res15_contraste_epoch6.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest15 \
            --model_name_or_path models/contraste_model_after_8_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res15_contraste_epoch8.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest15 \
            --model_name_or_path models/contraste_model_after_12_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res15_contraste_epoch12.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest15 \
            --model_name_or_path models/baseline_model_after_6_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res15_baseline_epoch6.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest15 \
            --model_name_or_path models/baseline_model_after_8_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res15_baseline_epoch8.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest15 \
            --model_name_or_path models/baseline_model_after_12_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res15_baseline_epoch12.txt \


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
            --num_train_epochs 20 > res16_aste.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest16 \
            --model_name_or_path models/contraste_model_after_4_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res16_contraste_epoch4.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest16 \
            --model_name_or_path models/contraste_model_after_6_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res16_contraste_epoch6.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest16 \
            --model_name_or_path models/contraste_model_after_8_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res16_contraste_epoch8.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest16 \
            --model_name_or_path models/contraste_model_after_12_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res16_contraste_epoch12.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest16 \
            --model_name_or_path models/baseline_model_after_6_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res16_baseline_epoch6.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest16 \
            --model_name_or_path models/baseline_model_after_8_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res16_baseline_epoch8.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset rest16 \
            --model_name_or_path models/baseline_model_after_12_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > res16_baseline_epoch12.txt \


python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset lap14 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > lap14_aste.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset lap14 \
            --model_name_or_path models/contraste_model_after_4_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > lap14_contraste_epoch4.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset lap14 \
            --model_name_or_path models/contraste_model_after_6_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > lap14_contraste_epoch6.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset lap14 \
            --model_name_or_path models/contraste_model_after_8_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > lap14_contraste_epoch8.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset lap14 \
            --model_name_or_path models/contraste_model_after_12_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > lap14_contraste_epoch12.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset lap14 \
            --model_name_or_path models/baseline_model_after_6_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > lap14_baseline_epoch6.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset lap14 \
            --model_name_or_path models/baseline_model_after_8_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > lap14_baseline_epoch8.txt \

python main_ckpt1.py --task aste \
            --target_mode temp \
            --dataset lap14 \
            --model_name_or_path models/baseline_model_after_12_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > lap14_baseline_epoch12.txt \