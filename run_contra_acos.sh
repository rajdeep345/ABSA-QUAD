python main_ckpt1.py --task acos \
            --target_mode temp \
            --dataset rest \
            --model_name_or_path models/contraste_model_after_4_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > rest_contra_acos_epoch4.txt \

python main_ckpt1.py --task acos \
            --target_mode temp \
            --dataset rest \
            --model_name_or_path models/contraste_model_after_6_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > rest_contra_acos_epoch6.txt \

python main_ckpt1.py --task acos \
            --target_mode temp \
            --dataset rest \
            --model_name_or_path models/contraste_model_after_8_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > rest_contra_acos_epoch8.txt \

python main_ckpt1.py --task acos \
            --target_mode temp \
            --dataset rest \
            --model_name_or_path models/contraste_model_after_12_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > rest_contra_acos_epoch12.txt \

python main_ckpt1.py --task acos \
            --target_mode temp \
            --dataset laptop \
            --model_name_or_path models/contraste_model_after_4_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > laptop_contra_acos_epoch4.txt \

python main_ckpt1.py --task acos \
            --target_mode temp \
            --dataset laptop \
            --model_name_or_path models/contraste_model_after_6_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > laptop_contra_acos_epoch6.txt \

python main_ckpt1.py --task acos \
            --target_mode temp \
            --dataset laptop \
            --model_name_or_path models/contraste_model_after_8_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > laptop_contra_acos_epoch8.txt \

python main_ckpt1.py --task acos \
            --target_mode temp \
            --dataset laptop \
            --model_name_or_path models/contraste_model_after_12_epochs \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 > laptop_contra_acos_epoch12.txt \