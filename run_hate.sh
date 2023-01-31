python main_ckpt1.py --task hateXplain \
            --target_mode para \
            --dataset hateXplain \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 10 > hateXplain_para.txt