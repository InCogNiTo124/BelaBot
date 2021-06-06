#!/usr/bin/env bash
for i in {201..500}; do
    echo "Iter: $i";
    last_model=$(ls -t models/*.pth | head -n 1);
    for j in {0..4}; do
        CUDA_VISIBLE_DEVICES=0 python3 main.py --from-checkpoint $last_model --save-as $last_model --epochs 50;
    done

    CUDA_VISIBLE_DEVICES=0 python3 main.py --test --from-checkpoint $last_model --no-save --epochs 100;
done
