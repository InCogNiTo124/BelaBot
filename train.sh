#!/usr/bin/env bash
for i in {141..199}; do
    last_model=$(ls -t *.pth | head -n 1);
    echo "Iter $i: $last_model";
    CUDA_VISIBLE_DEVICES=0 python3 main.py --from-checkpoint $last_model --save-as $last_model --epochs 50;
done
