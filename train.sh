for i in {0..99}; do
    last_model=$(ls -c *.pth | head -n 1);
    echo "Iter $i: $last_model";
    CUDA_VISIBLE_DEVICES=0 python3 main.py --from-checkpoint $last_model;
done
