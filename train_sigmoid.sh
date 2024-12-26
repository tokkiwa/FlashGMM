python3 train_sigmoid.py --cuda -d /root/shared_smurai/mlic_dataset\
    --N 192 --K 3 --lambda 0.03 --epochs 100 --learning-rate 1e-4 --lr_epoch 90 --batch-size 8 \
    --save_path /root/shared_smurai/Sigmoid_log/ --patch-size 256 256 \
    --kodak_path /root/shared_smurai/kodak\