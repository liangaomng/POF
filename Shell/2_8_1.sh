
export alpha=0.1
export modes=32
export seed=12
export dat="2_8_1"

export Train_epoch=1000
export Train_batch_size=20

python Operator/Train.py --alpha $alpha --modes $modes --seed $seed --dat $dat
echo "finish"