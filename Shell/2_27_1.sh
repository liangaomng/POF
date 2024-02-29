
export alpha=0.1
export modes=32
export seed=12
export dat="2_27_1"

export Train_epoch=2

python Operator/Train.py --alpha $alpha --modes $modes --seed $seed --dat $dat --epoch $Train_epoch 
echo "finish"