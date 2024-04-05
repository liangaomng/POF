
# Array of alpha values
alpha_values=(0.3 0.4)

# Array of seed values
seed_values=(10 100 200)

# Set other parameters
modes=256
base_dat="3_20_expr_"
Train_epoch=4000
expr_order=1

# Loop over each alpha value
for alpha in "${alpha_values[@]}"
do
  # Loop over each seed value
  for seed in "${seed_values[@]}"
  do

    # Export parameters and run the Python script
   
    dat="${base_dat}alpha_${alpha}_seed_${seed}"

    echo "name = $dat"
    export alpha=$alpha
    export seed=$seed
    export modes=$modes
    export dat=$dat
    export Train_epoch=$Train_epoch

    # Run the Python script with the current set of parameters
    python Operator/Train.py --alpha $alpha --modes $modes --seed $seed --dat $dat --epoch $Train_epoch 

    # Echo finish after each run
    echo "Finished with alpha=$alpha, seed=$seed"
    #序列
    # Increment expr_order after each alpha and seed combination
    ((expr_order++))
    echo $expr_order
   
  done
   
done
