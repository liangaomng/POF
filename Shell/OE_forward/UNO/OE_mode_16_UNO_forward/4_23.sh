
# Array of alpha values
alpha_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Array of seed values
seed_values=(10 100 200)

# trunk_net
trunk_nets=(UNO) 
# Set other parameters
modes=16
base_dat="Data_Expr_OE/4_23_expr"
Train_epoch=11000
expr_order=1
Data_set="Data/Train/NCHD"

# Loop over each alpha value

for trunk_net in "${trunk_nets[@]}"
do
  echo "trunk_net=${trunk_net}"
  echo "当前日期和时间：$(date +"%Y-%m-%d %H:%M:%S")"
  for alpha in "${alpha_values[@]}"
  do
    # Loop over each seed value
    for seed in "${seed_values[@]}"
    do

      # Export parameters and run the Python script
    
      export alpha=$alpha
      export seed=$seed
      export modes=$modes
      export dat=$dat
      export Train_epoch=$Train_epoch
      export pid_en=no
      dat="${base_dat}_trunk_${trunk_net}_alpha_${alpha}_mode_${modes}seed_${seed}_pid_${pid_en}"

      echo "name = $dat"
      # Run the Python script with the current set of parameters
      #python Operator/Train_NCHD.py --alpha $alpha --modes $modes --seed $seed --dat $dat --epoch $Train_epoch --data_folder $Data_set
    
      python Operator/Train_NCHD.py  --Trunk $trunk_net --alpha $alpha --modes $modes --seed $seed --dat $dat --epoch $Train_epoch --data_folder $Data_set --pid $pid_en
      
      # Echo finish after each run
      echo "Finished with alpha=$alpha, seed=$seed"
      #序列
      # Increment expr_order after each alpha and seed combination
      ((expr_order++))
      echo $expr_order
    
    done
    
  done

done