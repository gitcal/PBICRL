#!/bin/sh

python3 main_pbirl_w.py --num_steps=400000 --norm_flag=1 --N_demonstrations=60 --iterations=5 \
--fl_name='results_m10_m100' --data_fl_name='trajectories_m10_m100' --m1=0.0 --m2=0.0 --m3=0.0

python3 main_pbirl_w.py --num_steps=400000 --norm_flag=1 --N_demonstrations=60 --iterations=5 \
--fl_name='results_m10_m100' --data_fl_name='trajectories_m10_m100' --m1=0.0 --m2=2.9 --m3=6.0

python3 main_pbirl_w_IRL.py --num_steps=400000 --norm_flag=1 --N_demonstrations=60 --iterations=5 \
--fl_name='results_m10_m100' --data_fl_name='trajectories_m10_m100' --m1=0.0 --m2=0.0 --m3=0.0

