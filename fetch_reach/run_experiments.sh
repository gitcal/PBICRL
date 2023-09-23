#!/bin/sh

python3 main_pbirl_w_rect.py --num_steps=400000 --N_demonstrations=60 --norm_flag=1 --iterations=5 --m1=0.0 --m2=0.0 --m3=0.0 
python3 main_pbirl_w_rect.py --num_steps=400000 --N_demonstrations=60 --norm_flag=1 --iterations=5 --m1=1.0 --m2=1.5 --m3=2.5 
python3 main_pbirl_w_rect_IRL.py --num_steps=400000 --N_demonstrations=60 --norm_flag=1 --iterations=5


