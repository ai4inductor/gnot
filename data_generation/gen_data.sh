#python generate_heatsink3d.py --init 0   --end 240  2>&1 & sleep 30s
#python generate_heatsink3d.py --init 240 --end 480  2>&1 & sleep 30s
#python generate_heatsink3d.py --init 480 --end 720  2>&1 & sleep 30s
#python generate_heatsink3d.py --init 720 --end 960  2>&1 & sleep 30s
#python generate_heatsink3d.py --init 960 --end 1200
/home/zhongkai/miniconda3/bin/python3.9 -u /home/zhongkai/files/ml4phys/tno/gnot/data_generation/generate_heatsink3d.py --init 0   --end 240
/home/zhongkai/miniconda3/bin/python3.9 -u /home/zhongkai/files/ml4phys/tno/gnot/data_generation/generate_heatsink3d.py --init 240 --end 480
/home/zhongkai/miniconda3/bin/python3.9 -u /home/zhongkai/files/ml4phys/tno/gnot/data_generation/generate_heatsink3d.py --init 480 --end 720
/home/zhongkai/miniconda3/bin/python3.9 -u /home/zhongkai/files/ml4phys/tno/gnot/data_generation/generate_heatsink3d.py --init 720 --end 960
/home/zhongkai/miniconda3/bin/python3.9 -u /home/zhongkai/files/ml4phys/tno/gnot/data_generation/generate_heatsink3d.py --init 960 --end 1200


echo "All tasks are finished"