### transformer
#python train.py --gpu 7 --dataset ns2d_4ball --use-normalizer 0 --component all --comment 0  --loss-name rel2 --epochs 500 --batch-size 4 --model-name CGPT --optimizer AdamW --weight-decay 0.00001   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 96 --n-layers 3 --n-head 1 --mlp-layers 3  --use-tb 1  2>&1 & sleep 20s
#python train.py --gpu 0 --dataset ns2d_4ball --use-normalizer 0 --component all --comment 0  --loss-name rel2 --epochs 500 --batch-size 4 --model-name CGPT --optimizer AdamW --weight-decay 0.00001   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 96 --n-layers 3 --n-head 1 --mlp-layers 3  --use-tb 0 # 2>&1 & sleep 20s

#python train_scatter.py --gpu 0 --dataset ns2d_4ball --use-normalizer 0 --component all --comment 0  --loss-name rel2 --epochs 500 --scatter-batch-size 8192 --model-name MLP --optimizer AdamW --weight-decay 0.00001   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1  2>&1 & sleep 20s
python train_scatter.py --gpu 1 --dataset ns2d_4ball --use-normalizer 0 --component all --comment 0  --loss-name rel2 --epochs 500 --scatter-batch-size 8192 --model-name MLP --optimizer AdamW --weight-decay 0.00001   --lr 0.001 --lr-method step  --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1  2>&1 & sleep 20s

