### transformer
#python train.py --gpu 7 --dataset ns2d_4ball --use-normalizer 0 --component all --comment 0  --loss-name rel2 --epochs 500 --batch-size 4 --model-name CGPT --optimizer AdamW --weight-decay 0.00001   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 96 --n-layers 3 --n-head 1 --mlp-layers 3  --use-tb 1  2>&1 & sleep 20s
#python train.py --gpu 0 --dataset ns2d_4ball --use-normalizer 0 --component all --comment 0  --loss-name rel2 --epochs 500 --batch-size 4 --model-name CGPT --optimizer AdamW --weight-decay 0.00001   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 96 --n-layers 3 --n-head 1 --mlp-layers 3  --use-tb 0 # 2>&1 & sleep 20s

#python train_scatter.py --gpu 0 --dataset ns2d_4ball --use-normalizer 0 --component all --comment 0  --loss-name rel2 --epochs 500 --scatter-batch-size 8192 --model-name MLP_s --optimizer AdamW --weight-decay 0.00001   --lr 0.001 --lr-method step --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s
#python train_scatter.py --gpu 0 --dataset ns2d_4ball --use-normalizer none --component all --comment 0           --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s
#python train_scatter.py --gpu 1 --dataset ns2d_4ball --use-normalizer 1 --component all --comment normalized  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s
#python train_scatter.py --gpu 2 --dataset ns2d_4ball --use-normalizer 0 --component all --comment noshuffle    --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s
#python train_scatter.py --gpu 5 --dataset ns2d_4ball --use-normalizer 1 --component all --comment 0           --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s
#python train_scatter.py --gpu 6 --dataset ns2d_4ball --use-normalizer 1 --component all --comment rel1           --loss-name rel1 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s


#python train.py --gpu 0 --dataset ns2d_4ball --use-normalizer 0 --component all --comment 0  --loss-name rel2 --epochs 500 --scatter-batch-size 8192 --model-name MLP --optimizer AdamW --weight-decay 0.00001   --lr 0.001 --lr-method step  --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1  2>&1 & sleep 20s
#python train.py --gpu 2 --dataset ns2d_4ball --use-normalizer 0 --component all --comment 0  --loss-name rel2 --epochs 500 --scatter-batch-size 8192 --model-name MLP --optimizer AdamW --weight-decay 0.00001   --lr 0.001 --lr-method cycle  --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1  2>&1 & sleep 20s
#python train.py --gpu 4 --dataset ns2d_4ball --use-normalizer 0 --component all --comment 0  --loss-name l2 --epochs 500 --scatter-batch-size 8192 --model-name MLP --optimizer AdamW --weight-decay 0.00001   --lr 0.001 --lr-method cycle  --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1  2>&1 & sleep 20s

#python train_scatter.py --gpu 3 --dataset ns2d_4ball --use-normalizer 1 --component all --comment 0           --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s
#python train_scatter.py --gpu 4 --dataset ns2d_4ball --use-normalizer 1 --component all --comment large           --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 256 --n-layers 10 --use-tb 1   2>&1 & sleep 20s



#python train_scatter.py --gpu 1 --dataset inductor2d --use-normalizer 1 --component all --comment 0           --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 150 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s
#python train_scatter.py --gpu 2 --dataset inductor2d --use-normalizer 1 --component all --comment 0           --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s
#python train_scatter.py --gpu 3 --dataset inductor2d --use-normalizer 1 --normalize_x 1 --component all --comment 0           --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s
#python train_scatter.py --gpu 4 --dataset inductor2d --use-normalizer 1 --normalize_x 0 --component all --comment 0           --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 256 --n-layers 10 --use-tb 1   2>&1 & sleep 20s
#python train_scatter.py --gpu 1 --dataset inductor2d --use-normalizer minmax --component all --comment 0           --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s

#python train_scatter.py --gpu 6 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component all --comment 0        --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 1 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment 0        --loss-name rel2 --epochs 500 --scatter-batch-size 8000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --grad-clip 1000.0  --attn-type linear  --n-hidden 256 --n-layers 10 --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 0 --dataset inductor2d --use-normalizer minmax  --normalize_x minmax --component 1 --comment minmax       --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s


#python train.py --gpu 6 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment 0        --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 256 --n-layers 10 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 7 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment origloss        --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s

### test fourier features
#python train.py --gpu 0 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment 0        --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0    --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 2 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment fourier  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 256 --sigma 4  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 4 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment fourier  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 256 --sigma 0.25  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 5 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment fourier  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.0001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 256 --sigma 0.25  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 6 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment 0        --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0    --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s

#python train.py --gpu 5 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment fourier  --loss-name rel1 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name FourierMLP --optimizer Adam --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 256 --sigma 0.1  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s

#python train_scatter.py --gpu 3 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment fourier  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name FourierMLP --optimizer Adam --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 256 --sigma 4  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s


#python train.py --gpu 0 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment rel2  --loss-name rel1 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 1 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment rel1  --loss-name rel2 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 2 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment l1  --loss-name l1 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 3 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment rel2  --loss-name rel2 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name CGPT --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 4 --dataset inductor2d --use-normalizer none  --normalize_x none --component 1 --comment nonenorm  --loss-name rel2 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name CGPT --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 5 --dataset inductor2d --use-normalizer none  --normalize_x none --component 1 --comment nonenorm  --loss-name rel2 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name CGPT --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 128 --n-layers 3 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 6 --dataset inductor2d --use-normalizer none  --normalize_x none --component 1 --comment nonenorm  --loss-name rel2 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name CGPT --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 128 --n-layers 3 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 7 --dataset inductor2d --use-normalizer unit  --normalize_x none --component 1 --comment nonenorm  --loss-name rel2 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name CGPT --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 128 --n-layers 3 --use-tb 1    2>&1 & sleep 20s

#python train.py --gpu 7 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component all --comment rel2  --loss-name rel2 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name MLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s


#python train_scatter.py --gpu 0 --dataset ns2d_4ball --use-normalizer none --component all --comment 0           --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step --lr-step-size 100 --grad-clip 1000.0  --attn-type linear  --n-hidden 128 --n-layers 5 --use-tb 1   2>&1 & sleep 20s

#python train_scatter.py --gpu 0 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component all --comment 0  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s        --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method step  --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0   --sigma 4  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 1 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component all --comment 0  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s        --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0   --sigma 4  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 2 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component 1 --comment 0  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 128 --sigma 4  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s

#python train.py --gpu 3 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component all --comment rel2  --loss-name rel2 --epochs 500 --batch-size 4 --scatter-batch-size 40000 --model-name MLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 128 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 4 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component all --comment 0  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s        --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0   --sigma 4  --n-hidden 256 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 4 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component all --comment 0  --loss-name rel2 --epochs 500 --scatter-batch-size 20000 --model-name MLP_s        --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0   --sigma 4  --n-hidden 256 --n-layers 5 --use-tb 1    2>&1 & sleep 20s
#python train.py --gpu 5 --dataset inductor2d --use-normalizer unit  --normalize_x minmax --component all --comment rel2  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name MLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 256 --n-layers 5 --use-tb 1    2>&1 & sleep 20s

#python train.py        --gpu 1 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name MLP   --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 256 --n-layers 10 --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 1 --dataset inductor2d_b --use-normalizer unit  --normalize_x minmax --component all --comment rel2  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 256 --n-layers 10  --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 2 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 0.1  --n-hidden 256 --n-layers 10  --use-tb 1    2>&1 & sleep 20s

#### fourier features on space
#python train_scatter.py --gpu 2 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 128 --sigma 8  --n-hidden 256 --n-layers 5  --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 0 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 128 --sigma 16  --n-hidden 256 --n-layers 5  --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 1 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 128 --sigma 16  --n-hidden 256 --n-layers 5  --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 1 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 128 --sigma 4  --n-hidden 256 --n-layers 5  --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 3 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 64 --sigma 4  --n-hidden 256 --n-layers 5  --use-tb 1    2>&1 & sleep 20s


#### example of accelerator process
#CUDA_VISIBLE_DEVICES="4,5" accelerate launch --multi_gpu --main_process_port 5005  train_scatter_parallel.py --gpu 3 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment multi  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 4  --n-hidden 256 --n-layers 5  --use-tb 1    2>&1 & sleep 20s
# test
#CUDA_VISIBLE_DEVICES="4,5" accelerate launch --multi_gpu --main_process_port 5005  train_scatter_parallel.py --gpu 3 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment multi  --loss-name rel2 --epochs 2000 --batch-size 4 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 4  --n-hidden 256 --n-layers 5  --use-tb 1    2>&1 & sleep 20s
#CUDA_VISIBLE_DEVICES="6"   accelerate launch --multi_gpu --main_process_port 5006  train_scatter_parallel.py --gpu 3 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment multi  --loss-name rel2 --epochs 2000 --batch-size 4 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 4  --n-hidden 256 --n-layers 5  --use-tb 1    2>&1 & sleep 20s


####### inductor3d small
#python train_scatter.py --gpu 1 --dataset inductor3d_A1 --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 0 --sigma 4  --n-hidden 256 --n-layers 5  --use-tb 1    2>&1 & sleep 20s
python train_scatter.py --gpu 2 --dataset inductor3d_A1 --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 500 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 128 --sigma 4 --type exp --n-hidden 384 --n-layers 6  --use-tb 1    2>&1 & sleep 20s

### train 2d with 30 examples
#python train_scatter.py --gpu 0 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment 30samples --train-num 30  --loss-name rel2 --epochs 250 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 128 --sigma 4  --n-hidden 256 --n-layers 10  --use-tb 1    2>&1 & sleep 20s
#python train_scatter.py --gpu 0 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment 30samples --train-num 30  --loss-name rel2 --epochs 250 --scatter-batch-size 40000 --model-name MLP_s --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 128 --sigma 4 --type exp  --n-hidden 256 --n-layers 6  --use-tb 1    2>&1 & sleep 20s


### train with exp fourier features
#python train_scatter.py --gpu 1 --dataset inductor2d_b --use-normalizer unit  --normalize_x unit --component all --comment exp  --loss-name rel2 --epochs 1000 --batch-size 4 --scatter-batch-size 40000 --model-name FourierMLP --optimizer AdamW --weight-decay 0.0   --lr 0.001 --lr-method cycle --lr-step-size 100 --grad-clip 1000.0 --hfourier-dim 128 --sigma 16 --type exp  --n-hidden 256 --n-layers 5  --use-tb 1    2>&1 & sleep 20s
