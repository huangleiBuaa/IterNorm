#!/bin/bash

methods=(wr_BN wr_IterNorm_full)
depths=(28)
hNumbers=(48)
groups=(64)
seeds=(1)
batchSize=128
learningRate=0.1
weightDecay=0.0005
widen_factor=10
dr=0.3
nN=1
nIter=5
Count=0
maxEpoch=200
eStep="{60,120,160}"

l=${#methods[@]}
n=${#depths[@]}
m=${#hNumbers[@]}
t=${#groups[@]}
f=${#seeds[@]}

for ((a=0;a<$l;++a))
do 
   for ((i=0;i<$n;++i))
   do 
      for ((j=0;j<$m;++j))
      do	
        for ((k=0;k<$t;++k))
        do
          for ((b=0;b<$f;++b))
          do
        	#echo "methods=${methods[$a]}"
        	#echo "depths=${depths[$i]}"
        	#echo "hNumbers=${hNumbers[$j]}"
        	#echo "groups=${groups[$k]}"
   	        #echo "seed=${seeds[$b]}"
                
        CUDA_VISIBLE_DEVICES=${Count} th exp_6CVPR_Cifar10_NoW.lua -model ${methods[$a]} -depth ${depths[$i]} -hidden_number ${hNumbers[$j]} -m_perGroup ${groups[$k]} -seed ${seeds[$b]} -batchSize ${batchSize} -learningRate ${learningRate} -weightDecay ${weightDecay} -widen_factor ${widen_factor} -nIter ${nIter}  -dropout ${dr} -noNesterov ${nN} -epoch_step ${eStep} -max_epoch ${maxEpoch}

           done
         done
      done
   done
done
