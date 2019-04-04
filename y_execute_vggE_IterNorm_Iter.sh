#!/bin/bash

methods=(vggE_IterNorm)
depths=(56)
nIters=(0 1 3 5 7)
groups=(512)
seeds=(1)
batchSize=256
learningRate=0.1
weightDecay=0
widen_factor=1
learningRateDecayRatio=0.2
dr=0
nN=0
Count=0
maxEpoch=160
eStep="{60,120}"

l=${#methods[@]}
n=${#depths[@]}
m=${#nIters[@]}
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
        	echo "methods=${methods[$a]}"
        	echo "depths=${depths[$i]}"
        	echo "nIters=${nIters[$j]}"
        	echo "groups=${groups[$k]}"
   	      echo "seed=${seeds[$b]}"
                
                
          CUDA_VISIBLE_DEVICES=${Count} th exp_6CVPR_Cifar10_NoW.lua -model ${methods[$a]} -depth ${depths[$i]} -hidden_number 48 -nIter ${nIters[$j]} -m_perGroup ${groups[$k]} -seed ${seeds[$b]} -batchSize ${batchSize} -learningRate ${learningRate} -weightDecay ${weightDecay} -widen_factor ${widen_factor} -dropout ${dr} -noNesterov ${nN} -max_epoch ${maxEpoch} -learningRateDecayRatio ${learningRateDecayRatio} -epoch_step ${eStep}
             
           done
         done
      done
   done
done
