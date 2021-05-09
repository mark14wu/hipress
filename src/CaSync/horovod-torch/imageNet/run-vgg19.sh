mkdir results
bash horovodrun.sh 16 egpu1:1,egpu2:1,egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu9:1,egpu10:1,egpu11:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1   32    1  0        16777216    vgg19      graddrop   > results/vgg19_graddrop_16node_b32.out
bash horovodrun.sh 16 egpu1:1,egpu2:1,egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu9:1,egpu10:1,egpu11:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1   32    1  0        1048576     vgg19      tbq        > results/vgg19_tbq_16node_b32.out
bash horovodrun.sh 16 egpu1:1,egpu2:1,egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu9:1,egpu10:1,egpu11:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1   32    1  0        16777216    vgg19      terngrad   > results/vgg19_terngrad_16node_b32.out
