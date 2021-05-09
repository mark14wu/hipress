#batch_size = 80
mkdir results
bash horovodrun.sh  16 egpu1:1,egpu2:1,egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu9:1,egpu10:1,egpu11:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1 80 5 0 4194304 tbq      > results/16node_tbq_80.out
bash horovodrun.sh  16 egpu1:1,egpu2:1,egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu9:1,egpu10:1,egpu11:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1 80 5 0 4194304 graddrop > results/16node_graddrop_80.out
bash horovodrun.sh  16 egpu1:1,egpu2:1,egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu9:1,egpu10:1,egpu11:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1 80 5 0 4194304 terngrad > results/16node_terngrad_80.out