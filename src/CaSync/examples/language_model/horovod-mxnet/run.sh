
for i in $(seq 1 3); do 
    ./resource_usage.py --numprocess 16 --servers egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu2:1,egpu9:1,egpu10:1,egpu11:1,egpu1:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1 --numbatches 100 --extra "_dgc_thr_1MB" --horovodrun --comp-threshold 262143 --comp-alg "dgc"
    ./resource_usage.py --numprocess 16 --servers egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu2:1,egpu9:1,egpu10:1,egpu11:1,egpu1:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1 --numbatches 100 --extra "_graddrop_1MB" --horovodrun --comp-threshold 262143 --comp-alg "graddrop"
    ./resource_usage.py --numprocess 16 --servers egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu2:1,egpu9:1,egpu10:1,egpu11:1,egpu1:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1 --numbatches 100 --extra "_terngrad_1MB" --horovodrun --comp-threshold 262143 --comp-alg "terngrad"
    ./resource_usage.py --numprocess 16 --servers egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu2:1,egpu9:1,egpu10:1,egpu11:1,egpu1:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1 --numbatches 100 --extra "_ecq_1MB" --horovodrun --comp-threshold 262143 --comp-alg "ecq"
    ./resource_usage.py --numprocess 16 --servers egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu2:1,egpu9:1,egpu10:1,egpu11:1,egpu1:1,egpu13:1,egpu14:1,egpu15:1,egpu16:1,egpu17:1,egpu18:1 --numbatches 100 --extra "_tbq_1MB" --horovodrun --comp-threshold 262143 --comp-alg "tbq"
done
# for i in $(seq 1 3); do 
#     ./resource_usage.py --numprocess 8 --servers egpu3:1,egpu4:1,egpu5:1,egpu6:1,egpu7:1,egpu2:1,egpu9:1,egpu10:1 --numbatches 100 --extra "_ring_baseline" --horovodrun
# done
# for i in $(seq 1 3); do
#     ./resource_usage.py --numprocess 4 --servers egpu3:1,egpu4:1,egpu5:1,egpu6:1 --numbatches 100 --extra "_ring_baseline" --horovodrun
# done
# for i in $(seq 1 3); do
#     ./resource_usage.py --numprocess 2 --servers egpu3:1,egpu4:1 --numbatches 80 --extra "_ring_baseline" --horovodrun
# done