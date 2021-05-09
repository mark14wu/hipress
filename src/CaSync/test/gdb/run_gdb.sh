FILENAME='/home/togo/horovod/examples/.myPID'
PID=`cat $FILENAME`
while [ "$PID"x == x ]
do
    PID=`cat $FILENAME`
    echo PID=$PID
    sleep 1
done
echo PID=$PID
echo > $FILENAME
gdb -p $PID
