for((i=2; i<=2; i++))
do
{
    if (($i==8))
    then
        continue
    fi
    if (($i==12))
    then
        continue
    fi
    # echo $i
    # ret=`ssh egpu$i -t echo 1`
    # echo result of egpu$i is $ret
    for gzfile in anaconda3 horovod
    do
    {
        ssh egpu${i} -t rm -rf /home/togo/${gzfile}
        scp ~/${gzfile}.tar.gz egpu${i}:/home/togo/
        ssh egpu${i} -t tar -zxvf /home/togo/${gzfile}.tar.gz
    }
    done
}&
done
wait
echo done!