### The input for `SeCoPa`
Lines of gradients name and gradient size in number of float32 in an `input.txt` file, for example:
```
vgg0_dense0_weight,102760448
```

### The output for `SeCoPa`
Lines of gradients name, compressing or not and the number of partitions in an `SeCoPaPlan.txt` file, for example:
```
vgg0_dense0_weight,y,16
```

### Notes
We observed that $T_{send}$, $T_{enc}$ and $T_{dec}$ are all in a mode of $f(x)=A+B\times x$, so we measured in our local cluster and fit curves with the least square method to obtain the $A$ and $B$ respectively.

<font color='red'> We don't provide the profiling tools, one need to measure and fitting curves in different cluster and configurations. </font>