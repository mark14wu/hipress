## Pseudocode for system design

### construct response list
```c++
// responses is a double end queue
Response ar_response, ag_response, g_response, b_response;
while (!responses.empty()) {
    response = responses.front();
    responses.pop_front();
    if (response.response_type() == ALLREDUCE) {
        if (ar_response.tensor_names().size() == 0) { // the ar_response is null
            ar_response.set_response(response);
        } else { // the ar_response is filled with some responses
            if (can be fused) {
                // fusion operation
            } else {
                response_list.add_response(ar_response);
                ar_response.clear();
                ar_response.set_response(response);
            }
        }
    } else if (response.response_type() == ALLGATHER) {
        if (ag_response.tensor_names().size() == 0) {
            ag_response.set_response(response);
        } else {
            if (can be fused) {
                // fusion operation
            } else {
                response_list.add_response(ag_response);
                ag_response.clear();
                ag_response.set_response(response);
            }
        }
    } else if (response.response_type() == GATHER) {
        if (g_response.tensor_names().size() == 0) {
            g_response.set_response(response);
        } else {
            if (can be fused) {
                // fusion operation
            } else {
                response_list.add_response(g_response);
                g_response.clear();
                g_response.set_response(response);
            }
        }
    } else if (response.response_type() == BROADCAST) {
        if (b_response.tensor_names().size() == 0) {
            b_response.set_response(response);
        } else {
            if (can be fused) {
                // fusion operation
            } else {
                response_list.add_response(b_response);
                b_response.clear();
                b_response.set_response(response);
            }
        }
    } else { // error or other message
        response_list.add_response(response);
    }
}

if (ar_response.tensor_names().size() > 0) {
    response_list.add_response(ar_response);
}
if (ag_response.tensor_names().size() > 0) {
    response_list.add_response(ag_response);
}
if (g_response.tensor_names().size() > 0) {
    response_list.add_response(g_response);
}
if (b_response.tensor_names().size() > 0) {
    response_list.add_response(b_response);
}
```

### gather fusion buffer
```c++
// displcmnts, recvcounts, entry_component_sizes, entry_component_offsets
const void *send_data = nullptr;
void *recv_data = nullptr;
if (is_root) {
    allocate displcmnts, recvcounts, entry_component*;
    SetRecvcountsAndDisplacements(recvcounts, displcmnts);
    SetEntryComponentOffsets(entry_component_offsets);
    if (entries.size() > 1) {
        recv_data = fusion_buffer;
    } else {
        recv_data = (void *) e.output->data();
    }
} else {
    if (entries.size() > 1) {
        MemcpyInFusionBuffer(send_data); //copy in fusion buffer at offset 0
    } else {
        send_data = (void *)e.tensor->data();
    }
}

MPI_Gatherv(send_data,
            send_data == nullptr ? 0 : (int) total_num_elements,
            send_dtype,
            recv_data,
            recvcounts,
            displcmnts,
            recv_dtype,
            root,
            comm);

if (is_root) {
    if (entries.size() > 1) {
        MemcpyOutFusionBuffer();
    }
}
```

### broadcast fusion buffer
```c++
void *data_ptr = nullptr;
if (is_root) {
    if (entries.size() > 1) {
        MemcpyInFusionBuffer(data_ptr);
    } else {
        data_ptr = (void *) e.tensor->data();
    }
} else {
    if (entries.size() > 1) {
        data_ptr = fusion_buffer;
    } else {
        data_ptr = (void *) e.output->data();
    }
}

MPI_Bcast(data_ptr,
          send_counts,
          dtype,
          root,
          comm);

if (is_not_root) {
    if (entries.size() > 1) {
        MemcpyOutFusionBuffer();
    }
}
```

### choose which compression alg at python
```python
if grad.size > terngrad_threshold \
   and can be gathered into grad_cpu: # use terngrad
    if is not root_rank:
        terngrad(from grad to terngrad_gpu)
        copy to cpu buffer
    gather(from terngrad_cpu to grad_cpu)
    if is root_rank:
        for compressed_grad in grad_cpu not from root:
            copy to gpu buffer
            terngradr(from terngrad_gpu to grad) and reduce
        average()
        terngrad(from grad to terngrad_gpu)
        copy to cpu buffer
    broadcast(terngrad_cpu)
    if is not root:
        copy to gpu buffer
        terngradr(terngrad_gpu)
elif grad.size > dgc_threshold and \
     can be gathered into dgc_cpu: # use dgc
    if is not root_rank:
        dgc(from grad to dgc_gpu)
        copy to cpu buffer
    gather(2*ceil(N*s_percent)+1)
    if is root_rank:
        for compressed_grad in dgc_cpu and not from root:
            dgcr and reduce
        if threshold is satisfied: # change threshold
            dgc(from grad to dgc_gpu)
            copy to cpu buffer # broadcast more gradient elements
        else:
            maybe use terngrad or other compression alg
    broadcast(dgc_cpu)
    if is not root:
        if threshold is satisfied: # change threshold
            copy to gpu buffer
            average()
            dgcr()
        else:
            maybe use terngrad or other compression alg
else: # don't do compression
    allreduce()
```

### load balance and fixed compression output

```python
if do_compression:
    if grad.size > threshold:
        if can_be_gathered:
            rootid = grad.index%size()
            compressed_size = bits_of_compressed_grad(grad.size)/8 # in uint8
            if rankid != rootid:
                compress(grad, comp_gpu)
                comp_gpu[:compressed_size].copyto(comp_cpu[:compressed_size]) # copy partial
            gather(comp_cpu, rootid, compressed_size)

            if sparsity:
                if can_do_compress:
                    compressed_size_new = bits_of_compressed_grad(grad.size*size())
                    adjust the hyper parameters
                else:
                    warning("may use quantization")
            else:
                compressed_size_new = compressed_size

            if rankid == rootid:
                for idx in range(size()):
                    if idx != rootid:
                        start = idx*compressed_size; offset = start+compressed_size
                        comp_cpu[start:offset].copyto(comp_gpu[start:offset])
                        decompress(comp_gpu, grad)
                
                if sparsity:
                    if can_do_compress:
                        adjust the hyper parameters
                    else:
                        warning("may use quantization")
                    
                compress(grad, comp_gpu)
                comp_gpu[:compressed_size_new].copyto(comp_cpu[:compressed_size_new])
            
            mpibroadcast(comp_cpu, rootid, compressed_size_new)

            if rankid != rootid:
                if sparsity and cannot_do_compress:
                    warning("may use quantization")
                comp_cpu[:compressed_size_new].copyto(comp_gpu[:compressed_size_new])
                decompress(comp_gpu, grad)
            grad.__idiv__(size()) #average
        else:
            warning("more space for gathering)
    else:
        allreduce()

else:
    allreduce()
```

### generate a alltoall operator
```python
# construct send_counts and send_offsets from all other entries according to root_id
for root_rank in root_ranks:
    if root_rank == rank_id:
        construct recv_counts and recv_offsets
        recv_data = recv_fusion_buffer
    send_counts[root_rank] = gather_sizes[global_state.rank]
send_counts[global_state.rank] = 0 # don't send data to self
for i in range(1, send_counts.size()):
    send_offsets[i] += send_counts[i]
    memcpy into send_fusion_buffer
    send_data = send_fusion_buffer
```

### partition the huge message
```python
# if message size large than a threshold, the message will be split into rank_size partitions
if the comp alg is quantization:
    if grad.size >= 64MB: # compressed into 4MB
        initialize() # initialize the partitioned space on gpu and cpu for compressed data
        interval = grad.size // size()
        for rootid in range(size()): # size() return the number of participant nodes
            start = rootid * interval
            end = (rootid+1) * interval - 1
            compressed_size = byte(start, end)
            if rootid == size()-1:
                if grad.size > end:
                    end = grad.size
            if rank() != rootid: # rank() is the local rank of this node
                compress(data=grad[start:end], out=comp_gpu)
                comp_gpu.copyto(comp_cpu)
            gather_(comp_cpu, rootid=rootid)

            if rank() == rootid:
                for rootidx in range(size()):
                    if rootidx != rootid:
                        begin = rootidx * compressed_size
                        stop = begin + compressed_size
                        comp_cpu[begin:stop].copyto(comp_gpu[:compressed_size])
                        decompress(data=comp_gpu[:compressed_size], out=grad[start:end])
                compress(data=grad[start:end], out=comp_gpu)
                comp_gpu.copyto(comp_cpu)
            mpi_broadcast_(comp_cpu, rootid=rootid)

            if rank() != rootid:
                comp_cpu[:compressed_size].copyto(comp_gpu[:compressed_size])
                decompress(data=comp_gpu[:compressed_size], out=grad[start:end])
            
            average()
    else:

```