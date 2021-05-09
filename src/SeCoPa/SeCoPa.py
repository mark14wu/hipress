import os
import argparse

class cost_model(object):
    def __init__(self, weight, bias, overhead):
        self._weight = weight
        self._bias = bias
        self._overhead = overhead
        self._threshold = self._get_threshold()
    
    def get_cost(self,x):
        if x < self._threshold:
            return self._overhead
        else:
            return self._weight * x + self._bias

    def _get_threshold(self):
        if self._weight == 0:
            return 0
        return max([0, (self._overhead - self._bias) / self._weight])


class selective_compression(object):
    def __init__(self, nodes=16):
        self._encode = cost_model(1.4133e-8, 5.8938e-2, 0.05408)
        self._decode = cost_model(6.2041e-9, 1.3956e-1, 0.07287)
        self._aggr = cost_model(0, 0, 0)
        self._ring_send_one_step = cost_model(8.5005e-7, 2.0337e-1, 0.09766)
        self._nodes = nodes
        self._comp_ratio = 1. / 16
        if nodes == 4:
            self._gather = cost_model(2.7156e-6, 4.1930e-1, 0.15573)
            self._broadcast = cost_model(2.7157e-6, 2.2156e-1, 0.14820)
        elif nodes == 8:
            self._gather = cost_model(6.1167e-6, 8.0768e-1, 0.28778)
            self._broadcast = cost_model(6.1172e-6, 2.7285e-1, 0.29666)
        elif nodes == 16:
            self._gather = cost_model(1.2937e-5, 1.4406, 0.59503)
            self._broadcast = cost_model(1.2915e-5, 4.5538e-1, 0.58353)
        else:
            print("only support nodes in [4, 8, 16]")
            raise ValueError

    
    def get_orig_sync_cost(self, grad_size, comm_type = "PS"):
        if comm_type == "PS":
            # CPS ( S == N )
            # K partition number
            # colated PS
            # gather + (N-1) aggression + broadcast
            results = []
            for K in range(1, self._nodes + 1):
                gather = self._gather.get_cost(grad_size / K) 
                broadcast = self._broadcast.get_cost(grad_size / K)
                aggr = (self._nodes - 1) * self._aggr.get_cost(grad_size / K)
                results.append(gather + broadcast + aggr)
            min_value = min(results)
            K = results.index(min_value)
            return min_value, K+1

        else:
            # ring
            # 2(N-1) mpi send + (N-1) aggression 
            results = []
            for K in range(1, self._nodes + 1):
                send = 2 * (self._nodes - 1) * self._ring_send_one_step.get_cost(grad_size / K)
                aggr = (self._nodes - 1) * self._aggr.get_cost(grad_size / K)
                results.append(send + aggr)
            min_value = min(results)
            K = results.index(min_value)
            return min_value, K+1


    def get_cpr_sync_cost(self, grad_size, comm_type = "PS"):
        if comm_type == "PS":
            # CPS
            # K partition number
            # gather + broadcast + decode + encode + aggree
            results = []
            for K in range(1, self._nodes + 1):
                gather = self._gather.get_cost(grad_size * self._comp_ratio / K)
                broadcast = self._broadcast.get_cost(grad_size * self._comp_ratio / K)
                encode = min([K + 1, self._nodes]) * self._encode.get_cost(grad_size / K)
                decode = min([self._nodes - 1 + K, 2 * (self._nodes - 1)]) * self._decode.get_cost(grad_size * self._comp_ratio / K)
                aggr = (self._nodes - 1) * self._aggr.get_cost(grad_size / K)
                results.append(gather + broadcast + encode + decode + aggr)
            min_value = min(results)
            K = results.index(min_value)
            return min_value, K+1

        else:
            # K partition number 
            # 2(N-1) send + N encode + N decode + (N-1) aggression
            results = []
            for K in range(1, self._nodes + 1):
                send = 2 * (self._nodes - 1) * self._ring_send_one_step.get_cost(grad_size * self._comp_ratio / K)
                encode = self._nodes * self._encode.get_cost(grad_size / K)
                decode = self._nodes * self._decode.get_cost(grad_size * self._comp_ratio / K)
                aggr = (self._nodes - 1) * self._aggr.get_cost(grad_size / K)
                results.append(send + encode + decode + aggr)
            min_value = min(results)
            K = results.index(min_value)
            return min_value, K+1

    def get_best_result(self, grad_size, comm_type = "PS"):
        orig, orig_k = self.get_orig_sync_cost(grad_size, comm_type)
        # print("orig", orig, orig_k)
        comp, comp_k = self.get_cpr_sync_cost(grad_size, comm_type)
        # print("comp", comp, comp_k)
        if orig > comp:
            return 'y', comp_k
        else:
            return 'n', orig_k

# parse the parameters
def parseArgs():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        description='Begin the SeCoPa analysis.')
    parser.add_argument('--input', 
                        type = str, 
                        default = 'input.txt', 
                        help = 'input file for SeCoPa')
    parser.add_argument('--output', 
                        type = str, 
                        default = 'SeCoPaPlan.txt', 
                        help = 'output file for SeCoPa')
    parser.add_argument('--topology',
                        type = str,
                        default = 'PS',
                        choices = ['PS', 'RING'],
                        help = 'transmission topology')
    parser.add_argument('--nnodes',
                        type = int,
                        default = 16,
                        help = 'cluster size')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArgs()
    planner = selective_compression(nodes=int(args.nnodes))
    fout = open(args.output, 'w')
    with open(args.input, 'r') as f:
        for line in f:
            grad_name = str(line.split(',')[0])
            grad_size = int(line.split(',')[1])
            # print(grad_name, grad_size)
            compr, parts = planner.get_best_result(grad_size, args.topology)
            output = grad_name+','+str(grad_size)+','+str(compr)+','+str(parts)+'\n'
            fout.write(output)
    fout.close()