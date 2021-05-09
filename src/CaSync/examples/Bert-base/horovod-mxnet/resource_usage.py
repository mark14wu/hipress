#!/usr/bin/python3

import os
import re
import sys
import time
import socket
import argparse
import threading

batchsize = {'RTE' : 32, 'SST' : 16, 'MNLI' : 32}

#parse the parameters
def parseArgs():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        	description='measure the resource usage')
	parser.add_argument('--numprocess', type = int, default = 4,
							     help = 'the parameters for mpi -np')
	parser.add_argument('--servers', type = str, default = "egpu3:1,egpu4:1,egpu5:1,egpu6:1",
							     help = 'assign servers onto machine')
	parser.add_argument('--numbatches', type = int, default = 100,
									 help = 'number of batches per node for training')
	parser.add_argument('--interval', type = int, default = 20,
									 help = 'display interval')
	parser.add_argument('--lr', type=str, default=2e-5, help='Initial learning rate, default is 2e-5')
	parser.add_argument('--epsilon', type=str, default=1e-06, help='Small value to avoid division by 0, default is 1e-06')
	parser.add_argument('--max_len', type=int, default=128, help='Maximum length of the sentence pairs, default is 128')
	parser.add_argument('--sleeptime', type=int, default=10, help='after sleeping sleeptime seconds, the resource usage tools is launched')
	parser.add_argument('--strongscaling', action = 'store_true', help = 'if true, update batch size strongscaling, else weak')
	parser.add_argument('--extra', type = str, help = 'add some info by user')
	parser.add_argument('--usetools', action = 'store_true', help = 'if true, use network tools')
	parser.add_argument('--profiler', action = 'store_true', help = 'mxnet profile')
	parser.add_argument('--interface', type = str, default = 'tenGE', choices = ['tenGE', 'ib'],
								 	 help = 'the environment to execute')
	parser.add_argument('--model', type = str, default = 'RTE', choices = ['SST', 'MNLI'],
								 	 help = 'the training tasks')
	parser.add_argument('--horovodrun', action = 'store_true', help = 'use the horovodrun rather than mpirun')
	parser.add_argument('--bulk', type=int, default=15, help='bulk size at engine')
	parser.add_argument('--comp-threshold', type = int, default = sys.maxsize,
							     help = 'compression threshold, number of float data')
	parser.add_argument('--comp-alg', type = str, default = '',
							     help = 'compression algorithm name')

	return parser.parse_args()

class networkUsage:
	"""docstring for networkUsage"""
	def __init__(self, app, resultpath):
		self.thread_start = list()
		self.thread_finish = list()
		self.app = app
		self.resultpath = resultpath
	def launch_begin(self, hostid, parsedArgs):
		hostname = 'egpu' + str(hostid)
		if parsedArgs.interface == 'ib':
			netenv = 'ib0'
		elif parsedArgs.interface == 'tenGE':
			netenv = 'ens14f1'
		else:
			netenv = 'ens11f0'
		cmd = "ssh %s 'nohup ifstat -i %s -n 1> networkUsage_%s_%s_%s.log 2>&1 &'"%(hostname, netenv, hostname, self.app, str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
		os.system(cmd)
		print(time.ctime(time.time()), cmd)
	def start(self, parsedArgs):
		host_name_list = [elem.split(':')[0] for elem in parsedArgs.servers.split(',')]
		hostset = list()
		for hostname in host_name_list:
			x = re.search('.*?(\d+).*?', hostname)
			hostset.append(x.group(1))
		for hostid in hostset:
			t = threading.Thread(target = self.launch_begin, args = (hostid, parsedArgs, ))
			t.setDaemon(True)
			self.thread_start.append(t)
			t.start()
		# for t in self.thread_start:
		# 	t.join()
	def launch_finish(self, hostid):
		hostname = 'egpu' + str(hostid)
		# cmd = r'''ssh %s " kill \$(ps -aux | grep ifstat | awk '{print \$2}')"'''%(hostname)
		cmd = r'''ssh %s "killall ifstat"'''%(hostname)
		print(cmd)
		os.system(cmd)
		cmd = "scp %s:/home/gpu/networkUsage_* %s"%(hostname, self.resultpath)
		print(cmd)
		os.system(cmd)
		cmd = "ssh %s 'rm -f /home/gpu/networkUsage*'"%(hostname)
		os.system(cmd)
		# print(time.ctime(time.time()), 'network launch_finish:', hostname)
	def finish(self, parsedArgs):
		host_name_list = [elem.split(':')[0] for elem in parsedArgs.servers.split(',')]
		hostset = list()
		for hostname in host_name_list:
			x = re.search('.*?(\d+).*?', hostname)
			hostset.append(x.group(1))
		for hostid in hostset:
			t = threading.Thread(target = self.launch_finish, args = (hostid, ))
			t.setDaemon(True)
			self.thread_finish.append(t)
			t.start()
		for t in self.thread_finish:
			t.join()

class GPUcorrelation(object):
	"""gpu usage and gpu mem usage"""
	def __init__(self, app, resultpath):
		self.thread_start = list()
		self.thread_finish = list()
		self.app = app
		self.resultpath = resultpath
	def launch_begin(self, hostid, parsedArgs):
		hostname = 'egpu' + str(hostid)
		#grep '0  GeForce GTX' -A 1
		# nvidia-smi -lms 100
		cmd = '''ssh %s "nohup nvidia-smi -i 0 -q -d UTILIZATION -lms 100 | grep Gpu > gpuUsage_%s_%s_%s.log 2>&1 &"'''%(hostname, hostname, self.app, str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
		os.system(cmd)
		print(time.ctime(time.time()), cmd)
	def start(self, parsedArgs):
		host_name_list = [elem.split(':')[0] for elem in parsedArgs.servers.split(',')]
		hostset = list()
		for hostname in host_name_list:
			x = re.search('.*?(\d+).*?', hostname)
			hostset.append(x.group(1))
		for hostid in hostset:
			t = threading.Thread(target = self.launch_begin, args = (hostid, parsedArgs, ))
			t.setDaemon(True)
			self.thread_start.append(t)
			t.start()
	def launch_finish(self, hostid):
		hostname = 'egpu' + str(hostid)
		# cmd = r'''ssh %s " kill \$(ps -aux | grep nvidia-smi | awk '{print \$2}')"'''%(hostname)
		cmd = r'''ssh %s "killall nvidia-smi"'''%(hostname)
		os.system(cmd)
		cmd = "scp gpu@%s:/home/gpu/gpuUsage_* %s"%(hostname, self.resultpath)
		os.system(cmd)
		cmd = "ssh %s 'rm -f /home/gpu/gpuUsage_*'"%(hostname)
		os.system(cmd)
		# print(time.ctime(time.time()), 'gpu launch_finish:', hostname)
	def finish(self, parsedArgs):
		host_name_list = [elem.split(':')[0] for elem in parsedArgs.servers.split(',')]
		hostset = list()
		for hostname in host_name_list:
			x = re.search('.*?(\d+).*?', hostname)
			hostset.append(x.group(1))
		# for hostid in parsedArgs.servers+parsedArgs.workers:
		for hostid in hostset:
			t = threading.Thread(target = self.launch_finish, args = (hostid, ))
			t.setDaemon(True)
			self.thread_finish.append(t)
			t.start()
		for t in self.thread_finish:
			t.join()

class CPUcorrelation():
	'''cpu usage and cpu mem usage'''
	def __init__(self, app, resultpath):
		self.app = app
		self.thread_start = list()
		self.thread_finish = list()
		self.resultpath = resultpath
	def launch_begin(self, hostid, parsedArgs):
		hostname = 'egpu' + str(hostid)
		username = 'gpu'
		cmd = '''ssh %s "nohup top -b -d 1 -u %s | grep python3 > /home/gpu/cpuUsage_%s_%s_%s.log 2>&1 &"'''%(hostname, username, hostname, self.app, str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
		os.system(cmd)
		print(time.ctime(time.time()), cmd)
	def start(self, parsedArgs):
		host_name_list = [elem.split(':')[0] for elem in parsedArgs.servers.split(',')]
		hostset = list()
		for hostname in host_name_list:
			x = re.search('.*?(\d+).*?', hostname)
			hostset.append(x.group(1))
		for hostid in hostset:
			t = threading.Thread(target = self.launch_begin, args = (hostid, parsedArgs, ))
			t.setDaemon(True)
			self.thread_start.append(t)
			t.start()
	def launch_finish(self, hostid):
		hostname = 'egpu' + str(hostid)
		# cmd = r'''ssh %s " kill \$(ps -aux | grep top | awk '{print \$2}')"'''%(hostname)
		cmd = r'''ssh %s "killall top"'''%(hostname)
		os.system(cmd)
		cmd = "scp %s:/home/gpu/cpuUsage_* %s"%(hostname, self.resultpath)
		os.system(cmd)
		cmd = "ssh %s 'rm -f /home/gpu/cpuUsage_*'"%(hostname)
		os.system(cmd)
		# print(time.ctime(time.time()), 'cpu launch_finish:', hostname)
	def finish(self, parsedArgs):
		host_name_list = [elem.split(':')[0] for elem in parsedArgs.servers.split(',')]
		hostset = list()
		for hostname in host_name_list:
			x = re.search('.*?(\d+).*?', hostname)
			hostset.append(x.group(1))
		for hostid in hostset:
			t = threading.Thread(target = self.launch_finish, args = (hostid, ))
			t.setDaemon(True)
			self.thread_finish.append(t)
			t.start()
		for t in self.thread_finish:
			t.join()


class trainingApp:
	"""traing applications"""
	def __init__(self, app, parsedArgs, resultpath):
		self.app = app
		self.resultpath = resultpath
	def training(self, parsedArgs):
		cmd = '''./%(horo)s.sh %(numproc)d %(serverset)s %(bsize)d %(model)s %(batches)d %(inter)d %(lrate)s %(ep)s %(len)d %(blk)d %(alg)s %(thr)d 2>&1 | tee %(path)s/bert_ringtrain_%(time)s.log'''%(
			{
				'numproc': parsedArgs.numprocess,
				'serverset': parsedArgs.servers,
				'bsize': batchsize[self.app],
				'model': self.app,
				'batches': parsedArgs.numbatches,
				'inter': parsedArgs.interval,
				'lrate': parsedArgs.lr,
				'ep': parsedArgs.epsilon,
				'len': parsedArgs.max_len,
				'blk' : parsedArgs.bulk,
				'path': self.resultpath,
				'time': str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())),
				'horo': 'horovodrun' if parsedArgs.horovodrun else 'horovod',
				'alg': parsedArgs.comp_alg,
				'thr': parsedArgs.comp_threshold
			})
		print(cmd)
		os.system("echo %s > %s/command.log"%(cmd, self.resultpath))
		os.system(cmd)

def initialize():
	# initialize the batch size, strongscaling or weak
	if parsedArgs.strongscaling:
		for app in batchsize:
			batchsize[app] //= len(parsedArgs.servers.split(','))

def endOfonce(resultpath, parsedArgs):
	host_name_list = [elem.split(':')[0] for elem in parsedArgs.servers.split(',')]
	hostset = list()
	for hostname in host_name_list:
		x = re.search('.*?(\d+).*?', hostname)
		hostset.append(x.group(1))
	for hostid in hostset:
		hostname = 'egpu' + str(hostid)
		cmd = r'''ssh %s "kill \$(ps -aux | grep finetune_classifier | awk '{print \$2}')"'''%(hostname)
		print(cmd)
		os.system(cmd)
	if parsedArgs.profiler:
		for hostid in hostset:
			hostname = 'egpu' + str(hostid)
			cmd = "scp gpu@%s:/home/gpu/mxnet/horovod/examples/Bert-base/horovod-mxnet/*.json %s"%(hostname, resultpath)
			print(cmd)
			os.system(cmd)
			cmd = "ssh %s 'rm -f /home/gpu/mxnet/horovod/examples/Bert-base/horovod-mxnet/*.json'"%(hostname)
			print(cmd)
			os.system(cmd)
	print("end of main", time.ctime(time.time()))

if __name__ == '__main__':
	parsedArgs = parseArgs() #parse the parameters
	initialize()
	applications = [parsedArgs.model]
	for app in applications:
		dirname = str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
		if parsedArgs.extra:
			dirname = dirname + parsedArgs.extra
		basepath = '/home/gpu/mxnet/horovod/horovod-mxnet/trainResult/%s/%dmachines-%s/%s'%("Bert-"+app, len(parsedArgs.servers.split(',')), parsedArgs.numbatches, dirname)
		os.system("mkdir -p %s"%(basepath))

		# create measurement tools
		if parsedArgs.usetools:
			net = networkUsage(app, basepath)
			# gpu = GPUcorrelation(app, basepath)
			cpu = CPUcorrelation(app, basepath)

		train = trainingApp(app, parsedArgs, basepath)

		#start training
		# train.training(parsedArgs)
		thr = threading.Thread(target = train.training, args = (parsedArgs, ))
		thr.setDaemon(True)
		thr.start()

		#start the network, cpu, gpu tools per machine
		if parsedArgs.usetools:
			time.sleep(parsedArgs.sleeptime)
			net.start(parsedArgs)
			# gpu.start(parsedArgs)
			cpu.start(parsedArgs)
		thr.join()
		#finish the network, cpu, gpu tools per machine
		print('end of main')
		if parsedArgs.usetools:
			net.finish(parsedArgs)
			# gpu.finish(parsedArgs)
			cpu.finish(parsedArgs)
		endOfonce(basepath, parsedArgs)