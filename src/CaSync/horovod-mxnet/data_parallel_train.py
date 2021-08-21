#!/usr/bin/python3

import os
import re
import sys
import time
import socket
import argparse
import threading

#applications parameters
params = 	{
					'resnet50' : ['resnet50_v2', '3,224,224'],
					'resnet152' : ['resnet152_v2', '3,224,224'],
                                        'resnet18' : ['resnet18_v2', '3,224,224'],
					'vgg16' : ['vgg16', '3,224,224'],
					'vgg19' : ['vgg19', '3,224,224'],
					'inceptionv3' : ['inceptionv3', '3,299,299'],
					'inceptionv4' : ['inceptionv4', '3,299,299'],
					'lenet' : ['lenet', '3,224,224'],
					'alexnet' : ['alexnet', '3,224,224']
					}

# at local hosts
batchsize = {'resnet50' : 32, 'inceptionv3' : 32, 'inceptionv4' : 32, 'vgg16' : 16, 'vgg19' : 16, 'lenet' : 64, 'inception-v5' : 16, 'alexnet' : 128}
# at aws M60
# batchsize = {'resnet50' : 64, 'inception-v4' : 16, 'inception-v3' : 32, 'vgg16' : 32, 'vgg19' : 16, 'lenet' : 64, 'inception-v5' : 16, 'lenet1' : 64, 'resnet152' : 16}
# batchsize = {'resnet50' : 32, 'inception-v4' : 8, 'inception-v3' : 16, 'vgg19' : 8, 'resnet152' : 8}

#parse the parameters
def parseArgs():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        	description='measure the resource usage')
	parser.add_argument('--numprocess', type = int, default = 4,
							     help = 'the parameters for mpi -np')
	parser.add_argument('--servers', type = str, default = "egpu3:1,egpu4:1,egpu5:1,egpu6:1",
							     help = 'assign servers onto machine')
	parser.add_argument('--numexamples', type = int, default = 51200,
									 help = 'number of examples for training')
	parser.add_argument('--strongscaling', action = 'store_true', help = 'if true, update batch size strongscaling, else weak')
	parser.add_argument('--extra', type = str, help = 'add some info by user')
	parser.add_argument('--usetools', action = 'store_true', help = 'if true, use network tools')
	parser.add_argument('--profiler', action = 'store_true', help = 'mxnet profile')
	parser.add_argument('--interface', type = str, default = 'ens14f1',
								 	 help = 'the network interface')
	parser.add_argument('--model', type = str, default = 'resnet50', choices = ['lenet', 'resnet152', 'vgg19', 'inceptionv4', 'resnet50', 'alexnet'],
								 	 help = 'the training model')
	parser.add_argument('--bulk', type = int, default = 15,
							     help = 'bulk size of mxnet engine')
	parser.add_argument('--comp-threshold', type = int, default = 262144,
							     help = 'compression threshold, number of float data')
	parser.add_argument('--comp-alg', type = str, default = 'tbq',
							     help = 'compression algorithm name')
	parser.add_argument('--comprplan', type = str, default = 'tbq',
							     help = 'compressing plan from SeCoPa')
	parser.add_argument('--horovodrun', action = 'store_true', help = 'use the horovodrun rather than mpirun')
	parser.set_defaults(horovodrun=True)
	parser.add_argument('--lenet', action = 'store_true', help = 'use the lenet as training model')
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
		print(time.ctime(time.time()), 'network launch_finish:', hostname)
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
		print(time.ctime(time.time()), 'gpu launch_finish:', hostname)
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
		cmd = '''ssh %s "nohup top -b -d 1 -u %s | grep python3 | grep -v python3.6 > /home/gpu/cpuUsage_%s_%s_%s.log 2>&1 &"'''%(hostname, username, hostname, self.app, str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
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
		print(time.ctime(time.time()), 'cpu launch_finish:', hostname)
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
		if "inception" in self.app:
			train_rec = "/home/gpu/trainData/traindata-299.rec"
			train_idx = "/home/gpu/trainData/traindata-299.idx"
		else:
			train_rec = "/home/gpu/trainData/traindata.rec"
			train_idx = "/home/gpu/trainData/traindata.idx"
		cmd = '''./%(horo)s.sh %(numproc)d %(serverset)s %(bsize)d %(examples)d %(model)s %(shape)s %(rec)s %(idx)s %(blk)d %(alg)s %(thr)d %(net)s %(lenet)s 2>&1 | tee %(path)s/imagenet_ringtrain_%(time)s.log'''%(
			{
				'numproc': parsedArgs.numprocess,
				'serverset': parsedArgs.servers,
				'model': params[self.app][0],
				'bsize': batchsize[self.app],
				'examples': parsedArgs.numexamples,
				'path': self.resultpath,
				'time': str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())),
				'shape': params[self.app][1],
				'horo': 'horovodrun' if parsedArgs.horovodrun else 'horovod',
				'rec' : train_rec,
				'idx' : train_idx,
				'blk' : parsedArgs.bulk,
				'alg' : parsedArgs.comp_alg,
				'thr' : parsedArgs.comp_threshold,
				'net' : parsedArgs.interface,
				'lenet' : "--lenet" if parsedArgs.lenet else ''
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
		cmd = r'''ssh %s "kill \$(ps -aux | grep mxnet_imagenet | awk '{print \$2}')"'''%(hostname)
		print(cmd)
		os.system(cmd)
	if parsedArgs.profiler:
		for hostid in hostset:
			hostname = 'egpu' + str(hostid)
			cmd = "scp %s:/home/gpu/mxnet/horovod/horovod-mxnet/*.json %s"%(hostname, resultpath)
			os.system(cmd)
			cmd = "ssh %s 'rm -f /home/gpu/mxnet/horovod/horovod-mxnet/*.json'"%(hostname)
			os.system(cmd)
			hostname = 'egpu' + str(hostid)
			cmd = "scp %s:/home/gpu/mxnet/horovod/horovod-mxnet/*.log %s"%(hostname, resultpath)
			os.system(cmd)
			cmd = "ssh %s 'rm -f /home/gpu/mxnet/horovod/horovod-mxnet/*.log'"%(hostname)
			os.system(cmd)
			hostname = 'egpu' + str(hostid)
			cmd = "scp %s:/home/gpu/mxnet/horovod/horovod-mxnet/*.txt %s"%(hostname, resultpath)
			os.system(cmd)
			cmd = "ssh %s 'rm -f /home/gpu/mxnet/horovod/horovod-mxnet/*.txt'"%(hostname)
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
		basepath = './trainResult/%s/%dmachines-%s/%s'%(app, len(parsedArgs.servers.split(',')), parsedArgs.numexamples, dirname)
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

		time.sleep(10)
		#start the network, cpu, gpu tools per machine
		if parsedArgs.usetools:
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
