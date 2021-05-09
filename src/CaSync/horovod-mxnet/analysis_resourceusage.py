'''
collect cpu gpu network message from source files
'''

import os
import sys
import argparse
import re

#parse the parameters
def parseArgs():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        	description='measure the resource usage')
	parser.add_argument('--clean', action = 'store_true',
							     help = 'clean the .data files')
	parser.add_argument('--root', type = str, default = './',
								 help = 'the begin directory')
	return parser.parse_args()

def cpu(root, filename):
	part = filename.split('_')
	if 'cpu' in part[0]:
		file = root + '/' + filename
		cmd = "cat %s | awk '{print $9\",\"$10}' > %s/cpu_%s.csv"%(file, root, part[1])
		print(cmd)
		os.system(cmd)

def gpu(root, filename):
	part = filename.split('_')
	if 'gpu' in part[0]:
		file = root + '/' + filename
		cmd = "cat %s | awk '{print $3}' > %s/gpu_%s.csv"%(file, root, part[1])
		print(cmd)
		os.system(cmd)
def network(root, filename):
	part = filename.split('_')
	if 'network' in part[0]:
		file = root + '/' + filename
		cmd = "cat %s | awk '/^\s*[0-9]+.[0-9]*/ {print $1/1024\",\"$2/1024}' > %s/network_%s.csv"%(file, root, part[1])
		print(cmd)
		os.system(cmd)

def speedAndtime(root, filename):
	print(filename)
	part = filename.split('_')
	if 'imagenet' in part[0] or 'bert' in part[0] or 'language' in part[0]:
		file = root + '/' + filename
		cmd = "cat %s"%(file)
		print(cmd)
		content = os.popen(cmd).readlines()
		speed = list()
		timecost = list()
		for line in content:
			x = re.search('.*?Average Speed.*?(\d+\.\d*)', line)
			if x:
				speed.append(float(x.group(1)))
			else:
				y = re.search('.*?Time cost=(\d+\.\d*)', line)
				if y:
					timecost.append(float(y.group(1)))
		output = dict()
		print(speed)

		#output to file
		f = open(root+'/imagenet.csv', 'w')
		# speed_out = [str(i) for i in speed]
		# time_out = [str(i) for i in timecost]
		# f.write(','.join(speed_out) + '\n' + ','.join(time_out))
		if len(speed) > 0:
			speed_out = max(speed)
		if len(timecost) > 0:
			time_out = min(timecost)
		if len(speed) > 0:
			f.write(str(speed_out)+'\n')
		f.close()


def cleanfile(root):
	cmd = "rm -f %s/*.data"%(root)
	os.system(cmd)
	cmd = "rm -f %s/*.csv"%(root)
	os.system(cmd)

if __name__ == '__main__':
	# workernum = int(sys.argv[1])
	args = parseArgs()
	count = 0
	for root, dirs, files in os.walk(args.root):
		# print(files)
		if root != './': # cd the child dir of current path
			if len(dirs) == 0 and len(files) != 0: #the dir contains .log files
				if args.clean:
					cleanfile(root)
				else:
					print(root)
					for file in files:
						speedAndtime(root, file)
						cpu(root, file)
						gpu(root, file)
						network(root, file)
					count += 1
	print(count)


