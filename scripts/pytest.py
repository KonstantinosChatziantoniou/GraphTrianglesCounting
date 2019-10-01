import os 
import csv
import subprocess


pwd = os.getcwd()
print(pwd)
os.chdir('../')

pwd = os.getcwd()
print(pwd)

datasets = ['auto.mtx', 'delaunay_n22.mtx', 'great-britain_osm.mtx']
paths = ['cuda', 'serial_cilk']
stats_dict = {}

for i in datasets:
    stats_dict[i] = {}


print(stats_dict)


reps = 10
for d in datasets:
    for p in paths:
        print(d + ' ' + p)
        os.chdir(pwd + '/scripts')
        f = open(d + p + '.csv', '+a')
        for r in range(reps):
            os.chdir(pwd + '/' + p)
            output = subprocess.check_output(['./mainProgram',d])
            output = output.decode('utf-8')
            print(output)
            output = output.split('\n')
            stats = []
            for o in output:
                #print(o)
                if len(o) > 0:
                    if o[0] != '-':
                        stats.append(o)
            i = 0
            for s in stats:
                s = s.split(',')
                time = int(s[1]) + (int(s[2]))/1000000
                i += 1
                if i == len(stats):
                    f.write(str(time) + '\n')
                else:
                    f.write(str(time) + ',')
        f.close()
