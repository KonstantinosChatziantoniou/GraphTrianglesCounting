

import pandas
import matplotlib.pyplot as plt

datasets = ['auto.mtx', 'delaunay_n22.mtx', 'great-britain_osm.mtx']
paths = ['cuda', 'serial_cilk']

stats = {}
stats['cuda'] = []
stats['serial'] = []
stats['cilk'] = []
stats['matlab'] = [3.3, 4.43 ,1.6]

stats['cuda960'] = [0.07 + 0.08, 0.1 + 0.39 ,0.1 + 0.47]

# read cuda
for d in datasets:
        p = paths[0]
        df = pandas.read_csv(d+p+'.csv', header=None)
        df = df.mean()
        read_time = df[0]
        csr_time = df[1]
        mem_time = df[2]
        exec_time = df[3]
        stats['cuda'].append(df[2] + df[3])


# read serial
for d in datasets:
        p = paths[1]
        df = pandas.read_csv(d+p+'.csv', header=None)
        df = df.mean()
        read_time = df[0]
        csr_time = df[1]
        mem_time = df[2]
        exec_time = df[3]
        stats['serial'].append(df[2])
        stats['cilk'].append(df[3])

print(stats)

df = pandas.DataFrame({'cuda': stats['cuda'], 'serial':stats['serial'], 'cilk': stats['cilk'],'cuda960': stats['cuda960'],'matlab':stats['matlab']}, index=datasets)
a = df.plot.bar( rot=0, title='Mean execution times')
a.set_xlabel('dataset')
a.set_ylabel('sec')
print(df)
plt.show()



############## FOR MIN ############
stats = {}
stats['cuda'] = []
stats['serial'] = []
stats['cilk'] = []
stats['matlab'] = [3.3, 4.43 ,1.6]
stats['cuda960'] = [0.07 + 0.08, 0.1 + 0.39 ,0.1 + 0.47]

# read cuda
for d in datasets:
        p = paths[0]
        df = pandas.read_csv(d+p+'.csv', header=None)
        df = df.min()
        read_time = df[0]
        csr_time = df[1]
        mem_time = df[2]
        exec_time = df[3]
        stats['cuda'].append(df[2] + df[3])


# read serial
for d in datasets:
        p = paths[1]
        df = pandas.read_csv(d+p+'.csv', header=None)
        df = df.min()
        read_time = df[0]
        csr_time = df[1]
        mem_time = df[2]
        exec_time = df[3]
        stats['serial'].append(df[2])
        stats['cilk'].append(df[3])


print(stats)

df = pandas.DataFrame({'cuda': stats['cuda'], 'serial':stats['serial'], 'cilk': stats['cilk'],'cuda960': stats['cuda960'],'matlab':stats['matlab']}, index=datasets)
a = df.plot.bar( rot=0, title='Min execution times')
a.set_xlabel('dataset')
a.set_ylabel('sec')

print(df)
plt.show()
