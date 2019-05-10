import sys
import numpy as np

if len(sys.argv) != 3:
    print('Usage: <beam> <filename>')
    exit(-1)
beam = int(sys.argv[1])
filename = sys.argv[2]

x = sys.stdin.readlines()
x = [x[i] for i in range(len(x)) if (i%beam== 0) ]
ids = []
lines = []
for raw_line in x:
    raw_line_array = raw_line.strip().split('\t')
    ids.append(int(raw_line_array[0][2:]))
    lines.append(raw_line_array[2])

# sort lines by ids
idx = np.array(ids).argsort()

ofile = open(filename, 'w')
for line in np.array(lines)[idx]:
    ofile.write(line + "\n")
ofile.close()
