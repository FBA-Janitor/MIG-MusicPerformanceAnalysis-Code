import os

def write_txt(seg, filename, output_dir):
    new_file = open(os.path.join(output_dir, filename), 'w')
    for i in range(len(seg)):
        output = '{:.6f},{:.6f},{:.6f}\n'.format(seg[i][0], seg[i][1], seg[i][2])
        new_file.write(output)
    new_file.close()