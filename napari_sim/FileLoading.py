import tifffile
import h5py


def load_file(path):
    if path.split('.')[-1] == 'tiff' or path.split('.')[-1] == 'tif':
        return tifffile.imread(path)
    else:
        f = h5py.File(path, 'r')
        return f[list(f.keys())[0]]


