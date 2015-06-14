# -*- coding: utf-8 -*

import h5py
import os
import numpy as np
from numpy import nan
import collections
import datetime


def hdf_open(filename, *hdf_file, **kw_hdf_file):
    hdf_file = h5py.File(filename, *hdf_file, **kw_hdf_file)
    return Group(hdf_file['/'])


class Interface(object):
    """Dynamic and lightweight interface on h5py.

    """

    def __init__(self, hdf):
        if isinstance(hdf, (Group, Dataset)):
            self.__dict__['_hdf'] = hdf._hdf
        else:
            self.__dict__['_hdf'] = hdf

    def __dir__(self):
        return list(self._hdf.attrs.keys()) + list(self.__dict__.keys())

    def __setitem__(self, key, item):
        self._hdf[key] = item

    def __delitem__(self, key):
        del self._hdf[key]

    @property
    def attrs(self):
        return self._hdf.attrs


class Group(Interface):

    def __getitem__(self, key):
        item = self._hdf[key]

        if isinstance(item, h5py.Dataset):
            return Dataset(item)
        elif isinstance(item, h5py.Group):
            return Group(item)
        else:
            return self._hdf[key]

    def __repr__(self):
        return str(self.keys())

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def keys(self):
        return list(self._hdf.keys())

    def move(self, source, destination):
        self._hdf.move(source, destination)

    def copy(self, source, destination, *copy, **kw_copy):
        self._hdf.copy(source, destination, *copy, **kw_copy)

    def close(self):
        self._hdf.file.close()

    def create_dataset(self, key, override=False, date=True,
                       dtype=np.float64, fillvalue=np.nan, **kwargs):

        if override is True:
            try:
                del self[key]
            except KeyError:
                pass

        dataset = self._hdf.create_dataset(key, dtype=dtype, fillvalue=fillvalue, **kwargs)

        if date is True:
            # Standart date format '2014/10/31 14:25:57'
            dataset.attrs['date'] = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

        return Dataset(dataset)


class Dataset(Interface):

    def __init__(self, dataset):
        super().__init__(dataset)

        self.__dict__['trim'] = True

    def __getitem__(self, key):

        # Handle floating point slice numbers
        if isinstance(key, slice):
            if key.start is None:
                start = None
            else:
                start = int(key.start)

            if key.stop is None:
                stop = None
            else:
                stop = int(key.stop)

            if key.step is None:
                step = None
            else:
                step = int(key.step)

            # Pack new slice with integer values
            key = slice(start, stop, step)

        return self._hdf[key]

    def __repr__(self):
        return repr(self._hdf)

    def __len__(self):
        """Number of levels.

        """
        return self._hdf.size

    @property
    def shape(self):
        """Datatpye of the signal.

        """
        return self._hdf.shape

    @property
    def dims(self):
        """Access to dimension scales.

        """
        return self._hdf.dims

#    def add_data(self, loop_pos, data):
#        """Append data to dset
#
#        """
#
#        # Check measurment dimension -> 1d
#        if len(self._hdf.shape) == 1:
#            self._add_data_1d(loop_pos, data)
#
#        # Check measurment dimension -> 2d
#        elif len(self._hdf.shape) == 2:
#            self._add_data_2d(loop_pos, data)
#
#        # Check measurment dimension -> 3d
#        elif len(self._hdf.shape) == 3:
#            self._add_data_3d(loop_pos, data)
#
#        else:
#            err_str = 'add_data only works in 1d, 2d and 3d'
#            raise NotImplementedError(err_str)
#
#    def _add_data_1d(self, loop_pos, data):
#
#        # Single datapoint
#        if len(data) == 1:
#            self._hdf[loop_pos] = data[0]
#        elif type(data) == tuple:
#            self._hdf[loop_pos] = data
#
#        # Multiple datapoints
#        else:
#            start_pos = list(loop_pos)
#            start_pos[-1] = loop_pos[-1] - (len(data) - 1)
#            self._hdf[start_pos[-1]:loop_pos[-1] + 1] = data
#
#    def _add_data_2d(self, loop_pos, data):
#
#        # Single datapoint
#        if len(data) == 1:
#            self._hdf[loop_pos] = data[0]
#        elif type(data) == tuple:
#            self._hdf[loop_pos] = data
#
#        # Multiple datapoints
#        else:
#            # in one line
#            if not (len(data) - 1) > loop_pos[-1]:
#                start_pos = list(loop_pos)
#                start_pos[-1] = loop_pos[-1] - (len(data) - 1)
#
#                self._hdf[loop_pos[-2], start_pos[-1]:loop_pos[-1] + 1] = data
#
#            # in multiple lines
#            else:
#                shape = self._hdf.shape
#
#                y_pos = loop_pos[-2]
#                x_pos = loop_pos[-1] - (len(data) - 1)
#                while x_pos < 0:
#                    y_pos -= 1
#                    x_pos = x_pos + shape[-1]
#                start_pos = list(loop_pos)
#                start_pos[-2] = y_pos
#                start_pos[-1] = x_pos
#                d_ind = [0, (shape[-1] - 1) - x_pos]
#                while y_pos < loop_pos[-2]:
#                    d_ind[0] += x_pos + (shape[-1])
#                    d_ind[1] += x_pos + (shape[-1])
#                    self._hdf[y_pos, x_pos:shape[-1]+1] = data[d_ind[0]:d_ind[1]+1]
#                    x_pos = 0
#                    y_pos += 1
#
#                self._hdf[loop_pos[-3], y_pos, 0:loop_pos[-1]+1] = data[d_ind[0]:]
#
#    def _add_data_3d(self, loop_pos, data):
#
#        # Single datapoint
#        if len(data) == 1:
#            self._hdf[loop_pos] = data[0]
#        elif type(data) == tuple:
#            self._hdf[loop_pos] = data
#
#        # Multiple datapoints
#        else:
#            # in one line
#            if not (len(data) - 1) > loop_pos[-1]:
#                start_pos = list(loop_pos)
#                start_pos[-1] = loop_pos[-1] - (len(data) - 1)
#
#                self._hdf[loop_pos[-3], loop_pos[-2], start_pos[-1]:loop_pos[-1] + 1] = data
#
#            # in multiple lines
#            else:
#                shape = self._hdf.shape
#
#                y_pos = loop_pos[-2]
#                x_pos = loop_pos[-1] - (len(data) - 1)
#                while x_pos < 0:
#                    y_pos -= 1
#                    x_pos = x_pos + shape[-1]
#                start_pos = list(loop_pos)
#                start_pos[-2] = y_pos
#                start_pos[-1] = x_pos
#                d_ind = [0, (shape[-1] - 1) - x_pos]
#                while y_pos < loop_pos[-2]:
#                    d_ind[0] += x_pos + (shape[-1])
#                    d_ind[1] += x_pos + (shape[-1])
#                    self._hdf[loop_pos[-3], y_pos, x_pos:shape[-1]+1] = data[d_ind[0]:d_ind[1]+1]
#                    x_pos = 0
#                    y_pos += 1
#
#                self._hdf[loop_pos[-3], y_pos, 0:loop_pos[-1]+1] = data[d_ind[0]:]
