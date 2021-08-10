import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

"""

"""
# Paths
dataset_dir = '../HarPData/'
image_scans_path = dataset_dir + 'Scans/' + '100/'
image_labels_path = dataset_dir + 'Labels/Released_data_MNC_v1.3/Labels_100_MINC/'


class SegPipeA:

    def __init__(self, d_dir, x_dir, y_dir, view, hem):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.view = view
        self.hem = hem

        self.x_paths = []
        self.y_paths = []
        self.training_path = d_dir + 'Training/'

    def load_paths(self):
        print('Gathering training mnc file paths...')
        for file in os.listdir(self.x_dir):
            if not file.startswith('.') and file.endswith('.mnc'):
                file_path = self.x_dir + file
                # print('Loading video: ', file_path)
                if not os.path.exists(file_path):
                    raise FileNotFoundError
                self.x_paths.append(file_path)

        for file in os.listdir(self.y_dir):
            if not file.startswith('.'):
                seg_id = file.split('_')[-2]
                if not (seg_id == 'CSF'):
                    hem_id = file.split('_')
                    hem_id = hem_id[-1].split('.')[0]
                    if hem_id.lower() == self.hem.lower():
                        file_path = self.y_dir + file
                        # print('Loading video: ', file_path)
                        if not os.path.exists(file_path):
                            raise FileNotFoundError
                        self.y_paths.append(file_path)
        self.x_paths = sorted(self.x_paths)
        self.y_paths = sorted(self.y_paths)

        print('File paths gathered.')

    def get_view(self, mnc_data):
        data = []
        if self.view.lower() == 'axial':
            for i in range(len(mnc_data)):
                slices = mnc_data[i, :, :]
                data.append(slices)
        elif self.view.lower() == 'coronal':
            for i in range(len(mnc_data)):
                slices = mnc_data[:, i, :]
                data.append(slices)
        elif self.view.lower() == 'sagittal':
            for i in range(len(mnc_data)):
                slices = mnc_data[:, :, i]
                data.append(slices)

        return data

    @staticmethod
    def get_seg_idx(slices):
        seg_idx = []

        def is_slice_empty(_img):
            _img = (_img > 0).astype(bool)
            if _img.any():
                return False
            elif not _img.all():
                return True

        for i, img in enumerate(slices):
            if not is_slice_empty(img):
                seg_idx.append(i)
            else:
                pass

        return seg_idx

    @staticmethod
    def get_id(path):
        path_id = path.split('_')
        path_id = path_id[-2]

        return path_id

    def gen_training_images(self):
        print('Generating training image data')
        for i, path in enumerate(self.y_paths):
            path_id = self.get_id(path)
            obj = nib.load(path)
            data = obj.get_fdata()
            yslices = self.get_view(data)
            obj2 = nib.load(self.x_paths[i])
            data2 = obj2.get_fdata()
            xslices = self.get_view(data2)

            idxs = self.get_seg_idx(yslices)
            x = slice(idxs[0], (idxs[-1]+1))
            print(idxs, x)
            yslices = yslices[x]
            xslices = xslices[x]
            print(yslices.__len__(), xslices.__len__())

            if os.path.exists(dataset_dir + 'Training'):
                pass
            else:
                os.mkdir(self.training_path)
                os.mkdir(self.training_path + 'x/')
                os.mkdir(self.training_path + 'y/')

            xpath = (self.training_path + 'x/')
            ypath = (self.training_path + 'y/')
            for j, img in enumerate(xslices):
                plt.imsave(xpath +
                           (path_id + '_x_' + self.hem + '_' + self.view + '_' + str(idxs[j])),
                           img,
                           format='png',
                           cmap='gray',
                           origin='lower')
            for j, img in enumerate(yslices):
                plt.imsave(ypath +
                           (path_id + '_y_' + self.hem + '_' + self.view + '_' + str(idxs[j])),
                           img,
                           format='png',
                           cmap='gray',
                           origin='lower')


"""
if __name__ == '__main__':
    pipe = SegPipeA(dataset_dir, image_scans_path, image_labels_path, 'sagittal', 'l')
    pipe.load_paths()
    pipe.gen_training_images()
"""


