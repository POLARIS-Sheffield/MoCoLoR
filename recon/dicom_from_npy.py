#!/usr/bin/env python3
"""
Conversion of numpy array output of imoco/mocolor recon to DICOM series - adapted from imoco_py/dicom_creation.py
https://github.com/PulmonaryMRI/imoco_recon/blob/master/imoco_py/dicom_creation.py
Command line inputs needed: raw data (numpy array) directory; original DICOM directory
Modified by: Neil J Stewart njstewart-eju (2023/08)
Original author: Fei Tan (ftan1)
"""
import numpy as np
import pydicom as pyd
import datetime
import os
import glob
import sys
import argparse
import logging

if __name__=='__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
         description='Convert numpy array to DICOM')
    
    parser.add_argument('input_file', type=str, help='path to numpy file')
    parser.add_argument('orig_dicom_dir', type=str, help='path to DICOM directory')
    parser.add_argument('--flip_lr_flag', type=int, default=1, help='flag to flip the images')
    parser.add_argument('--flip_ud_flag', type=int, default=1, help='flag to flip the images')
    args = parser.parse_args()

    flip_lr_flag = args.flip_lr_flag
    flip_ud_flag = args.flip_ud_flag
    input_file = args.input_file
    orig_dicom_dir = args.orig_dicom_dir
    logging.basicConfig(level=logging.INFO)

    # set directories
    ims = np.load(input_file)
    
    if ims.ndim == 3:
        ims = np.transpose(ims, axes=[0, 2, 1])
    elif ims.ndim == 4:
        ims = np.transpose(ims, axes=[0, 1, 3, 2])
    
    new_dicom_dir = input_file[:-4] + '_dcm'

    # load original dicom
    # exam number, series number
    dcm_ls = glob.glob(orig_dicom_dir + '/*.dcm')
    series_mimic_dir = dcm_ls[0]
    ds = pyd.dcmread(series_mimic_dir)
    # parse exam number, series number
    dcm_file = os.path.basename(series_mimic_dir)
    exam_number, series_mimic, _ = dcm_file.replace('Exam', '').replace(
        'Series', '_').replace('Image', '_').replace('.dcm', '').split('_')
    exam_number = int(exam_number)

    ims_shape = np.shape(ims)
    if ims.ndim == 3:
        phases = 1
        ds.Columns, ds.Rows = ims_shape[-2], ims_shape[-1]
    else:
        phases, ds.Columns, ds.Rows = ims_shape[0], ims_shape[-2], ims_shape[-1]

    spatial_resolution = ds.SliceThickness

    # Update SliceLocation information
    series_mimic_slices = np.double(ds.Columns)  # assume recon is isotropic
    SliceLocation_center = ds.SliceLocation - \
        (series_mimic_slices - 1) / 2 * ds.SpacingBetweenSlices
    ImagePosition_zcenter = ds.ImagePositionPatient[2] + (
        series_mimic_slices - 1) / 2 * ds.SpacingBetweenSlices

    ds.SpacingBetweenSlices = spatial_resolution
    ds.PixelSpacing = [spatial_resolution, spatial_resolution]
    ds.SliceThickness = spatial_resolution
    ds.ReconstructionDiameter = spatial_resolution * ims_shape[-1]

    SliceLocation_original = ds.SliceLocation
    ImagePositionPatient_original = ds.ImagePositionPatient

    # Comment this line to flip images up-down
    if flip_ud_flag:
        if ims.ndim == 3:
            ims = np.flip(ims,0)
        else:
            ims = np.flip(ims,1)
    if flip_lr_flag:
        if ims.ndim == 3:
            ims = np.flip(ims,2)
        else:
            ims = np.flip(ims,3)

    try:
        os.mkdir(new_dicom_dir)
    except OSError as error:
        pass

    for ph in range(phases):
        if phases > 1:
            im = ims[ph,]
            ds.SeriesDescription = '3D UTE - mocolor phase' + str(ph+1)
        else:
            ds.SeriesDescription = '3D UTE - imoco' 
            im = ims
        # adding 10, this should ensure no overlap with other series numbers for Philips numbering which uses series number (in order acquired) *100
        series_write = int(series_mimic) + (10*(ph+1))

        # modified time
        dt = datetime.datetime.now()


        im = np.abs(im) / np.amax(np.abs(im)) * 4095  # 65535
        im = im.astype(np.uint16)

        # Window and level for the image
        ds.WindowCenter = int(np.amax(im) / 2)
        ds.WindowWidth = int(np.amax(im))

        # dicom series UID
        ds.SeriesInstanceUID = pyd.uid.generate_uid()

        # not currently accounting for oblique slices...
        for z in range(ds.Rows):
            ds.InstanceNumber = z + 1
            ds.SeriesNumber = series_write
            ds.SOPInstanceUID = pyd.uid.generate_uid()
            # SOPInstanceUID should == MediaStorageSOPInstanceUID
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            Filename = '{:s}/E{:d}S{:d}I{:d}.DCM'.format(
                new_dicom_dir, exam_number, series_write, z + 1)
            ds.SliceLocation = SliceLocation_original + \
                (ims_shape[-1] / 2 - (z + 1)) * spatial_resolution
            ds.ImagePositionPatient = pyd.multival.MultiValue(float, [float(ImagePositionPatient_original[0]), float(
                ImagePositionPatient_original[1]), ImagePositionPatient_original[2] - (ims_shape[-1] / 2 - (z + 1)) * spatial_resolution])
            b = im[z, :, :].astype('<u2')
            ds.PixelData = b.T.tobytes()
            #ds.is_little_endian = False
            ds.save_as(Filename)
