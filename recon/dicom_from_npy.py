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
import argparse
import logging
import re

if __name__=='__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
         description='Convert numpy array to DICOM')
    
    parser.add_argument('input_file', type=str, help='path to numpy file')
    parser.add_argument('orig_dicom_dir', type=str, help='path to DICOM directory')
    parser.add_argument('--phase', type=int, default=0, help='index of phase to export')
    args = parser.parse_args()

    input_file = args.input_file
    orig_dicom_dir = args.orig_dicom_dir
    phase = args.phase
    logging.basicConfig(level=logging.INFO)

    # set directories
    im = np.load(input_file)
    new_dicom_dir = input_file[:-4] + '_dcm'
    
    dcm_ls = glob.glob(orig_dicom_dir + '/*.dcm')
    # Sort by slice number
    dcm_ls = sorted(dcm_ls,key=lambda s: int(re.findall(r'\d+',s)[-1]))
    series_mimic_dir = dcm_ls[-1]
    ds = pyd.dcmread(series_mimic_dir)
    # parse exam number, series number
    dcm_file = os.path.basename(series_mimic_dir)
    exam_number, series_mimic, _ = dcm_file.replace('Exam', '').replace(
        'Series', '_').replace('Image', '_').replace('.dcm', '').split('_')
    exam_number = int(exam_number)

    # Check size and zero-pad as needed (cannot rotate otherwise)
    # Assume first dimension is the number of phases
    if (im.shape[1] != im.shape[2] and im.shape[2] != im.shape[3] and im.shape[1] != im.shape[3]): 
        logging.warn('Data is not square - padding')
        imSize = np.max(im.shape)
        imOut = np.zeros((imSize,imSize,imSize),dtype=np.dtype('complex64'))
        pad_x = int(np.floor((imSize - im.shape[1])/2))
        pad_y = int(np.floor((imSize - im.shape[2])/2))
        pad_z = int(np.floor((imSize - im.shape[3])/2))
        imOut[pad_x:(pad_x+im.shape[1]),pad_y:(pad_y+im.shape[2]),pad_z:(pad_z+im.shape[3])] = im
    else:
        imOut = np.copy(im)

    im_shape = imOut.shape
    phases, ds.Columns, ds.Rows = im_shape[0], im_shape[-2], im_shape[-1]

    # Edit spatial resolution according to recon to 255 instead of 256
    spatial_resolution = ds.PixelSpacing[0] * (len(dcm_ls)/ds.Columns)

    # Update SliceLocation information
    series_mimic_slices = np.double(ds.Columns)  # assume recon is isotropic
#    ds.SpacingBetweenSlices = spatial_resolution
#    ds.SliceThickness = spatial_resolution
    ds.ReconstructionDiameter = spatial_resolution * imOut.shape[-1]
    # Update pixel spacing according to 255 vs 256 pixels
    ds.PixelSpacing = spatial_resolution

    SliceLocation_original = ds.SliceLocation
    ImagePositionPatient_original = ds.ImagePositionPatient 
    # Shift slice location by one pixel to make IPP match
    ImagePositionPatient_original[-1] = ImagePositionPatient_original[-1] - spatial_resolution
    
    logging.info(f'IPP original: {ImagePositionPatient_original}')

    # default orientation is Rotate270CounterClockwise, TransposeOff
    imOut = np.rot90(imOut,k=-3,axes=(2,3))
    # default flip 
    imOut = np.flip(imOut,1)
    imOut = np.flip(imOut,2)

    try:
        os.mkdir(new_dicom_dir)
    except OSError as error:
        pass

    imExport = imOut[phase,]
    ds.SeriesDescription = '3D UTE - mocolor phase' + str(phase)
    # adding 10, this should ensure no overlap with other series numbers for Philips numbering which uses series number (in order acquired) *100
    series_write = int(series_mimic) + (10*(phase+1))

    # modified time
    dt = datetime.datetime.now()

    imExport = np.abs(imExport) / np.amax(np.abs(imExport)) * 4095  # 65535
    imExport = imExport.astype(np.uint16)

    # Window and level for the image
    ds.WindowCenter = int(np.amax(imExport) / 2)
    ds.WindowWidth = int(np.amax(imExport))

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
            (im_shape[-1] / 2 - (z + 1)) * spatial_resolution
        ds.ImagePositionPatient = pyd.multival.MultiValue(float, [float(ImagePositionPatient_original[0]), float(
            ImagePositionPatient_original[1]), ImagePositionPatient_original[2] - (im_shape[-1] / 2 - (z + 1)) * spatial_resolution])
        b = imExport[z, :, :].astype('<u2')
        ds.PixelData = b.T.tobytes()
        #ds.is_little_endian = False
        ds.save_as(Filename)
