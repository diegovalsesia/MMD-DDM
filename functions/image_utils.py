import cv2 # opencv
import numpy as np
import matplotlib.pyplot as plt
import torch

def generate_sample_sheet(sample_img_list, step, image_res):
    # generate and save the 32x32 sample sheet from sample_img_list
    (image_res, image_res, ch) = sample_img_list[0].shape
    sample_sheet_all = np.zeros((image_res*4,image_res*4,ch))
    single_img_height = single_img_width = image_res
    k=0
    for i in range(4):
        for j in range(4):
            start_row_pos = i*single_img_height
            end_row_pos = (i+1)*single_img_height
            start_col_pos = j*single_img_width
            end_col_pos = (j+1)*single_img_width            
            sample_sheet_all[start_row_pos:end_row_pos,start_col_pos:end_col_pos,:] = sample_img_list[k]
            k+=1
            
    # save this sheet
    plt.imsave(f"./sheet_{step}.png", sample_sheet_all)
    
    return
def generate_sample_sheet_6(sample_img_list, step, image_res):
    # generate and save the 32x32 sample sheet from sample_img_list
    (image_res, image_res, ch) = sample_img_list[0].shape
    sample_sheet_all = np.zeros((image_res*3,image_res*3,ch))
    single_img_height = single_img_width = image_res
    k=0
    for i in range(3):
        for j in range(3):
            start_row_pos = i*single_img_height
            end_row_pos = (i+1)*single_img_height
            start_col_pos = j*single_img_width
            end_col_pos = (j+1)*single_img_width            
            sample_sheet_all[start_row_pos:end_row_pos,start_col_pos:end_col_pos,:] = sample_img_list[k]
            k+=1
            
    # save this sheet
    plt.imsave(f"./sheet_{step}.png", sample_sheet_all)
    
    return

def generate_sample_sheet_8(sample_img_list, step, image_res):
    # generate and save the 32x32 sample sheet from sample_img_list
    (image_res, image_res, ch) = sample_img_list[0].shape
    sample_sheet_all = np.zeros((image_res*5,image_res*5,ch))
    single_img_height = single_img_width = image_res
    k=0
    for i in range(5):
        for j in range(5):
            start_row_pos = i*single_img_height
            end_row_pos = (i+1)*single_img_height
            start_col_pos = j*single_img_width
            end_col_pos = (j+1)*single_img_width            
            sample_sheet_all[start_row_pos:end_row_pos,start_col_pos:end_col_pos,:] = sample_img_list[k]
            k+=1
            
    # save this sheet
    plt.imsave(f"./sheet_{step}.png", sample_sheet_all)
    
    return

def generate_sample_sheet_4(sample_img_list, step, image_res):
    # generate and save the 32x32 sample sheet from sample_img_list
    (image_res, image_res, ch) = sample_img_list[0].shape
    sample_sheet_all = np.zeros((image_res*2,image_res*2,ch))
    single_img_height = single_img_width = image_res
    k=0
    for i in range(2):
        for j in range(2):
            start_row_pos = i*single_img_height
            end_row_pos = (i+1)*single_img_height
            start_col_pos = j*single_img_width
            end_col_pos = (j+1)*single_img_width            
            sample_sheet_all[start_row_pos:end_row_pos,start_col_pos:end_col_pos,:] = sample_img_list[k]
            k+=1
            
    # save this sheet
    plt.imsave(f"./sheet_{step}.png", sample_sheet_all)
    
    return


def torch_sheet(sample_img_list, step):
    # generate and save the 32x32 sample sheet from sample_img_list
    
    (ch, single_img_height, single_img_width) = sample_img_list[0].shape
    sample_sheet_all = torch.zeros((ch, single_img_height*4,single_img_width*2))
    k=0
    for i in range(4):
        for j in range(2):
            start_row_pos = i*single_img_height
            end_row_pos = (i+1)*single_img_height
            start_col_pos = j*single_img_width
            end_col_pos = (j+1)*single_img_width            
            sample_sheet_all[start_row_pos:end_row_pos,start_col_pos:end_col_pos,:] = sample_img_list[k]
            k+=1
    print(sample_sheet_all)
    print(sample_sheet_all.shape)        
    
    # save this sheet
    plt.imsave(f"./sheet_{step}.png", sample_sheet_all)
    
    return