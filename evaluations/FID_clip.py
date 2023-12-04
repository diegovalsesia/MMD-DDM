from cleanfid import fid

score = fid.compute_fid('./cifar_test/', '.cifar_train/', mode ="legacy_pytorch", batch_size = 2000, model_name="clip_vit_b_32") 
print(f"The FID is: {score}")