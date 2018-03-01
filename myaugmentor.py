import cv2
import numpy as np
import random


cv2.namedWindow('display',0)

img=cv2.imread('./imgs/iamword.jpg',0)
TH,TW=img.shape

param_gamma_low=.3
param_gamma_high=2
param_mean_gaussian_noise=0
param_sigma_gaussian_noise=100**0.5
param_var_otsu=50
param_displacement_affine=.1
param_slant_correction_affine=2.5


for i in range(5000):
	# add gaussian noise
	gauss = np.random.normal(param_mean_gaussian_noise,param_sigma_gaussian_noise,(TH,TW))
	gauss = gauss.reshape(TH,TW)
	gaussiannoise = np.uint8(np.clip(np.float32(img) + gauss,0,255))

	# add random gamma correction
	gamma=np.random.uniform(param_gamma_low,param_gamma_high)
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	gammacorrected = cv2.LUT(np.uint8(gaussiannoise), table)

	# randomly binarize image
	otsu_th,otsu = cv2.threshold(gammacorrected,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	th,binarized = cv2.threshold(gammacorrected,otsu_th + int(np.random.uniform(-param_var_otsu,param_var_otsu)),1,cv2.THRESH_BINARY_INV)
	pseudo_binarized = binarized * (255-gammacorrected)


	#cv2.imshow('tmp',np.hstack((img,gaussiannoise,pseudo_binarized)))
	canvas=np.zeros((3*TH,3*TW),dtype=np.uint8)
	canvas[TH:2*TH,TW:2*TW]=pseudo_binarized


	#add random affine transform
	A=np.array([[(TW + (.5*TW)) ,(TH)],
		[(TW) , (2*TH)],
		[(2*TW) , (2*TH)]],np.float32)
	B=np.array([[(TW + (.5*TW))*np.random.uniform(1-param_slant_correction_affine*param_displacement_affine,1+param_displacement_affine) ,(TH)*np.random.uniform(1-param_displacement_affine,1+param_displacement_affine)],
		[(TW) , (2*TH)],
		[(2*TW) , (2*TH)]],np.float32)

	M=cv2.getAffineTransform(A,B)
	warped=cv2.warpAffine(canvas,M,(3*TW,3*TH))
	
	#warp clean image to find out its boundding box
	points = np.argwhere(warped!=0)
	points = np.fliplr(points) 
	r = cv2.boundingRect(np.array([points]))

	final_image=np.uint8(warped[r[1]:r[1]+r[3],r[0]:r[0]+r[2]])
	
	cv2.imshow('display',final_image)
	k=cv2.waitKey()
	if chr(k&255) == 'q':
		break 
