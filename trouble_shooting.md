MarcalAugmentor_v3

1. OpenCV resize error 
    
    Fix: target_width plus 1 pixel before applying resize

2. RuntimeWarning: overflow encountered in square

    Fix: change np.dtype from float32 to float64

3. Cannot go out of the while loop

    while(len(points)<1):
    	# random shear
    	shear_angle=np.random.uniform(param_min_shear,param_max_shear)
    	M=np.float32([[1,shear_angle,0],[0,1,0]])
    	sheared = cv2.warpAffine(canvas,M,(3*TW,3*TH),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC)

    	# random rotation
    	M = cv2.getRotationMatrix2D((3*TW/2,3*TH/2),np.random.uniform(-param_rotation,param_rotation),1)
    	rotated = cv2.warpAffine(sheared,M,(3*TW,3*TH),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC)

    	# random scaling
    	scaling_factor=np.random.uniform(1-param_scale,1+param_scale)
    	scaled = cv2.resize(rotated,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_CUBIC)

    	# detect cropping parameters
    	points = np.argwhere(scaled!=0)
    	points = np.fliplr(points)

    Fix: Add a count variable, if the loop has been over 100 times, then break the while loop
