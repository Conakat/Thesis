import numpy as np #type:ignore
import os
import cv2 #type:ignore

def process_video(video_directory, video_file, hand, median_filter_size):

    path = os.path.join(video_directory, video_file)
    cap = cv2.VideoCapture(path)

    width = int(cap.get(3))
    height = int(cap.get(4))

    crop_width = 42
    crop_height = 60

    params = dict(pyr_scale=0.5, levels=5,
    winsize=15, iterations=10, poly_n=7, poly_sigma=1.5,
    flags=0)
    
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        
    ret, frame1 = cap.read()
    frame1 = frame1[crop_height:height-crop_height,crop_width:width-crop_width]
    previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    if hand == 'left':
       previous_frame = cv2.flip(previous_frame, 1)

    filtered_previous_frame = cv2.medianBlur(previous_frame, median_filter_size)    
    #filtered_previous_frame = cv2.GaussianBlur(previous_frame, (median_filter_size, median_filter_size), 0)

    horizontal_list = np.zeros((height-2*crop_height, width-2*crop_width))
    vertical_list = np.zeros((height-2*crop_height, width-2*crop_width))
    
    vertical_flow = []
    horizontal_flow = []
    magnitude = []
    
    block_size = 4
    num_blocks_x = (height-2*crop_height) // block_size
    num_blocks_y = (width-2*crop_width) // block_size
    
    m = 0 

    while(cap.isOpened()):
        
        m+=1

        h_list = np.zeros((num_blocks_x, num_blocks_y))
        v_list = np.zeros((num_blocks_x, num_blocks_y))
        ret, frame2 = cap.read()

        if not ret:
            break

        cropped_frame = frame2[crop_height:height-crop_height,crop_width:width-crop_width]
        current_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                
        if hand == 'left':
            # Flip the frame horizontally
            current_frame = cv2.flip(current_frame, 1)

        filtered_current_frame = cv2.medianBlur(current_frame, median_filter_size) 
        #filtered_current_frame = cv2.GaussianBlur(current_frame, (median_filter_size, median_filter_size), 0)
        
        flow = cv2.calcOpticalFlowFarneback(filtered_previous_frame, filtered_current_frame, None, **params)
        
        horizontal_list = flow[..., 0]
        vertical_list = flow[..., 1]
        #print(np.shape(horizontal_flow))
        mag, ang = cv2.cartToPolar(horizontal_list, vertical_list)

        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                block_frame_h = horizontal_list[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size]
                block_frame_v = vertical_list[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size]
                #print(np.shape(block_frame))
                avg_value_h = np.mean(block_frame_h)
                avg_value_v = np.mean(block_frame_v)
                #print(avg_value)  
                h_list[i,j] = avg_value_h
                v_list[i,j] = avg_value_v
        horizontal_flow.append(h_list)
        vertical_flow.append(v_list)
        magnitude.append(mag)

        if hand == "":
            if m == 15:
                break
        
        filtered_previous_frame = filtered_current_frame

    horizontal_flow = np.asarray(horizontal_flow)
    vertical_flow = np.asarray(vertical_flow)
    magnitude = np.asarray(magnitude)

    horizontal_flow = np.sum(horizontal_flow, axis=0)
    vertical_flow = np.sum(vertical_flow, axis=0)
    magnitude = np.sum(magnitude, axis=0)
   
    abs_max1 = np.max(np.abs(horizontal_flow))
    abs_max2 = np.max(np.abs(vertical_flow))
    abs_max3 = np.max(np.abs(magnitude))

    horizontal_flow = horizontal_flow / abs_max1
    vertical_flow = vertical_flow / abs_max2
    magnitude = magnitude / abs_max3
    #cv2.normalize(horizontal_flow, horizontal_flow, 0, 1, cv2.NORM_MINMAX)
    #cv2.normalize(vertical_flow, vertical_flow, 0, 1, cv2.NORM_MINMAX)

    cap.release()
    '''
    a=sys.getsizeof(horizontal_flow)
    b=sys.getsizeof(vertical_flow)
    print(a,b)
    '''
    #return magnitude
    return horizontal_flow, vertical_flow

def equal_frames(frame1,frame2):

    return np.array_equal(frame1,frame2)

def diff(frame1, frame2):

    return cv2.absdiff(frame1, frame2)

def process_video2(video_directory, video_file, hand):
    
    filtered_frames = []
    path = os.path.join(video_directory, video_file)

    cap = cv2.VideoCapture(path)

    width = int(cap.get(3))
    height = int(cap.get(4))

    crop_width = 42
    crop_height = 60

    previous_frame = None
    num_of_duplicates = 0
    num_of_frames = 0
   
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        
    while(cap.isOpened()):
  
        ret, frame = cap.read()

        if ret == True:

            current_frame = frame[crop_height:height-crop_height,crop_width:width-crop_width]
            
            current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)

            if hand == 'left':
                current_frame = cv2.flip(current_frame, 1)

            if isinstance(previous_frame, np.ndarray):
                if equal_frames(current_frame, previous_frame):
                    num_of_duplicates += 1
                else:
                    num_of_frames += 1
                    if (num_of_frames == 16 and hand==""):
                       break
                    else:
                        previous_frame = current_frame
                        current_frame_filter = cv2.medianBlur(previous_frame, 3)
                        filtered_frames.append(current_frame_filter)
            else:
                previous_frame = current_frame
                current_frame_filter = cv2.medianBlur(previous_frame, 3)
                filtered_frames.append(current_frame_filter)
                num_of_frames += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
       

        #print(num_of_duplicates)
    filtered_frames = np.array(filtered_frames)
    #print(len(filtered_frames))
    #print(num_of_frames)
    total_sum = np.zeros((height-2*crop_height, width-2*crop_width))
    
    for i in range(len(filtered_frames)-1):
        current_frame = filtered_frames[i]
        next_frame = filtered_frames[i+1]
        sub = diff(current_frame, next_frame)
        total_sum = sub + total_sum

    
    cv2.normalize(total_sum, total_sum, 0, 255, cv2.NORM_MINMAX)

    total_sum = total_sum.astype(np.uint8)

    activity_map = total_sum
    
    return activity_map

def binarize_maps(map, threshold):

    block_size = 15
    height, width = np.shape(map)
    
    num_blocks_x = height // block_size
    num_blocks_y = width // block_size
    down_sampled_matrix = np.zeros((num_blocks_x, num_blocks_y))
    
    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            block = map[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size]
            avg_block = np.mean(block)
            if avg_block > threshold:
                down_sampled_matrix[i,j] = 255
            else:
                down_sampled_matrix[i,j] =  0

    #print(np.shape(down_matrix))
    down_sampled_matrix = down_sampled_matrix / 255
    down_sampled_matrix = down_sampled_matrix.astype(int)
    #print(down_matrix)
    return down_sampled_matrix