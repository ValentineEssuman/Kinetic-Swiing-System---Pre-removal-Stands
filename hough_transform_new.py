'''
Author: Justice Essuman
Code for tracking swinging sticks and extracting angles of short and long sticks of the structure
'''

# Libraries
import cv2
import cv2.img_hash
import numpy as np
import pandas as pd
import math
import statistics as st
import csv

# Read the video
cap = cv2.VideoCapture('data/WIN_20240502_16_43_40_Pro.mp4')

# Video information
width,height,fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT),cap.get(cv2.CAP_PROP_FPS)
print("Frame Width:",width)
print("Frame Height:",height)
print("FPS:",fps)

# Region of interest (custom based on video captured)
roi_x, roi_y, roi_w, roi_h = 130,0,350,480

# Frame limitting
start_frame =fps*0 # Index of the first frame to save
end_frame = fps*1
auto_run = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
current_frame = start_frame

# Constants
center = [179,248]
left_stand_vals = []
right_stand_vals = []
ppd_threshold = 12.5
past_long_theta = 0
past_short_theta = 0
past_long_theta_raw=0

left_stand_frame=[]
right_stand_frame=[]
long_frame=[]
short_frame=[]

def find_slope(x1,y1,x2,y2):
    '''Code to calculate slope from two points'''
    s = (y2-y1)/((x2-x1)+1e-6)
    return s

def median_values(array):
    '''Code to calculate median values of rho and theta of hough lines'''
    rho_values = [rho for rho, theta in array]
    theta_values =  [theta for rho, theta in array]
    idx = np.argpartition(rho_values, len(rho_values) // 2)[len(rho_values) // 2]
    rho_median = rho_values[idx]
    theta_median = theta_values[idx]
    return [rho_median,theta_median]

def hough_transform(image):
    '''Code to perform hough transform on image frame'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0) 
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 90)
    return lines

def polar_to_rectangle(rho,theta):
    '''Code to convert polar coordinates of Hough to End Point Coordinates of line'''
    theta_d = np.ceil(math.degrees(theta))
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return x1,y1,x2,y2,theta_d

def poi(long_r,long_t,short_r,short_t):
    '''Code to calculate point of intersection of two lines'''
    lx1,ly1,lx2,ly2,l_theta = polar_to_rectangle(long_r,long_t)
    sx1,sy1,sx2,sy2,s_theta = polar_to_rectangle(short_r,short_t)
    la = ly2-ly1
    lb = lx1-lx2
    lc = (lx2-lx1)*ly1+(ly1-ly2)*lx1
    sa = sy2-sy1
    sb = sx1-sx2
    sc = (sx2-sx1)*sy1+(sy1-sy2)*sx1
    x = int((lb*sc-sb*lc)/((la*sb-sa*lb)+1e-6))
    y = int((sa*lc-la*sc)/((la*sb-sa*lb)+1e-6))
    return x,y

def perpendicular_dist(x1,y1,x2,y2):
    '''Code to calculate perpendicular distance between one line and one point'''
    ppd_num = abs(((y2-y1)*center[0]+(x1-x2)*center[1]+(y1-y2)*x1+(x2-x1)*y1))
    ppd_den = np.sqrt(np.square(y2-y1)+np.square(x1-x2))
    ppd = ppd_num/ppd_den
    return ppd

def ploi(rho,theta,x_point,y_point,d):
    '''Code to calculate point of intersection at distance d from equation of line and one point'''
    xa,ya,xb,yb,theta_d = polar_to_rectangle(rho,theta)
    m = find_slope(xa,ya,xb,yb)
    x1 = int(x_point + (d)/np.sqrt(1+np.square(m)))
    y1 = int(y_point + (d*m)/np.sqrt(1+np.square(m)))
    x2 = int(x_point - (d)/np.sqrt(1+np.square(m)))
    y2 = int(y_point - (d*m)/np.sqrt(1+np.square(m)))
    return x1,y1,x2,y2

def plot_line(image,rho,theta):
    '''Code to draw line for given rho and theta on image frame'''
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))    
    cv2.line(image, (x1,y1), (x2,y2), (0,0, 255), 1)

def find_points_from_poi(x1, y1, x2, y2, x3, y3, d):
    # Calculate the direction vector of the line
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate the length of the line segment
    length = math.sqrt(dx**2 + dy**2)
    
    # Normalize the direction vector
    unit_dx = dx / length
    unit_dy = dy / length
    
    # Calculate the points at distance d in both directions from the third point
    point1_x = int(x3 + d * unit_dx)
    point1_y = int(y3 + d * unit_dy)
    
    point2_x = int(x3 - d * unit_dx)
    point2_y = int(y3 - d * unit_dy)
    
    return point1_x, point1_y, point2_x, point2_y

# # Opening csv file to save values
# with open('dummy_file.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['frame','time','previous_long','long','ott'])


############ MAIN LOOP ##############

# To Check Frame by Frame
while current_frame <= end_frame:

# To Check Full comment above line and uncomment below line
# while cap.isOpened():

    print("FRAME:",current_frame)

    ret,image = cap.read()
    image = cv2.resize(image,(640,480),interpolation=cv2.INTER_LINEAR)
    image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    copy = image.copy()
    lines = hough_transform(image)

    # # Filter arrays
    long_align_flag=0
    short_align_flag=0
    bad_long_flag = 0
    rotate_flag = 0
    left_stand = []
    right_stand = []
    long=[]
    short = []

    # normal hough transform check
    if lines is not None:
        for rho, theta in lines[:, 0]:
            x1,y1,x2,y2,theta_d = polar_to_rectangle(rho,theta)
            cv2.circle(image, center,1, (0, 0, 255), 2)

            ppd = perpendicular_dist(x1,y1,x2,y2)
            slope = find_slope(x1,y1,x2,y2)

            # print("x1,y1,x2,y2 =",x1,",",y1,",",x2,",",y2)
            # print("Rho and Theta: ",rho,theta_d)
            # print("PPD ",ppd)
            # print("Slope:",slope)

            if 17<theta_d<25 and ppd<ppd_threshold:
                # print("left_stand")
                # print("x1,y1,x2,y2 =",x1,",",y1,",",x2,",",y2)
                # print("Rho and Theta: ",rho,theta_d)
                # print("PPD",ppd)
                # cv2.line(image, (x1, y1), (x2,y2), (0, 0, 255), 1)
                left_stand.append([rho,theta])
                # left_stand.append([x1,y1,x2,y2])

            elif 155<theta_d<163 and ppd<ppd_threshold:
                # print("right_stand")
                # print("x1,y1,x2,y2 =",x1,",",y1,",",x2,",",y2)
                # print("Rho and Theta: ",rho,theta_d)
                # print("PPD",ppd)
                # cv2.line(image, (x1, y1), (x2,y2), (0, 0, 255), 1)
                right_stand.append([rho,theta])
                # right_stand.append([x1,y1,x2,y2])
            
            else:
                if ppd<ppd_threshold:
                    # print("long")
                    # print("x1,y1,x2,y2 =",x1,",",y1,",",x2,",",y2)
                    # print("Rho and Theta: ",rho,theta_d)
                    # print("PPD",ppd)
                    # cv2.line(image, (x1, y1), (x2,y2), (0, 0, 255), 1)
                    long.append([rho,theta])
                    # long.append([x1,y1,x2,y2])

                else:
                    # print("short")
                    # print("x1,y1,x2,y2 =",x1,",",y1,",",x2,",",y2)
                    # print("Rho and Theta: ",rho,theta_d)
                    # print("PPD",ppd)
                    # cv2.line(image, (x1, y1), (x2,y2), (0, 0, 255), 1)
                    short.append([rho,theta])
                    # short.append([x1,y1,x2,y2])

    


    # MEDIAN LOGIC
    left_stand_median = median_values(left_stand)
    left_stand_frame.append(left_stand_median)
    # print("LEFT MEDIAN: ",left_stand_median)
    x1,y1,x2,y2,left_theta_d = polar_to_rectangle(left_stand_median[0],left_stand_median[1])
    # cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)

    right_stand_median = median_values(right_stand)
    right_stand_frame.append(right_stand_median)
    # print("RIGHT MEDIAN: ",right_stand_median)
    x1,y1,x2,y2,right_theta_d = polar_to_rectangle(right_stand_median[0],right_stand_median[1])
    # cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)

    if len(long)>0:
        long_median = median_values(long)
        long_frame.append(long_median)
        # print("LONG MEDIAN: ",long_median)
        x1,y1,x2,y2,long_theta_d = polar_to_rectangle(long_median[0],long_median[1])
        cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)
    else:
        long_align_flag=1

    if len(short)>0:
        short_median = median_values(short)
        short_frame.append(short_median)
        # print("SHORT MEDIAN: ",short_median)
        x1,y1,x2,y2,short_theta_d = polar_to_rectangle(short_median[0],short_median[1])
        cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)
    else:
        short_align_flag=1





    #write missing logic for long and short

    if long_align_flag==1:
        # print("Long Aligned")
        
        if 154<past_long_theta_raw<168:
            long_theta_d=right_theta_d
            long_median = right_stand_median
        else:
            long_theta_d=left_theta_d 
            long_median = left_stand_median

    if short_align_flag ==1:
        short_median = long_median
        short_theta_d = long_theta_d

    # plot_line(image,long_median[0],long_median[1])
    # plot_line(image,short_median[0],short_median[1])
    
    # write point of intersection logic

    if short_align_flag==0:
        # poi_x,poi_y = poi(long_mean[0],long_mean[1],short_mean[0],short_mean[1])
        poi_x,poi_y = poi(long_median[0],long_median[1],short_median[0],short_median[1])
    else:
        _,_,poi_x,poi_y = ploi(long_median[0],long_median[1],center[0],center[1],70)
    # print("POI:",poi_x,poi_y)

    cv2.circle(image, (poi_x,poi_y),1, (255, 0, 255), 2)

    x1,y1,x2,y2 = ploi(long_median[0],long_median[1],center[0],center[1],70)
    # cv2.circle(image, (x1,y1),1, (0, 0, 255), 2)
    # cv2.circle(image, (x2,y2),1, (0, 0, 255), 2)
    
    past_long_theta_raw = long_theta_d

    # write long angle logic
    # print("Original Long Angle:",long_theta_d)

    if (poi_x<=center[0]):
    # if poi_y<center[1]:
        long_theta_d+=180
    else:
        long_theta_d = long_theta_d


    if 175<= abs(past_long_theta-long_theta_d)<=185:
        if long_theta_d>=180:
            long_theta_d-=180
        else:
            long_theta_d+=180
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x1,y1,x2,y2,_ = polar_to_rectangle(short_median[0],short_median[1])
    # cv2.line(image, (x1,y1), (x2,y2), (255, 255, 0), 1)
    p1_x,p1_y,p2_x,p2_y = find_points_from_poi(x1,y1,x2,y2,poi_x,poi_y,100)
    cv2.circle(image, (p1_x,p1_y),1, (0, 255, 255), 2)
    cv2.circle(image, (p2_x,p2_y),1, (0, 0, 255), 2)
    
    # print(p1_x,p1_y,p2_x,p2_y)
    # print(image[p1_y,p1_x],gray[p1_y,p1_x],copy[p1_y,p1_x],image[p2_y,p2_x],gray[p2_y,p2_x],copy[p2_y,p2_x])
    # cv2.imshow("GRAY",gray)

    print(p1_x,p1_y,gray[p1_y,p1_x],p2_x,p2_y,gray[p2_y,p2_x])

    # cv2.imshow("COPY",copy)

    if bad_long_flag==0:
        if abs(long_theta_d-past_long_theta)>240 and current_frame>0:
            rotate_flag=1
    else:
        bad_long_flag=0

    # writer.writerow([current_frame,current_frame/fps,past_long_theta,long_theta_d,rotate_flag])
    print("PAST LONG:",past_long_theta,"CURRENT LONG:",long_theta_d)
    # print("Frame:",current_frame,"Time: %.4f",current_frame*1/fps,"Previous Long Angle:", past_long_theta,"Long Angle:", long_theta_d,"Short Angle:", short_theta_d,"Left Angle:", left_theta_d,"Right Angle:", right_theta_d,"Bad Data:",bad_long_flag,"Rotate:",rotate_flag)
    # print("Frame:",current_frame,"Time:",current_frame*1/fps,"Previous Long Angle:", past_long_theta,"Long Angle:", long_theta_d,"Short Angle:", short_theta_d,"Left Angle:", left_theta_d,"Right Angle:", right_theta_d,"Bad Data:",bad_long_flag,"Rotate:",rotate_flag)
    # print("Frame: {0:d} Time: {1:.4f} Previous Long: {2:.1f} Long: {3:.1f} Short: {4:.1f} Left: {5:.1f} Right: {6:.1f} Bad Long: {7:d} Rotate: {8:d}".format(current_frame,current_frame/fps,past_long_theta,long_theta_d,short_theta_d,left_theta_d,right_theta_d,bad_long_flag,rotate_flag))
    past_long_theta = long_theta_d
    past_short_theta = short_theta_d
    
    
    
    # point = [179,35]
    # src = np.uint8([[image[point[0],point[1]]]])
    # hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    # # print(src,hsv)
    # pts = np.array([[175,30],[185,30],[185,40],[175,40]])
    # cv2.circle(image,point,1, (0, 0, 255), 2)
    # cv2.polylines(image,[pts],1,(0,0,255),1)

    
    # print("###############")        


    current_frame += 1

    # print(filtered_lines)

    # TO CHECK OUTPUT WITHOUT THRESHOLD

    cv2.imshow('Hough Lines Detection', image)
    if cv2.waitKey(1) &0xFF == ord('q'): # to see whole video
        break

    if auto_run==0:
        cv2.waitKey(0) #press q to go frame by frame

cv2.destroyAllWindows()


frame = np.arange(start_frame,end_frame)
time_stamp = np.arange(start_frame/fps,end_frame/fps,1/fps)

print('DONE')


'''
default_long_angle
turn_count ++ if 360>0
turn count -- if 0>360
cumulative_long_angle = default_long_angle + turn_count * 360

'''