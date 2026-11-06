'''
Author: Justice Essuman
Code for tracking swinging sticks and extracting angles of short and long sticks of the structure
'''

# Libraries
import cv2      
import numpy as np
import pandas as pd
import math
import statistics as st
import csv
from sklearn.cluster import KMeans


# Read the video
cap = cv2.VideoCapture('data/first_24_hours_data.mp4')  #('data/WIN_20241114_15_26_37_Pro.mp4')  

# Video information
width,height,fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT),cap.get(cv2.CAP_PROP_FPS)
print("Frame Width:",width)
print("Frame Height:",height)
print("FPS:",fps)

# Region of interest (custom based on video captured)
roi_x, roi_y, roi_w, roi_h = 130,0, 480, 470 #350, 460  # upated region of interest roi_h from 480 to 470

# Frame limittingqqqqqqqq
start_frame = fps*100 # Index of the first frame to saveq
end_frame = fps*72000*8            
auto_run = 0   
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
current_frame = start_frame

# Constants
cycle_counter = 0
center = [195,265]  
dist_to_poi_longshort_sticks = 70
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
prev_poi_x, prev_poi_y =  center[0], center[1]
prev_p1_x, prev_p1_y = center[0], center[1]
prev_p2_x, prev_p2_y = center[0], center[1]
s_end_x,s_end_y = center[0], center[1]
l_end_x,l_end_y = center[0], center[1]

median_logic = 1

def find_slope(x1,y1,x2,y2):
    '''Code to calculate slope from two points'''
    s = (y2-y1)/((x2-x1)+1e-6)
    return s

def mean_values(array):
    '''Code to calculate mean values of rho and theta of hough lines'''
    rho_values = [rho for rho, theta in array]
    theta_values =  [theta for rho, theta in array]
    rho_mean = int(sum(rho_values) / len(rho_values))
    theta_mean = (sum(theta_values) / len(theta_values))
    return [rho_mean,theta_mean]

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
    #lines = cv2.HoughLines(edges, 1, np.pi / 180, 85)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100)
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

def plot_line(rho,theta):
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


def get_parallel_line_equation(x1, y1, x2, y2, u, v):
    """
    Get the line equation parameters for a line parallel to AB passing through C.
    Returns the slope and y-intercept for equation: y = mx + b
    For vertical lines, returns None for slope and the x-coordinate
    
    Parameters:
    x1, y1: coordinates of point A
    x2, y2: coordinates of point B  
    u, v: coordinates of point C
    
    Returns:
    tuple: (slope, intercept) or (None, x_value) for vertical lines
    """
    # Handle vertical line case
    if x2 == x1:
        return None, u  # Vertical line at x = u
    
    # Calculate slope of line AB
    slope = (y2 - y1) / (x2 - x1)
    
    # Calculate y-intercept for line through C with same slope
    # y = mx + b, so b = y - mx = v - slope * u
    intercept = v - slope * u
    
    return slope, intercept


def draw_line_from_equation(image, slope, intercept, color=(0, 255, 0), thickness=2):
    """
    Draw a line on the image using the line equation y = mx + b
    
    Parameters:
    image: OpenCV image (numpy array)
    slope: slope of the line (None for vertical lines)
    intercept: y-intercept for normal lines, x-coordinate for vertical lines
    color: line color in BGR format (default: green)
    thickness: line thickness (default: 2)
    
    Returns:
    image with line drawn
    """
    height, width = image.shape[:2]
    
    # Handle vertical line case
    if slope is None:
        x = int(intercept)
        if 0 <= x < width:
            cv2.line(image, (x, 0), (x, height-1), color, thickness)
        return image
    
    # For non-vertical lines, find intersection points with image boundaries
    points = []
    
    # Check intersection with left edge (x = 0)
    y_left = slope * 0 + intercept
    if 0 <= y_left < height:
        points.append((0, int(y_left)))
    
    # Check intersection with right edge (x = width-1)
    y_right = slope * (width-1) + intercept
    if 0 <= y_right < height:
        points.append((width-1, int(y_right)))
    
    # Check intersection with top edge (y = 0)
    if slope != 0:  # Avoid division by zero
        x_top = (0 - intercept) / slope
        if 0 <= x_top < width:
            points.append((int(x_top), 0))
    
    # Check intersection with bottom edge (y = height-1)
    if slope != 0:  # Avoid division by zero
        x_bottom = (height-1 - intercept) / slope
        if 0 <= x_bottom < width:
            points.append((int(x_bottom), height-1))
    
    # Remove duplicate points and draw line
    if len(points) >= 2:
        # Remove duplicates
        unique_points = []
        for point in points:
            if point not in unique_points:
                unique_points.append(point)
        
        if len(unique_points) >= 2:
            cv2.line(image, unique_points[0], unique_points[1], color, thickness)
    
    return unique_points

def rectangle_to_polar(x1, y1, x2, y2):
    '''Code to convert End Point Coordinates of line to polar coordinates (rho, theta) for Hough transform'''
    
    # Calculate line parameters: ax + by + c = 0
    # From two points, we get: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1
    
    # Normalize the line equation so that a² + b² = 1
    norm = math.sqrt(a*a + b*b)
    a_norm = a / norm
    b_norm = b / norm
    c_norm = c / norm
    
    # rho is the perpendicular distance from origin to line
    # For normalized line equation ax + by + c = 0, rho = |c|
    rho = abs(c_norm)
    
    # theta is the angle of the normal to the line
    # The normal vector is (a_norm, b_norm)
    theta = math.atan2(b_norm, a_norm)

    
    # Ensure rho is positive and adjust theta accordingly
    if c_norm < 0:
        theta = theta + math.pi
    
    # Normalize theta to [0, 2π)
    #print(f"checking theta: {theta}")
    if theta < 0:
        theta += math.pi
    elif theta >= 2 * math.pi:
        theta -= math.pi

    return rho, theta


def angle_clustering(angles_data):
    
    left_stand_min_angle = 265
    left_stand_max_angle = 275

    right_stand_min_angle = 40
    right_stand_max_angle = 50
    
    left_stand_angles = []
    right_stand_angles = []
    variable_angles = []

    
    
    for angle in angles_data:
        if left_stand_min_angle <= angle <= left_stand_max_angle:
            left_stand_angles.append(angle)
        elif right_stand_min_angle <= angle <= right_stand_max_angle:
            right_stand_angles.append(angle)
        else:
            variable_angles.append(angle)

    # Prepare data for K-Means (reshape to a 2D array)
    variable_angles_np = np.array(variable_angles).reshape(-1, 1)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10) # n_init for modern scikit-learn
    kmeans.fit(variable_angles_np)
    cluster_labels = kmeans.labels_

    short_stick_angles = []
    long_stick_angles = []

    # Assuming the cluster with the lower mean angle is 'short_stick' and higher is 'long_stick'
    cluster_0_mean = np.mean(variable_angles_np[cluster_labels == 0])
    cluster_1_mean = np.mean(variable_angles_np[cluster_labels == 1])

    if cluster_0_mean < cluster_1_mean:
        short_stick_angles = variable_angles_np[cluster_labels == 0].flatten().tolist()
        long_stick_angles = variable_angles_np[cluster_labels == 1].flatten().tolist()
    else:
        short_stick_angles = variable_angles_np[cluster_labels == 1].flatten().tolist()
        long_stick_angles = variable_angles_np[cluster_labels == 0].flatten().tolist()

    print("\n--- Grouped Angles ---")
    print("Left Stand Angles:", left_stand_angles)
    print("Right Stand Angles:", right_stand_angles)
    print("Short Stick Angles:", short_stick_angles)
    print("Long Stick Angles:", long_stick_angles)
    

# Opening csv file to save values
print("center:",center) 
with open('continuous_long_short_angle_file.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['frame','time','previous_long','long','continuous_long_angle','previous_short', 'short','ott'])

# To Check Frame by Frame
    while current_frame <= end_frame:

    # To Check Full comment above line and uncomment below line
    # while cap.isOpened():

        print("###################################################################################################")
        print("FRAME:",current_frame)

        ret,image = cap.read()
        image = cv2.resize(image, (640,480),interpolation=cv2.INTER_LINEAR)
        image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        copy = image.copy()
        lines = hough_transform(image)

        # # Filter arrays
        long_align_flag=0
        short_align_flag=0
        bad_long_flag = 0
        rotate_flag = 0
        left_stand = [[261.0, 0.34906584], [259.0, 0.33161256]]
        right_stand = [[-80.0, 2.7925267]]
        long=[] 
        short = []
        prev_short = []
        angles_list = []
        
        left_stand_lines = []
        right_stand_lines = []
        variable_lines_pre_clustering = []
        LEFT_STAND_MIN_ANGLE = 265
        LEFT_STAND_MAX_ANGLE = 275
        RIGHT_STAND_MIN_ANGLE = 140
        RIGHT_STAND_MAX_ANGLE = 160

        # normal hough transform check
        if lines is not None:
            if len(lines) < 5:
                continue
            print("Lines:", lines.shape)
            # for rho, theta in lines[:, 0]:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                            # Calculate the angle of the line
                dx = x2 - x1
                dy = y2 - y1
                theta_radians = math.atan2(dy, dx)
                angle_degrees = math.degrees(theta_radians)

                # Adjust the angle to be between 0 and 360 degrees
                if angle_degrees < 0:
                    angle_degrees += 360
                    theta_d = angle_degrees
                #angles_list.append(angle_degrees)
        
                print(f"Line from ({x1}, {y1}) to ({x2}, {y2}) has an angle of {angle_degrees:.2f} degrees")
                
                
                if LEFT_STAND_MIN_ANGLE <= angle_degrees <= LEFT_STAND_MAX_ANGLE:
                    left_stand_lines.append((x1, y1, x2, y2, angle_degrees))
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                elif RIGHT_STAND_MIN_ANGLE <= angle_degrees <= RIGHT_STAND_MAX_ANGLE:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    right_stand_lines.append((x1, y1, x2, y2, angle_degrees))
                else:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    variable_lines_pre_clustering.append((x1, y1, x2, y2, angle_degrees))
                    
                    
                # --- Outputting the Grouped Lines ---
                print("--- Lines grouped into categories (Pre-Clustering) ---")

                print("\n--- Left Stand Lines ---")
                if left_stand_lines:
                    for x1, y1, x2, y2, angle in left_stand_lines:
                        print(f"Line from ({x1}, {y1}) to ({x2}, {y2}) has an angle of {angle:.2f} degrees")
                else:
                    print("No Left Stand lines detected in this run.")

                print("\n--- Right Stand Lines ---")
                if right_stand_lines:
                    for x1, y1, x2, y2, angle in right_stand_lines:
                        print(f"Line from ({x1}, {y1}) to ({x2}, {y2}) has an angle of {angle:.2f} degrees")
                else:
                    print("No Right Stand lines detected in this run.")

                print("\n--- Variable Lines (for Clustering) ---")
                if variable_lines_pre_clustering:
                    for x1, y1, x2, y2, angle in variable_lines_pre_clustering:
                        print(f"Line from ({x1}, {y1}) to ({x2}, {y2}) has an angle of {angle:.2f} degrees")
                else:
                    print("No Variable lines detected for clustering in this run.")

        #angle_clustering(angles_list)   
                


                # Draw the line (optional, for visualization)
                # x1theta_d = polar_to_rectangle(rho,theta)
    

        #         cv2.circle(image, center,1, (0, 0, 255), 2) # center
        #         # cv2.line(image, (x1, y1), (x2,y2), (0, 0, 255), 1)

        #        ppd = perpendicular_dist(x1,y1,x2,y2)
        #         slope = find_slope(x1,y1,x2,y2)

        #         # print("x1,y1,x2,y2 =",x1,",",y1,",",x2,",",y2)
        #         # print("Rho and Theta: ",rho,theta_d)
        #         # print("PPD ",ppd)
        #         # print("Slope:",slope)

                # if 17<theta_d<25 and ppd<ppd_threshold:
                #     print("left_stand")
        #             # print("x1,y1,x2,y2 =",x1,",",y1,",",x2,",",y2)
        #             # print("Rho and Theta: ",rho,theta_d)
        #             # print("PPD",ppd)
        #             # cv2.line(image, (x1, y1), (x2,y2), (0, 0, 255), 1)
        #             left_stand.append([rho,theta])
        #             # left_stand.append([x1,y1,x2,y2])

        #         elif 155<theta_d<162 and ppd<ppd_threshold:
        #             # print("right_stand")
        #             # print("x1,y1,x2,y2 =",x1,",",y1,",",x2,",",y2)
        #             # print("Rho and Theta: ",rho,theta_d)
        #             # print("PPD",ppd)
        #             # cv2.line(image, (x1, y1), (x2,y2), (0, 0, 255), 1)
        #             right_stand.append([rho,theta])
        #             # right_stand.append([x1,y1,x2,y2])
                
        #         elif ppd<ppd_threshold:
        #             # print("long")
        #             # print("x1,y1,x2,y2 =",x1,",",y1,",",x2,",",y2)
        #             # print("Rho and Theta: ",rho,theta_d)
        #             # print("PPD",ppd)
        #             # cv2.line(image, (x1, y1), (x2,y2), (0, 0, 255), 1)
        #             long.append([rho,theta])
        #             # long.append([x1,y1,x2,y2])

        #         else:
        #             #print("short")
        #             # print("x1,y1,x2,y2 =",x1,",",y1,",",x2,",",y2)
        #             #print("Rho and Theta: ",rho,theta_d)
        #             # cv2.line(image, (x1, y1), (x2,y2), (0, 0, 255), 1)
        #             # short.append([x1,y1,x2,y2])
        #             short.append([rho,theta])
        #             # short.append([x1,y1,x2,y2])

        
        # if median_logic==1:

        #     # MEDIAN LOGIC
        #     left_stand_median = median_values(left_stand)
        #     left_stand_frame.append(left_stand_median)
        #     left_stand_x1,left_stand_y1,left_stand_x2,left_stand_y2,left_theta_d = polar_to_rectangle(left_stand_median[0],left_stand_median[1])
        #     # cv2.line(image, (left_stand_x1,left_stand_y1), (left_stand_x2,left_stand_y2), (0, 255, 0), 1)

        #     right_stand_median = median_values(right_stand)
        #     right_stand_frame.append(right_stand_median)
        #     right_stand_x1,right_stand_y1,right_stand_x2,right_stand_y2, right_theta_d = polar_to_rectangle(right_stand_median[0],right_stand_median[1])
        #     # cv2.line(image, (right_stand_x1,right_stand_y1), (right_stand_x2,right_stand_y2), (0, 255, 0), 1)
        #     print(f"LEFT MEDIAN: {left_stand_median} and RIGHT MEDIAN: {right_stand_median}")

        #     if len(long)>0:
        #         long_median = median_values(long) 
        #         long_frame.append(long_median)
        #         #print("LONG MEDIAN: ",long_median)
        #         x1,y1,x2,y2,long_theta_d = polar_to_rectangle(long_median[0],long_median[1])
                
        #         # Calculate line equation for the parallel line through C
        #         long_slope, long_intercept = get_parallel_line_equation(x1,y1, x2,y2, center[0], center[1])
        #         #print(f"Long Slope: {long_slope}, Long Intercept: {long_intercept}")
        #         unique_points = draw_line_from_equation(image, long_slope, long_intercept, color=(255, 255, 0), thickness=1)  # Yellow line
        #         # print(f"Unique Points: {unique_points} and x1,y1,x2,y2: {x1, y1, x2 ,y2}")
        #         long_median[0],long_median[1] = rectangle_to_polar(x1,y1,x2,y2)
        #         #cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)
        #     else:
        #         long_align_flag=1

        #     if len(short)>0:
        #         short_median = median_values(short)
        #         short_frame.append(short_median)
        #         x1,y1,x2,y2,short_theta_d = polar_to_rectangle(short_median[0], short_median[1])
        #         if perpendicular_dist(x1,y1,x2,y2) < 216:              # threhold for horizontal line at the bottom
        #             #print("SHORT THETA: ",short_theta_d)
        #             cv2.line(image, (x1,y1), (x2,y2), (30, 0, 250), 1)
        #     else:
        #         short = prev_short
        #         #short_align_flag=1
        # print("SHORT MEDIAN: ",short_median)
                

        # #write missing logic for long and short
        # if long_align_flag==1:
        #     # print("Long Aligned")
            
        #     if 153<past_long_theta_raw<162:
        #         long_theta_d=right_theta_d
        #         long_median = right_stand_median
        #     else:
        #         long_theta_d=left_theta_d 
        #         long_median = left_stand_median 


        # if short_align_flag ==1:
        #     if 153<past_short_theta<162:
        #         short_theta_d =right_theta_d
        #         short_median = right_stand_median
        #     else:
        #         short_theta_d =left_theta_d 
        #         short_median = left_stand_median 

        #     short_median = long_median
        #     short_theta_d = long_theta_d
            
            
        # if 177<= abs(past_short_theta-short_theta_d)<=185:
        #     if short_theta_d>=180:
        #         short_theta_d+=180
        #     else:
        #         short_theta_d-=180

            
        # if short_align_flag == 0:
        #     #print(f"Long: {long_median} Short: {short_median}") #long_mean[0],long_mean[1],short_mean[0],short_mean[1])
        #     poi_x,poi_y = poi(long_median[0],long_median[1],short_median[0],short_median[1])
        #     print("here")
        # else:
        #     _,_,poi_x,poi_y = ploi(long_median[0],long_median[1],center[0],center[1],dist_to_poi_longshort_sticks)
        #     print("there")
        # print("POI:",poi_x,poi_y)
        
        

        
        
             
        # #making redunduncy lock on center stationary not wabbling
        # long_stick_x1, long_stick_y1, long_stick_x2, long_stick_y2, _ = polar_to_rectangle(long_median[0],long_median[1])
        
        # other_p_x, other_p_y, actual_center_x, actual_center_y = find_points_from_poi(long_stick_x1, long_stick_y1, long_stick_x2, long_stick_y2,center[0],center[1],dist_to_poi_longshort_sticks)
        # # #tracking center
        
        # #choose the closest point to poi
        # if (other_p_x-poi_x)**2 + (other_p_y-poi_y)**2 < (actual_center_x-poi_x)**2 + (actual_center_y-poi_y)**2:
        #     poi_x,poi_y = other_p_x,other_p_y
        # else:
        #     poi_x,poi_y = actual_center_x,actual_center_y 
        
        # # the change in distance between prev poi and current poi must be less than 10 pixels
        # poi_diff = math.sqrt((poi_x-prev_poi_x)**2 + (poi_y-prev_poi_y)**2)
        # print("POI Distance Change:",poi_diff )
        # if poi_diff > 70 and current_frame >= 1: # poi dffifference threshold
        #     poi_x,poi_y = prev_poi_x,prev_poi_y
        # cv2.circle(image, (poi_x,poi_y),1, (0,0,0), 2)  
        

        # x1,y1,x2,y2 = ploi(long_median[0],long_median[1],center[0],center[1],dist_to_poi_longshort_sticks)
        # # cv2.circle(image, (x1,y1),1, (0, 0, 255), 2)
        # # cv2.circle(image, (x2,y2),1, (0, 0, 255), 2)
        
        # past_long_theta_raw = long_theta_d

        # # write long angle logic
        # # print("Original Long Angle:",long_theta_d)
        # if (poi_x<=center[0])  :      
        # # if poi_y<center[1]:
        #     long_theta_d = long_theta_d
        # else:
        #     long_theta_d +=180 


        # if 177<= abs(past_long_theta-long_theta_d)<=185:
        #     if long_theta_d>=180:
        #         long_theta_d-=180
        #     else:
        #         long_theta_d+=180
        
        # # Detect wrap from ~360 to ~0 (full cycle completed)
        # if past_long_theta > 300 and long_theta_d < 60:
        #     cycle_counter += 1
        # # Detect CCW full cycle (wrap-around backward: ~0 -> ~360)
        # elif past_long_theta < 60 and long_theta_d > 300:
        #     cycle_counter -= 1
        # # print(f"===============================================")
        # # print(f"Cycle completed! Total cycles: {cycle_counter}")
        # # print(f"===============================================")
        # # Compute continuous angle
        # continuous_long_angle = long_theta_d + 360 * cycle_counter
        # # print(f"Continuous Long Angle: {continuous_long_angle}")


        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # short_stick_x1,short_stick_y1,short_stick_x2,short_stick_y2,_ = polar_to_rectangle(short_median[0],short_median[1])
        # # cv2.line(image, (x1,y1), (x2,y2), (255, 255, 0), 1)q
        # p1_x,p1_y,p2_x,p2_y = find_points_from_poi( short_stick_x1,short_stick_y1,short_stick_x2,short_stick_y2,poi_x,poi_y,107)
        
               
        # try:
        #     b1, g1, r1 = copy[p1_y,p1_x]
        #     b2, g2, r2 = copy[p2_y,p2_x]
            
        # except:
        #     b1, g1, r1 = 98, 163, 199
        #     b2, g2, r2 = 129, 125, 124
            

        # print(f" {b1}, {g1}, {r1} ||  {b2}, {g2}, {r2} , -diff: {g2-g1}")
        # if b1 < 100 and b2 < 100: # long end color as green
        #     # print("bad flag")
        #     # #print current and previous p1 and p2
        #     # print(f" current p1 : {[p1_x,p1_y]} short_prev : {[s_end_x,s_end_y]}")
        #     # print(f" current p2 : {[p2_x,p2_y]} long_prev : {[l_end_x,l_end_y]}")
        #     # print("distances:",math.dist([p1_x,p1_y],[s_end_x,s_end_y]),math.dist([p2_x,p2_y],[s_end_x,s_end_y]))
        #     if math.dist([p1_x,p1_y],[s_end_x,s_end_y]) < math.dist([p2_x,p2_y],[s_end_x,s_end_y]):
        #         print("p1 is closer to short end")
        #         s__end_x, s_end_y = p1_x,p1_y
        #         l_end_x, l_end_y = p2_x ,p2_y 
        #     else:
        #         print("p2 is closer to short end")
        #         l_end_x, l_end_y = p1_x ,p1_y 
        #         s_end_x, s_end_y = p2_x,p2_y
        # elif b1 > 100: # long end color as green
        #     l_end_x, l_end_y = p1_x,p1_y
        #     s_end_x, s_end_y = p2_x,p2_y

        # else:
        #     l_end_x, l_end_y = p2_x,p2_y
        #     s_end_x, s_end_y = p1_x,p1_y 
        
        # cv2.circle(image, (s_end_x,s_end_y),1, (255, 0, 0), 5)  #
        # cv2.circle(image, (l_end_x,l_end_y),1, (0, 255, 0), 5)  #
        
        # if s_end_x < l_end_x and s_end_y < l_end_y and short_median[1] <= 3.0019662:
        #     short_theta_d = short_theta_d + 180
        # #elif s_end_x < l_end_x and s_end_y > l_end_y and :
        # #     short_theta_d = short_theta_d + 180
        # # elif s_end_x < l_end_x and l_end_y > s_end_y:
        # #     short_theta_d = short_theta_d + 180
            
            
            
        # # if abs(short_theta_d - past_short_theta) > 100:
        # #     short_theta_d = past_short_theta)
  
        # # elif p1_x < p2_x and short_median[1] > 0.0:
        # #     short_theta_d = short_theta_d + 180
        # # elif p1_x < p2_x and short_median[1] < 0.0:
        # #     short_theta_d = short_theta_d
        
            
        # # # Write angle logic
        # # print("Short",s_end_x,s_end_y,"Long",l_end_x,l_end_y)
        
        # # cv2.circle(image, (p1_x,p1_y),1, (255, 0, 0), 5)  #
        # # cv2.circle(image, (p2_x,p2_y),1, (0, 255, 0), 5)  #
        # print("++++++++++++++++++++++++++++++++++++++++++",(s_end_x,s_end_y),(l_end_x,l_end_y))
        # # print(image[p1_y,p1_x],gray[p1_y,p1_x],copy[p1_y,p1_x],image[p2_y,p2_x],gray[p2_y,p2_x],copy[p2_y,p2_x])
        # # cv2.imshow("GRAY",gray)
 
        
        
   
 
      
        # # cv2.imshow("COPY",copy)

        # # if bad_long_flag==0:
        # #     if abs(long_theta_d-past_long_theta)>240 and current_frame>0:
        # #         rotate_flag=1
        # # else:
        # #     bad_long_flag=0                                       
        
       
        # writer.writerow([current_frame,current_frame/fps,past_long_theta,long_theta_d, continuous_long_angle, past_short_theta, short_theta_d, rotate_flag])
        # # print("PAST LONG:",past_long_theta,"CURRENT LONG:",long_theta_d)
        # print("Frame:",current_frame,"Time: %.4f",current_frame*1/fps,"Previous Long Angle:", past_long_theta,"Long Angle:", long_theta_d, "Previous Short Angle:", past_short_theta,"Short Angle:", short_theta_d,"Left Angle:", left_theta_d,"Right Angle:", right_theta_d,"Bad Data:",bad_long_flag,"Rotate:",rotate_flag)
        # # print("Frame:",current_frame,"Time:",current_frame*1/fps,"Previous Long Angle:", past_long_theta,"Long Angle:", long_theta_d,"Short Angle:", short_theta_d,"Left Angle:", left_theta_d,"Right Angle:", right_theta_d,"Bad Data:",bad_long_flag,"Rotate:",rotate_flag)
        # #print("Frame: {0:d} Time: {1:.4f} Previous Long: {2:.1f} Long: {3:.1f} Short: {4:.1f} Left: {5:.1f} Right: {6:.1f} Bad Long: {7:d} Rotate: {8:d}".format(current_frame,current_frame/fps,past_long_theta,long_theta_d,short_theta_d,left_theta_d,right_theta_d,bad_long_flag,rotate_flag))
        
        # prev_short = short
        
        # past_long_theta = long_theta_d
        # past_short_theta = short_theta_d
        
        # prev_poi_x = poi_x
        # prev_poi_y = poi_y
        
        # prev_p1_x = p1_x
        # prev_p1_y = p1_y
        # prev_p2_x = p2_x
        # prev_p2_y = p2_y
           
        # current_frame += 1

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