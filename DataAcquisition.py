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


class DataAcquisition:
    """
    Class responsible for data acquisition process using Hough Transform
    to track swinging sticks and extract angles of short and long sticks.
    """
    
    def __init__(self, video_path='data/first_24_hours_data.mp4', roi_x=130, roi_y=0, roi_w=400, roi_h=470, start_frame_multiplier=1, end_frame_multiplier=15*60*60,
                 auto_run=0, output_csv='continuous_long_short_angle_file.csv'):
        """
        Initialize the DataAcquisition class with video and processing parameters.
        """
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        
        # Video information
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Region of interest
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_w = roi_w
        self.roi_h = roi_h
        
        # Frame limiting
        self.start_frame = int(self.fps * start_frame_multiplier)
        self.end_frame = int(self.fps * end_frame_multiplier)
        self.auto_run = auto_run
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.current_frame = self.start_frame
        
        # Constants
        self.center = [195, 265]
        self.dist_to_poi_longshort_sticks = 70
        self.ppd_threshold = 12.5
        self.median_logic = 1
        
        """
         guardrail around left and right stand above which grouping detetction of lines for long, short stick  and 
         left stand and right stand fail
         this where the small gap in the dataset comes from
        """
        self.left_stand_range = [17, 25] 
        self.right_stand_range = [153, 162]
        
        # State variables
        self.past_long_theta = 0
        self.past_short_theta = 0
        self.past_long_theta_raw = 0
        
        self.left_stand_frame = []
        self.right_stand_frame = []
        self.long_frame = []
        self.short_frame = []
        
        self.prev_poi_x, self.prev_poi_y = self.center[0], self.center[1]
        self.prev_p1_x, self.prev_p1_y = self.center[0], self.center[1]
        self.prev_p2_x, self.prev_p2_y = self.center[0], self.center[1]
        self.s_end_x, self.s_end_y = self.center[0], self.center[1]
        self.l_end_x, self.l_end_y = self.center[0], self.center[1]
        
        self.cycle_counter = 0
        self.prev_short = []
        self.prev_long = []

        # Freeze fallback state
        self.prev_short_median = [[-80.0, 2.7925267]]
        self.prev_long_median  = [[261.0, 0.34906584]]
        self.prev_short_ppd    = None
        self.long_align_count  = 0

        # Initialization flags : skip jump rejection and alignment freeze
        # until the first valid detection has been accepted
        self.long_initialized  = False
        self.short_initialized = False
        
        # Output CSV file path
        self.output_csv = output_csv
    
    def find_slope(self, x1, y1, x2, y2):
        """Calculate slope from two points."""
        s = (y2 - y1) / ((x2 - x1) + 1e-6)
        return s
    
    def mean_values(self, array):
        """Calculate mean values of rho and theta of hough lines."""
        rho_values = [rho for rho, theta in array]
        theta_values = [theta for rho, theta in array]
        rho_mean = int(sum(rho_values) / len(rho_values))
        theta_mean = (sum(theta_values) / len(theta_values))
        return [rho_mean, theta_mean]
    
    def median_values(self, array):
        """Calculate median values of rho and theta of hough lines."""
        rho_values = [rho for rho, theta in array]
        theta_values = [theta for rho, theta in array]
        idx = np.argpartition(rho_values, len(rho_values) // 2)[len(rho_values) // 2]
        rho_median = rho_values[idx]
        theta_median = theta_values[idx]
        return [rho_median, theta_median]
    
    def hough_transform(self, image):
        """Perform hough transform on image frame."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 170, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 90)
        return lines
    
    def polar_to_rectangle(self, rho, theta):
        """Convert polar coordinates of Hough to End Point Coordinates of line."""
        theta_d = np.ceil(math.degrees(theta))
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        return x1, y1, x2, y2, theta_d
    
    def poi(self, long_r, long_t, short_r, short_t):
        """Calculate point of intersection of two lines."""
        lx1, ly1, lx2, ly2, l_theta = self.polar_to_rectangle(long_r, long_t)
        sx1, sy1, sx2, sy2, s_theta = self.polar_to_rectangle(short_r, short_t)
        la = ly2 - ly1
        lb = lx1 - lx2
        lc = (lx2 - lx1) * ly1 + (ly1 - ly2) * lx1
        sa = sy2 - sy1
        sb = sx1 - sx2
        sc = (sx2 - sx1) * sy1 + (sy1 - sy2) * sx1
        x = int((lb * sc - sb * lc) / ((la * sb - sa * lb) + 1e-6))
        y = int((sa * lc - la * sc) / ((la * sb - sa * lb) + 1e-6))
        return x, y
    
    def perpendicular_dist(self, x1, y1, x2, y2):
        """Calculate perpendicular distance between one line and one point."""
        ppd_num = abs(((y2 - y1) * self.center[0] + (x1 - x2) * self.center[1] + 
                       (y1 - y2) * x1 + (x2 - x1) * y1))
        ppd_den = np.sqrt(np.square(y2 - y1) + np.square(x1 - x2))
        ppd = ppd_num / ppd_den
        return ppd
    
    def ploi(self, rho, theta, x_point, y_point, d):
        """Calculate point of intersection at distance d from equation of line and one point."""
        xa, ya, xb, yb, theta_d = self.polar_to_rectangle(rho, theta)
        m = self.find_slope(xa, ya, xb, yb)
        x1 = int(x_point + (d) / np.sqrt(1 + np.square(m)))
        y1 = int(y_point + (d * m) / np.sqrt(1 + np.square(m)))
        x2 = int(x_point - (d) / np.sqrt(1 + np.square(m)))
        y2 = int(y_point - (d * m) / np.sqrt(1 + np.square(m)))
        return x1, y1, x2, y2
    
    def plot_line(self, image, rho, theta):
        """Draw line for given rho and theta on image frame."""
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    def find_points_from_poi(self, x1, y1, x2, y2, x3, y3, d):
        """Calculate points at distance d from point (x3, y3) along the line defined by (x1, y1) and (x2, y2)."""
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        unit_dx = dx / length
        unit_dy = dy / length
        point1_x = int(x3 + d * unit_dx)
        point1_y = int(y3 + d * unit_dy)
        point2_x = int(x3 - d * unit_dx)
        point2_y = int(y3 - d * unit_dy)
        return point1_x, point1_y, point2_x, point2_y
    
    def get_parallel_line_equation(self, x1, y1, x2, y2, u, v):
        """
        Get the line equation parameters for a line parallel to AB passing through C.
        Returns the slope and y-intercept for equation: y = mx + b
        For vertical lines, returns None for slope and the x-coordinate
        """
        if x2 == x1:
            return None, u
        slope = (y2 - y1) / (x2 - x1)
        intercept = v - slope * u
        return slope, intercept
    
    def draw_line_from_equation(self, image, slope, intercept, color=(0, 255, 0), thickness=2):
        """Draw a line on the image using the line equation y = mx + b"""
        height, width = image.shape[:2]
        if slope is None:
            x = int(intercept)
            if 0 <= x < width:
                cv2.line(image, (x, 0), (x, height-1), color, thickness)
            return image
        points = []
        y_left = slope * 0 + intercept
        if 0 <= y_left < height:
            points.append((0, int(y_left)))
        y_right = slope * (width-1) + intercept
        if 0 <= y_right < height:
            points.append((width-1, int(y_right)))
        if slope != 0:
            x_top = (0 - intercept) / slope
            if 0 <= x_top < width:
                points.append((int(x_top), 0))
        if slope != 0:
            x_bottom = (height-1 - intercept) / slope
            if 0 <= x_bottom < width:
                points.append((int(x_bottom), height-1))
        if len(points) >= 2:
            unique_points = []
            for point in points:
                if point not in unique_points:
                    unique_points.append(point)
            if len(unique_points) >= 2:
                cv2.line(image, unique_points[0], unique_points[1], color, thickness)
        return unique_points
    
    def rectangle_to_polar(self, x1, y1, x2, y2):
        """Convert End Point Coordinates of line to polar coordinates (rho, theta) for Hough transform."""
        a = y2 - y1
        b = -(x2 - x1)
        c = (x2 - x1) * y1 - (y2 - y1) * x1
        norm = math.sqrt(a*a + b*b)
        a_norm = a / norm
        b_norm = b / norm
        c_norm = c / norm
        rho = abs(c_norm)
        theta = math.atan2(b_norm, a_norm)
        if c_norm < 0:
            theta = theta + math.pi
        if theta < 0:
            theta += math.pi
        elif theta >= 2 * math.pi:
            theta -= math.pi
        return rho, theta

    @staticmethod
    def _circ_dist(a, b):
        """Circular distance between two angles in degrees (0-360 space)."""
        d = abs(a - b) % 360
        return min(d, 360 - d)

    def process_frame(self, image, copy):
        """
        Process a single frame to detect sticks and calculate angles.
        
        Returns:
        --------
        dict : Dictionary containing all calculated values for the frame
        """
        lines = self.hough_transform(image)
        
        # Draw the lines on copy to show where Hough detects
        for rho, theta in lines[:, 0]:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # cv2.imshow('Copy', copy)
        
        # Initialize flags and arrays
        long_align_flag = 0
        short_align_flag = 0
        bad_long_flag = 0
        rotate_flag = 0
        left_stand = [[261.0, 0.34906584], [259.0, 0.33161256]]
        right_stand = [[-80.0, 2.7925267]]
        
        long = []
        short = []
        long_median = []
        short_median = []
        
        short_theta_d = 0
        long_theta_d = 0
        short_negative_OTT_count = 0
        short_positive_OTT_count = 0
        left_theta_d = 0
        right_theta_d = 0
        
        # Process hough lines : classify into buckets
        if lines is not None:
            for rho, theta in lines[:, 0]:
                line_x1, line_y1, line_x2, line_y2, theta_d = self.polar_to_rectangle(rho, theta)
                
                cv2.circle(image, self.center, 1, (0, 0, 255), 2)
                
                ppd = self.perpendicular_dist(line_x1, line_y1, line_x2, line_y2)
                slope = self.find_slope(line_x1, line_y1, line_x2, line_y2)
                
                if self.left_stand_range[0] < theta_d < self.left_stand_range[1] and ppd < self.ppd_threshold:
                    left_stand.append([rho, theta])
                elif self.right_stand_range[0] < theta_d <= self.right_stand_range[1] and ppd < self.ppd_threshold:
                    right_stand.append([rho, theta])
                elif ppd < self.ppd_threshold:
                    long.append([rho, theta])
                else:
                    short.append([rho, theta])
        
        # Apply median logic
        if self.median_logic == 1:

            left_stand_median = self.median_values(left_stand)
            self.left_stand_frame.append(left_stand_median)
            left_stand_x1, left_stand_y1, left_stand_x2, left_stand_y2, left_theta_d = \
                self.polar_to_rectangle(left_stand_median[0], left_stand_median[1])
            
            right_stand_median = self.median_values(right_stand)
            self.right_stand_frame.append(right_stand_median)
            right_stand_x1, right_stand_y1, right_stand_x2, right_stand_y2, right_theta_d = \
                self.polar_to_rectangle(right_stand_median[0], right_stand_median[1])
            
            if len(long) > 0:
                long_median = self.median_values(long)
                self.long_frame.append(long_median)
                long_stick_x1, long_stick_y1, long_stick_x2, long_stick_y2, long_theta_d = \
                    self.polar_to_rectangle(long_median[0], long_median[1])
                
                long_slope, long_intercept = self.get_parallel_line_equation(
                    long_stick_x1, long_stick_y1, long_stick_x2, long_stick_y2,
                    self.center[0], self.center[1])
                
                long_median[0], long_median[1] = self.rectangle_to_polar(
                    long_stick_x1, long_stick_y1, long_stick_x2, long_stick_y2)
                
                cv2.line(image, (long_stick_x1, long_stick_y1), (long_stick_x2, long_stick_y2), (255, 255, 0), 1)

            elif len(long) == 0:
                long_align_flag = 1
            
            if len(short) > 0:
                short_median = self.median_values(short)
                self.short_frame.append(short_median)
                short_x1, short_y1, short_x2, short_y2, short_theta_d = \
                    self.polar_to_rectangle(short_median[0], short_median[1])
                if self.perpendicular_dist(short_x1, short_y1, short_x2, short_y2) < 216:
                    cv2.line(image, (short_x1, short_y1), (short_x2, short_y2), (30, 0, 250), 1)

                self.prev_short_ppd = self.perpendicular_dist(short_x1, short_y1, short_x2, short_y2)

            elif len(short) == 0:
                short = self.prev_short
                short_align_flag = 1
                print("not detecting short")
        
        # Handle long stick alignment (long stick indistinguishable from stand)
        if long_align_flag == 1:
            self.long_align_count += 1

            # Map corrected past angle to 0-180 for stand selection
            past_long_mapped = self.past_long_theta % 180
            if self.right_stand_range[0]-2 <= past_long_mapped <= self.right_stand_range[1]+2:
                long_theta_d = right_theta_d
                long_median = right_stand_median
            else:
                long_theta_d = left_theta_d
                long_median = left_stand_median

            long_stick_x1, long_stick_y1, long_stick_x2, long_stick_y2, long_theta_d = \
                self.polar_to_rectangle(long_median[0], long_median[1])

            # Restore correct hemisphere: if stick was >180 before, stand angle needs += 180
            if self.past_long_theta > 180:
                long_theta_d += 180

            # Freeze for short transient events (<=3 frames); only use stand for sustained alignment
            # Skip freeze if not yet initialized to avoid locking at 0
            if self.long_align_count <= 3 and self.long_initialized:
                long_theta_d = self.past_long_theta
                long_median = list(self.prev_long_median)
                long_stick_x1, long_stick_y1, long_stick_x2, long_stick_y2, _ = \
                    self.polar_to_rectangle(long_median[0], long_median[1])

            # Always freeze short during any long alignment
            short_theta_d = self.past_short_theta
            short_median = list(self.prev_short_median)
            short_align_flag = 0

        else:
            # Long cleanly detected : reset counter and save median
            self.long_align_count = 0
            self.prev_long_median = list(long_median)

        # Short stick fallback when Hough misses it
        if short_align_flag == 1:
            if len(self.prev_short) > 0:
                short_median = self.median_values(self.prev_short)
                short_theta_d = self.polar_to_rectangle(short_median[0], short_median[1])[4]
            else:
                short_theta_d = self.past_short_theta
                short_median = list(self.prev_short_median)
        
        # Calculate point of intersection
        if short_align_flag == 0:
            poi_x, poi_y = self.poi(long_median[0], long_median[1], short_median[0], short_median[1])
        else:
            _, _, poi_x, poi_y = self.ploi(long_median[0], long_median[1], self.center[0], self.center[1], self.dist_to_poi_longshort_sticks)
        
        cv2.line(image, (long_stick_x1, long_stick_y1), (long_stick_x2, long_stick_y2), (255, 255, 0), 1)
        
        other_p_x, other_p_y, actual_center_x, actual_center_y = self.find_points_from_poi(
            long_stick_x1, long_stick_y1, long_stick_x2, long_stick_y2,
            self.center[0], self.center[1], self.dist_to_poi_longshort_sticks)
        
        # Choose the closest point to poi
        if (other_p_x - poi_x)**2 + (other_p_y - poi_y)**2 < (actual_center_x - poi_x)**2 + (actual_center_y - poi_y)**2:
            poi_x, poi_y = other_p_x, other_p_y
        else:
            poi_x, poi_y = actual_center_x, actual_center_y
        
        # Freeze POI if it jumped too far (stale/wrong intersection)
        poi_diff = math.sqrt((poi_x - self.prev_poi_x)**2 + (poi_y - self.prev_poi_y)**2)
        if poi_diff > 70 and self.current_frame >= 1:
            poi_x, poi_y = self.prev_poi_x, self.prev_poi_y
        
        cv2.circle(image, (poi_x, poi_y), 1, (0, 0, 0), 2)
        
        x1, y1, x2, y2 = self.ploi(long_median[0], long_median[1], self.center[0], self.center[1], self.dist_to_poi_longshort_sticks)
        
        # Store raw Hough angle before hemisphere correction (for reference)
        self.past_long_theta_raw = long_theta_d

        # -----------------------------------------------------------------------
        # LONG STICK HEMISPHERE SELECTION : TRAJECTORY-BASED
        # -----------------------------------------------------------------------
        # Select the long stick's hemisphere based on its previous position.
        #
        # Using poi_x alone caused the stick to get stuck when an incorrect angle
        # was fed back into the next frame. The flip guard also prevented recovery
        # because it kept rejecting the 180-degree correction.
        #
        # There was also an issue near the 0/360 boundary, where a small change in
        # angle could be interpreted as a large jump and rejected.
        #
        # To avoid both issues, compare the two possible angles (raw and raw + 180)
        # with the previous angle using circular distance, and choose the closer one.
        # This keeps the angle consistent with the stick's previous trajectory.

        # Root cause of anomalies in this log:
        #   - poi_x feedback loop: long frozen at 2 degrees caused wrong poi_x, which
        #     kept long at 2 degrees . Flip guard (|2-182|=180) cancelled every +180.
        #   - Wrap-around lockup: raw abs() difference treated 0 ->358  as 358
        #     jump, MAX_LONG_JUMP rejected it, long frozen at 0degrees permanently.
        #
        # Fix: pick whichever hemisphere (raw OR raw+180) has the smallest
        # CIRCULAR distance to past_long_theta. This uses trajectory continuity
        # instead of poi_x, breaking the feedback loop and handling wrap-around.
        # -----------------------------------------------------------------------
        candidate_a = long_theta_d % 360           # raw Hough angle (0-360)
        candidate_b = (long_theta_d + 180) % 360   # opposite hemisphere

        if self._circ_dist(candidate_a, self.past_long_theta) <= \
           self._circ_dist(candidate_b, self.past_long_theta):
            long_theta_d = candidate_a
        else:
            long_theta_d = candidate_b

        # -----------------------------------------------------------------------
        # MAX_LONG_JUMP, spike rejection using CIRCULAR distance
        #
        # Old code used raw abs() difference which treated the 0degrees/360degrees wrap
        # as a huge jump (approx 358 degrees ) and rejected it, locking the stick at 0degrees.
        # Circular distance: min(diff, 360-diff) correctly sees 0degrees -> 358degrees as 2degrees.
        # -----------------------------------------------------------------------
        MAX_LONG_JUMP = 60  # degrees per frame : tune to rotation speed
        if self.long_initialized:
            circular_diff = self._circ_dist(long_theta_d, self.past_long_theta)
            if circular_diff > MAX_LONG_JUMP:
                long_theta_d = self.past_long_theta
        else:
            # First valid frame : accept freely, mark initialized
            self.long_initialized = True
        
        # Detect wrap from ~360 to ~0 (full CW cycle completed)
        if self.past_long_theta > 300 and long_theta_d < 60:
            self.cycle_counter += 1
        # Detect CCW full cycle (wrap-around backward: ~0 -> ~360)
        elif self.past_long_theta < 60 and long_theta_d > 300:
            self.cycle_counter -= 1
        
        # Compute continuous angle
        continuous_long_angle = long_theta_d + 360 * self.cycle_counter
        
        # Process short stick endpoints
        short_stick_x1, short_stick_y1, short_stick_x2, short_stick_y2, _ = \
            self.polar_to_rectangle(short_median[0], short_median[1])
        
        p1_x, p1_y, p2_x, p2_y = self.find_points_from_poi(
            short_stick_x1, short_stick_y1, short_stick_x2, short_stick_y2,
            poi_x, poi_y, 107)

        # Correct Hough 180degrees ambiguity for short stick
        # Both branches include % 360 to prevent values > 360 corrupting past_short_theta
        if self.past_short_theta - short_theta_d >= 100:
            p2_x, p1_x = p1_x, p2_x
            p2_y, p1_y = p1_y, p2_y
            short_theta_d += 180
            short_theta_d = short_theta_d % 360
        elif self.past_short_theta - short_theta_d < -100:
            short_theta_d += 180
            short_theta_d = short_theta_d % 360
            if abs(short_theta_d - self.past_short_theta) >= 170:
                print('clockwise')

        # Safe pixel sampling with bounds check
        h, w = copy.shape[:2]

        def safe_pixel(img, x, y):
            if 0 <= x < w and 0 <= y < h:
                return img[y, x]
            return (98, 163, 199)  # default: treat as blue (long stick end)

        b1, g1, r1 = safe_pixel(copy, p1_x, p1_y)
        b2, g2, r2 = safe_pixel(copy, p2_x, p2_y)
        
        # Determine end points based on color
        if b1 < 100 and b2 < 100:
            if math.dist([p1_x, p1_y], [self.s_end_x, self.s_end_y]) < math.dist([p2_x, p2_y], [self.s_end_x, self.s_end_y]):
                s_end_x, s_end_y = p1_x, p1_y
                l_end_x, l_end_y = p2_x, p2_y
            else:
                l_end_x, l_end_y = p1_x, p1_y
                s_end_x, s_end_y = p2_x, p2_y
        elif b1 > 100:
            l_end_x, l_end_y = p1_x, p1_y
            s_end_x, s_end_y = p2_x, p2_y
        else:
            l_end_x, l_end_y = p2_x, p2_y
            s_end_x, s_end_y = p1_x, p1_y
        
        cv2.circle(image, (s_end_x, s_end_y), 1, (255, 0, 0), 5)
        cv2.circle(image, (l_end_x, l_end_y), 1, (0, 255, 0), 5)
        
        # Short over-the-top counts
        if (357 <= self.past_short_theta) and (short_theta_d <= 3) and self.current_frame > 615:
            short_positive_OTT_count = 1
        elif (0 <= self.past_short_theta <= 2) and (357 <= short_theta_d <= 360) and self.current_frame > 615:
            short_negative_OTT_count = -1
        
        if bad_long_flag == 0:
            if abs(long_theta_d - self.past_long_theta) > 240 and self.current_frame > 0:
                rotate_flag = 1
        else:
            bad_long_flag = 0
        
        # Correct Hough 180degrees flip when short stick is near 0degrees/360degrees boundary
        if self.past_short_theta >= 330 and 170 <= short_theta_d <= 190:
            short_theta_d = (short_theta_d - 180) % 360

        # Mark short as initialized after first valid detection
        if not self.short_initialized and short_theta_d != 0:
            self.short_initialized = True
        
        # Save previous values for CSV output (before updating)
        past_long_theta_for_csv = self.past_long_theta
        past_short_theta_for_csv = self.past_short_theta
        
        # Update state variables for next frame
        self.prev_short = short
        self.prev_long = long
        # prev_short_median updated here; prev_long_median handled in long_align else branch
        self.prev_short_median = list(short_median) if len(short_median) > 0 else self.prev_short_median
        self.past_long_theta = long_theta_d
        self.past_short_theta = short_theta_d
        self.prev_poi_x = poi_x
        self.prev_poi_y = poi_y
        self.prev_p1_x = p1_x
        self.prev_p1_y = p1_y
        self.prev_p2_x = p2_x
        self.prev_p2_y = p2_y
        self.s_end_x = s_end_x
        self.s_end_y = s_end_y
        self.l_end_x = l_end_x
        self.l_end_y = l_end_y
        
        # Return frame data
        return {
            'frame': self.current_frame,
            'time': self.current_frame / self.fps,
            'cycle_counter': self.cycle_counter,
            'past_long_theta': past_long_theta_for_csv,
            'long_theta_d': long_theta_d,
            'continuous_long_angle': continuous_long_angle,
            'past_short_theta': past_short_theta_for_csv,
            'short_theta_d': short_theta_d,
            'short_positive_OTT_count': short_positive_OTT_count,
            'short_negative_OTT_count': short_negative_OTT_count,
            'rotate_flag': rotate_flag,
            'left_theta_d': left_theta_d,
            'right_theta_d': right_theta_d,
            'bad_long_flag': bad_long_flag,
            'image': image
        }