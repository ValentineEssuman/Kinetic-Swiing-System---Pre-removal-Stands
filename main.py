from DataAcquisition import DataAcquisition
import cv2
import csv

#time secs start 209
def run_acquisition():
    # Create instance
    # we start reading from the 209th
    daq = DataAcquisition(video_path='data/first_24_hours_data.mp4', roi_x=130, roi_y=0, roi_w=400, roi_h=470, start_frame_multiplier=209, end_frame_multiplier=24*60*60,
                 auto_run=0, output_csv='sf2_long_short_angle_file.csv')
    
    # Open CSV file
    with open(daq.output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['frame', 'time', 'long stick cycles', 'previous_long', 
        #                'long', 'continuous_long_angle', 'previous_short', 'short',
        #                'short positive OTT count', 'short negative OTT count', 'ott'])
        writer.writerow(['frame', 'time', 'previous_long', 
                       'long', 'continuous_long_angle', 'previous_short', 'short'])
        
        # Main processing loop
        while daq.current_frame <= daq.end_frame:
            print("###################################################################################################")
            # if daq.current_frame > 24*60*60 - 100:
            print("FRAME:", daq.current_frame)
            
            ret, image = daq.cap.read()
            if not ret:
                break
            
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
            image = image[daq.roi_y:daq.roi_y+daq.roi_h, daq.roi_x:daq.roi_x+daq.roi_w]
            copy = image.copy()
            
            # Process frame
            frame_data = daq.process_frame(image, copy)
            
               
               
            # Write to CSV only if both angle diffs are less than 30
            long_diff = frame_data['long_theta_d'] - frame_data['past_long_theta']
            short_diff = frame_data['short_theta_d'] - frame_data['past_short_theta']
            # if short_diff > 170:
            #     frame_data['short_theta_d'] = frame_data['short_theta_d'] - short_diff
            # elif short_diff < -170:
            #     frame_data['short_theta_d'] = frame_data['short_theta_d'] + short_diff
            
            #Print frame information
            # print(f"Frame: {frame_data['frame']}, " f"Time: {frame_data['time']:.4f}, " f"Left Angle: {frame_data['left_theta_d']}, "f"Right Angle: {frame_data['right_theta_d']} ",
            #       f"Long stick cycles: {frame_data['cycle_counter']}")
            
            print(f"Previous Long Angle: {frame_data['past_long_theta']}, "
                  f"Long Angle: {frame_data['long_theta_d']}, "
                  f"Long Angle Diff: {frame_data['long_theta_d'] - frame_data['past_long_theta']}, "
                  f"Bad Data: {frame_data['bad_long_flag']}, "
                  f"Rotate: {frame_data['rotate_flag']}")

            print(f"Previous Short Angle: {frame_data['past_short_theta']}, "
                  f"Short Angle: {frame_data['short_theta_d']}, "
                  f"Short Angle Diff: {frame_data['short_theta_d'] - frame_data['past_short_theta']} ",
                f"Positive OTT Count: {frame_data['short_positive_OTT_count']}, "
                  f"Negative OTT Count: {frame_data['short_negative_OTT_count']}, ")
            
            writer.writerow([
                frame_data['frame'],
                frame_data['time'],
                #frame_data['cycle_counter'],
                frame_data['past_long_theta'],
                frame_data['long_theta_d'],
                frame_data['continuous_long_angle'],
                frame_data['past_short_theta'],
                frame_data['short_theta_d']
                # frame_data['short_positive_OTT_count'],
                # frame_data['short_negative_OTT_count'],
                # frame_data['rotate_flag']
            ]) 
             
            daq.current_frame += 1
            
            #Display image
            cv2.imshow('Hough Lines Detection', frame_data['image'])

            #draw a line at through two point son copy image
            # cv2.line(copy, (195,270), (120, 470), (255, 255, 0), 8)
            
            cv2.imshow('Copy', copy)  
            #dipslay  image with the                 

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if daq.auto_run == 0: 
                cv2.waitKey(0)  # Press any key to go frame by frame
        
        cv2.destroyAllWindows()
    
    print('DONE') 

if __name__ == '__main__':
    run_acquisition()

