import pickle
from vehicle_pipeline import Vehicle_pipeline
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from utils import read_image, draw_boxes, rgb, write_image, convert_video_frame, make_heatmap

if __name__ == '__main__':
    # input_file = 'project_video.mp4'
    # output_file = 'out.mp4'
    input_file = 'test_images/test4.jpg'
    output_file = 'test_images_processed/test4.jpg'

    heat_threshold = 2.25
    # load model
    print('Loading classifier from pickle classifier.p')
    with open('classifier.p', 'rb') as f:
        data = pickle.load(f)
        classifier = data['classifier']
        feature_parameters = data['feature_parameters']
        window_shape = data['shape']
        scaler = data['scaler']

        print('Feature parameters:')
        print(feature_parameters)
    # extract file extension
    file_extension = input_file.split('.')[-1].lower()

    if file_extension in ['jpg', 'png']:
        # process image
        # Instantiate detector
        vehicle_detector = Vehicle_pipeline(classifier, feature_parameters, window_shape, scaler, heat_threshold)

        print('Loading ' + input_file + ' as a ' + feature_parameters['cspace'] + ' image')
        img = read_image(input_file, feature_parameters['cspace'])
        output_to_file = output_file and len(output_file)

        print('Detecting vehicles')
        boxes = vehicle_detector(img, show_plots=False)
        print(boxes)
        output = draw_boxes(rgb(img, feature_parameters['cspace']), boxes)

        if output_to_file:
            print('Writing output to ' + output_file)
            write_image(output_file, output, 'RGB')
        else:
            plt.figure()
            plt.title(input_file)
            plt.imshow(output)
            plt.show()
    elif file_extension in ['mp4']:
        # process video
            vehicle_detector = Vehicle_pipeline(classifier, feature_parameters, window_shape, scaler, heat_threshold, alpha=0.125)

            def frame_handler(frame):
                boxes = vehicle_detector(convert_video_frame(frame, feature_parameters['cspace']))
                output = draw_boxes(frame, boxes)
                return output

            clip = VideoFileClip(input_file)

            clip = clip.fl_image(frame_handler)

            print("Writing video file to {}".format(output_file))
            clip.write_videofile(output_file, audio=False)
