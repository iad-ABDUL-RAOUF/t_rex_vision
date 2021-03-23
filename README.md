# t_rex_vision
"Don’t move! It can’t see us if we don’t move." Alan Grant about a T. Rex - Jurassic Park (1993)

ROS package implementing a mouvement detector from a video imput.

The algorithme is supposed to detect moving object in a static world while the camera is also moving. It is based on optical flow detection and then a RANSAC estimation of the global mouvement model (comming from ego-motion). Pixel outliers from this model are the moving object.

Currently optical flow detection is perform with openCV Farneback. The ego-motion model is an homography transformation matrix H between consecutive frame. This suppose that the scene is approximatively planar which is a valid assumption for aerial pictures of the ground taken from a drone. One could use a fondamental matrix F instead to get a more general model. This homography H is estimated with the Direct Linear Transform (DLT) algorithm.

# subscriber
sensor_msgs/Image on topic "image_raw"

# publisher
sensor_msgs/Image on topic "moving_object"
