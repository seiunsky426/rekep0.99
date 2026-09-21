uint8 REGISTER=1
uint8 REMOVE=2
uint8 RESET=3

std_msgs/Header header
uint8 action
string object_uuid
string display_name
uint32 rigid_group_id
string source_camera
bool excluded_from_static_map
sensor_msgs/Image seed_mask
sensor_msgs/RegionOfInterest roi
geometry_msgs/Point32[] positive_pixels
geometry_msgs/Point32[] negative_pixels
