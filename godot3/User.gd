extends Spatial

var skel

var neck_id
var left_arm_id
var right_arm_id

var neck_t
var left_arm_t
var right_arm_t

func _ready():
	skel = get_node("Armature/Skeleton")
	neck_id = skel.find_bone("Neck")
	left_arm_id = skel.find_bone("LeftShoulder")
	right_arm_id = skel.find_bone("RightShoulder")
	
	set_process(true)
	neck_t = skel.get_bone_pose(neck_id)
	left_arm_t = skel.get_bone_pose(left_arm_id)
	right_arm_t = skel.get_bone_pose(right_arm_id)

func _on_UDPServer_new_message(message):
	var msg_json = JSON.parse(message).result
	
	var neck_t_cpy = neck_t
	neck_t_cpy = neck_t_cpy.rotated(Vector3(1.0, 0.0, 0.0), msg_json['face']['pitch'])
	neck_t_cpy = neck_t_cpy.rotated(Vector3(0.0, 1.0, 0.0), msg_json['face']['yaw'])
	neck_t_cpy = neck_t_cpy.rotated(Vector3(0.0, 0.0, 1.0), msg_json['face']['roll'])
	skel.set_bone_pose(neck_id, neck_t_cpy)
	
	var larm_t_cpy = left_arm_t
	larm_t_cpy = larm_t_cpy.rotated(Vector3(1.0, 0.0, 0.0), msg_json['arm']['shoulder_left'])
	skel.set_bone_pose(left_arm_id, larm_t_cpy)
	
	var rarm_t_cpy = right_arm_t
	rarm_t_cpy = rarm_t_cpy.rotated(Vector3(1.0, 0.0, 0.0), msg_json['arm']['shoulder_right'])
	skel.set_bone_pose(right_arm_id, rarm_t_cpy)
	
