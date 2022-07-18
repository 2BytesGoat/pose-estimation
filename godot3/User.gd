extends Spatial

var skel

var key_points_names = ["RightUpLeg", "RightLeg"]
var key_points = {}

func _ready():
	skel = get_node("Armature/Skeleton")
	for kp_name in key_points_names:
		var id = skel.find_bone(kp_name)
		key_points[kp_name] = {
			"id": id,
			"init_pose": skel.get_bone_pose(id)
		}
	set_process(true)

func _on_UDPServer_new_message(message):
	#print(message)
	var msg_json = JSON.parse(message).result
	for kp_name in key_points_names:
		var id = key_points[kp_name]["id"]
		var t = key_points[kp_name]["init_pose"]
		
		t = t.rotated(Vector3(1.0, 0.0, 0.0), msg_json[kp_name]['pitch'])
		t = t.rotated(Vector3(0.0, 0.0, 1.0), msg_json[kp_name]['yaw'])
		t = t.rotated(Vector3(0.0, 1.0, 0.0), msg_json[kp_name]['roll'])

		skel.set_bone_pose(id, t)
