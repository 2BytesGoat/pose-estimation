extends Spatial

var skel
var id

var neck_rot = Vector3(0, 0, 0)

func _ready():
	skel = get_node("Armature/Skeleton")
	id = skel.find_bone("Neck")
	print("bone id:", id)
	var parent = skel.get_bone_parent(id)
	print("bone parent id:", id)
	var t = skel.get_bone_pose(id)
	print("bone transform: ", t)
	set_process(true)

func _on_UDPServer_new_message(message):
	var msg_json = JSON.parse(message).result
	var t = skel.get_bone_pose(id)
	t = t.rotated(Vector3(1.0, 0.0, 0.0), msg_json['face']['pitch'] - neck_rot.x)
	t = t.rotated(Vector3(0.0, 1.0, 0.0), msg_json['face']['yaw'] - neck_rot.y)
	t = t.rotated(Vector3(0.0, 0.0, 1.0), msg_json['face']['roll'] - neck_rot.z)
	neck_rot.x = msg_json['face']['pitch']
	neck_rot.y = msg_json['face']['yaw']
	neck_rot.z = msg_json['face']['roll']
	skel.set_bone_pose(id, t)
