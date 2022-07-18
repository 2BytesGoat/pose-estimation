extends Spatial

export(NodePath) var neck_bone_path
onready var neck_bone = get_node(neck_bone_path)

func _process(delta):
	if neck_bone:
		neck_bone.rotate(Vector3(1, 0, 0), 1.5 * delta)
		print(neck_bone.rotation_degrees)

func _on_UDPServer_new_message(message):
	var msg_json = JSON.parse(message).result
	print(msg_json)
	neck_bone.set_rotation_degrees(Vector3(
		msg_json['face']['pitch'],
		msg_json['face']['yaw'],
		msg_json['face']['roll']
		)
	)
