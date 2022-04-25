extends Node3D

var socket := StreamPeerTCP.new()
var done = false

var skeleton = get_node("Armature/Skeleton3D")
var neck_bone = skeleton.find_bone('Neck')

func _process(delta):
	#if Input.is_action_just_pressed('rotate_left'):
	#	skeleton.set_bone_pose_rotation(neck_bone, Quaternion(0, -0.1736482, 0, 0.9848078))
	pass
