extends Node

var view_idx = 0
var view_pos = [
	{
		"translation": Vector3(0, 1, 2),
		"rotation": Vector3.ZERO
	},
	{
		"translation": Vector3(-2, 1, 0),
		"rotation": Vector3(0, -90, 0)
	}
]
onready var camera = get_node("Camera")

func _input(event):
	if Input.is_action_pressed("ui_accept"):
		view_idx = (view_idx + 1) % len(view_pos)
		camera.translation = view_pos[view_idx]["translation"]
		camera.rotation_degrees = view_pos[view_idx]["rotation"]
