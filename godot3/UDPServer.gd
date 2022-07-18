# server.gd
extends Node

var server := UDPServer.new()
var peers = []

signal new_message(message)

func _ready():
	server.listen(4240)

func _process(delta):
	server.poll() # Important!
	if server.is_connection_available():
		var peer : PacketPeerUDP = server.take_connection()
		var pkt = peer.get_packet()
		emit_signal('new_message', pkt.get_string_from_utf8())
