import socket

class GodotUDPClient:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    def send_message(self, angles):
        message = {
            "face": {
                    "pitch": angles[0],
                    "yaw": angles[1],
                    "roll": angles[2]
                }
            }
        message_str = str(message).replace("'", '"')
        self.sock.sendto(message_str.encode('utf-8'), ("127.0.0.1", 4242))

if __name__ == '__main__':
    client = GodotUDPClient()
    client.send_message([0.75, 0, 0])