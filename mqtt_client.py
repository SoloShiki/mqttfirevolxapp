# mqtt_client.py
import paho.mqtt.client as mqtt

class MqttPublisher:
    def __init__(self, broker="localhost", port=1883, topic="edgeimpulse/detections",
                 client_id="ei-pc-01"):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id

        # Crear cliente MQTT
        self.client = mqtt.Client(client_id=self.client_id)
        try:
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            print(f"Conectado a MQTT broker {self.broker}:{self.port} con topic '{self.topic}'")
        except Exception as e:
            print(f"Error al conectar con el broker MQTT: {e}")

    def publish(self, message):
        """Publica inmediatamente el mensaje en el topic"""
        try:
            self.client.publish(self.topic, message)
            print(f"Publicado: {message}")
        except Exception as e:
            print(f"Error al publicar mensaje: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("Desconectado del broker MQTT")
