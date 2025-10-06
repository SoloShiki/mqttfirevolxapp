# mqtt_client.py
"""
Robust MQTT publisher wrapper using paho-mqtt.
"""

import time
from typing import Optional

import paho.mqtt.client as mqtt


class MqttPublisher:
    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic: str = "cigar/detect",
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tls: Optional[dict] = None,
        keepalive: int = 60,
        connect_timeout: int = 10,
    ):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id or f"ei-{int(time.time())}"

        self.client = mqtt.Client(client_id=self.client_id)
        if username:
            self.client.username_pw_set(username, password)

        if tls:
            try:
                self.client.tls_set(**tls)
            except Exception as e:
                print("TLS setup failed:", e)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self.connected = False
        self._connect(connect_timeout)

    def _connect(self, connect_timeout):
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            timeout = time.time() + connect_timeout
            while time.time() < timeout and not self.connected:
                time.sleep(0.05)
        except Exception as e:
            print("MQTT connection error:", e)

    def _on_connect(self, client, userdata, flags, rc):
        self.connected = (rc == 0)
        print("MQTT connected" if self.connected else f"MQTT connect failed (rc={rc})")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print("MQTT disconnected, rc=", rc)

    def publish(self, payload: str, topic: Optional[str] = None, qos: int = 0, retain: bool = False) -> bool:
        topic = topic or self.topic
        if not self.connected:
            print("Not connected to broker.")
            return False
        try:
            info = self.client.publish(topic, payload, qos=qos, retain=retain)
            info.wait_for_publish()
            return info.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print("MQTT publish error:", e)
            return False

    def disconnect(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
