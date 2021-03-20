import paho.mqtt.client as mqtt
import config

"""
发布
        mqtt_tool = MqttTool()
        result = mqtt_tool.pubish('aaa')
消费
需要写脚本，暂时没做，因为时间来不及了
"""


class MqttTool(object):
    """
    mqtt的链接工具
    """
    mqtt_user = ''
    mqtt_pwd = ''
    mqtt_host = ''
    mqtt_port = ''
    mqtt_bind_address = ''
    mqtt_bind_port = ''
    mqtt_keepalive = 60
    mqtt_qos = 0
    mqtt_topic = ''
    real_mqtt = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_mqtt_config(kwargs)
        self.real_mqtt = config.get_bases_conf().get('real_mqtt', False)

    @classmethod
    def on_connect(cls, userdata, flags, rc):
        print("Connected with result code: " + str(rc))

    @classmethod
    def on_message(cls, userdata, msg):
        print(msg.topic + " " + str(msg.payload))

    def on_subscribe(cls, userdata, mid, granted_qos):
        print('xxx', userdata, mid, granted_qos)

    def init_mqtt_config(self, kwargs):
        """
        初始化mqtt配置，若入参有输入则从入参中选择，否则从配置文件中选择
        """
        if 'set_mqtt' in kwargs:
            self.set_config(kwargs)
        else:
            mqtt_config = config.get_mqtt_conf()
            if mqtt_config:
                self.set_config(mqtt_config)

    def set_config(self, mqtt_config):
        """
        设置mqtt参数
        """
        self.mqtt_user = mqtt_config['user']
        self.mqtt_pwd = mqtt_config['pwd']
        self.mqtt_host = mqtt_config['host']
        self.mqtt_port = mqtt_config['port']
        self.mqtt_bind_address = mqtt_config['bind_address']
        self.mqtt_bind_port = mqtt_config['bind_port']
        self.mqtt_keepalive = mqtt_config['keepalive']
        self.mqtt_qos = mqtt_config['qos']
        self.mqtt_topic = mqtt_config['topic']

    def pubish(self, message):
        """
        发送消息
        """
        if self.real_mqtt:
            try:
                client = self.get_mqtt_client()
                result = client.publish(self.mqtt_topic, payload=message, qos=self.mqtt_qos)
                print('publish success', result)
                result = True
            except Exception as e:
                print('publish fail', e)
                result = False
        else:
            print('not really publish')
            result = True
        return result

    def get_mqtt_client(self):
        """
        获取连接器
        """
        client = mqtt.Client()
        client.username_pw_set(self.mqtt_user, password=self.mqtt_pwd)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(self.mqtt_host, port=self.mqtt_port, keepalive=self.mqtt_keepalive,
                       bind_address=self.mqtt_bind_address, bind_port=self.mqtt_bind_port)
        return client

    def subscribe(self):
        client = self.get_mqtt_client()
        result = client.subscribe(self.mqtt_topic, qos=self.mqtt_qos)
        print('xxx', result)
        client.loop()  # 保持连接
