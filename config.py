import os
import yaml
import platform


def get_root_path():
    return os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def get_config_path(file_name="config.yaml"):
    # root_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    file_path = replace_path_based_system(os.path.join(get_root_path(), file_name))
    return file_path


def get_mqtt_conf():
    return get_config("mqtt")


def get_bases_conf():
    return get_config("base")


def get_mysql_config():
    return get_config("mysql")


def get_mongo_config():
    return get_config("mongo")


def get_rabbit_config():
    return get_config("rabbit")


def get_email_config():
    return get_config("email")


def get_redis_config():
    return get_config("redis")


def replace_path_based_system(path):
    sys_type = platform.system()
    if sys_type == "Windows":
        path = path.replace("/", os.path.sep)
    return path


def get_config(key="", file_name="config.yaml"):
    """
    获取配置文件，若配置文件目录下存在 local文件，则会用local文件的内容覆盖config文件
    :param key:
    :param file_name:
    :return:
    """
    file_path = get_config_path(file_name)
    file_path_list = file_path.split("/")
    if len(file_path_list) > 1:
        local_file_name = "/" . join(file_path_list[:-1]) + '/local.' + file_path_list[-1]
    else:
        local_file_name = 'local.' + str(file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    config_dict = yaml.load(content)
    if os.path.exists(local_file_name):
        with open(local_file_name, "r", encoding="utf-8") as f:
            local_content = f.read()
        local_config_dict = yaml.load(local_content)
        config_dict.update(local_config_dict)
    if key:
        config_dict = config_dict.get(key, {})
    return config_dict

