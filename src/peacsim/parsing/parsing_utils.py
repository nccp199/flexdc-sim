import configparser


def init_config_parser(source):
    config = configparser.ConfigParser()
    try:
        config.read_file(source)
    except:
        with open(source, 'r') as file:
            config.read_file(file)

    return config


def get_float_if_exists(config, section, field):
    if config.has_option(section, field):
        value = config.getfloat(section, field)
    else:
        value = None
    return value
