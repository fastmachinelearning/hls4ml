def set_default_config(hls_conf, default_config):
    for key, value in default_config.items():
        if key not in hls_conf.keys():
            hls_conf[key] = value
    return hls_conf
