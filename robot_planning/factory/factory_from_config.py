def factory_from_config(factory_base, config_data, section_name):
    if section_name == "None":
        raise ValueError("factory_from_config: section_name is None!")
    base_type = config_data.get(section_name, "type")
    if base_type == "None":
        raise ValueError("factory_from_config: type is None!")
    instance = factory_base(base_type)
    instance.initialize_from_config(config_data, section_name)
    return instance
