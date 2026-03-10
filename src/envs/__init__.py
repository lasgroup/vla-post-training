def make_env(config, num_devices: int = 4):
    if config.collect.domain == "libero":
        from src.envs.libero import make_env_libero
        return make_env_libero(config, num_devices=num_devices)
    if config.domain == "molmo":
        from src.envs.molmo import make_env_molmo

        return make_env_molmo(config, num_devices=num_devices)
    else:
        raise NotImplementedError(f"Environment domain '{config.collect.domain}' is not implemented.")
