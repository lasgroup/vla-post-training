def make_env(config, num_devices: int = 4):
    if config.domain == "libero":
        from src.envs.libero import make_env_libero
        return make_env_libero(config, num_devices=num_devices)
    else:
        raise NotImplementedError(f"Environment domain '{config.domain}' is not implemented.")
