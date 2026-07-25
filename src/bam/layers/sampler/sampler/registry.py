SAMPLER_REGISTRY = {}

def register(name):
    def wrapper(cls):
        SAMPLER_REGISTRY[name] = cls
        return cls
    return wrapper