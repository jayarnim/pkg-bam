SCORE_REGISTRY = {}

def register(name):
    def wrapper(cls):
        SCORE_REGISTRY[name] = cls
        return cls
    return wrapper