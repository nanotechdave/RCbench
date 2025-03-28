from tasks.memorycapacity import memorycapacity


class Measurement():
    def __init__(self):
        return
    

class MemoryCapacity(Measurement):
    def __init__(self, path:str):
        data = memorycapacity.read_and_parse_to_df(path)
    
