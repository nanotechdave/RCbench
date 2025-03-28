from tasks.memorycapacity import memorycapacity


class Sample():
    """ Sample data class. Instance is an object containing all the measurements of a sample. """
    def __init__(self, name:str, path:str):
        self.name = name
        self.MC_vec = memorycapacity.folder_analysis_MC("E:/PoliTo/PhD/MIsure/InrimARC/NWN_Pad120M")
        print(self.MC_vec)
        return
    def __str__(self):
        objstr = f"Sample NWN_Pad{self.name}"
        return objstr