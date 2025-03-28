""" import numpy as np
from rcbench.tasks.memorycapacity.MemoryCapacityEvaluator import calculate_memory_capacity
from rcbench.tasks.memorycapacity.MemoryCapacityEvaluator import calculate_mc_from_file

def test_calculate_memory_capacity() -> None:
    estimated_waveform = np.ones((2,100))
    target_waveform = np.ones((2,100))
    MC, memc = calculate_memory_capacity(estimated_waveform, target_waveform)
    print(MC)
    assert MC ==2

def test_calculate_mc_from_file_linear() -> None:
    path = "tests/test_files/011_INRiMARC_NWN_Pad131M_gridSE_MemoryCapacity_2024_03_29.txt"
    MC, MC_vec = calculate_mc_from_file(path, "linear", 30, "08", "17")
    MC = np.round(MC,1)
    assert MC == 2.3

def test_calculate_mc_from_file_ridge() -> None:
    path = "tests/test_files/011_INRiMARC_NWN_Pad131M_gridSE_MemoryCapacity_2024_03_29.txt"
    MC, MC_vec = calculate_mc_from_file(path, "ridge", 30, "08", "17")
    MC = np.round(MC,1)
    assert MC == 1.9 """