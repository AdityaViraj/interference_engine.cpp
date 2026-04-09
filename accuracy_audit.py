import numpy as np

def run_audit():
    # Load predictions from both environments
    py_preds = np.fromfile("python_truth.bin", dtype=np.float64)
    cpp_preds = np.fromfile("cpp_preds.bin", dtype=np.float64)

    if py_preds.shape != cpp_preds.shape:
        print(f"Error: Row count mismatch! Py: {len(py_preds)}, Cpp: {len(cpp_preds)}")
        return

    # Calculate Relative Error
    diff = np.abs(py_preds - cpp_preds)
    max_rel_error = np.max(diff / (np.abs(py_preds) + 1e-9))

    print(f"Max Absolute Error: {np.max(diff):.2e}")
    print(f"Max Relative Error: {max_rel_error:.2e}")

    if max_rel_error < 1e-10:
        print("STATUS: BIT-PERFECT DOMINANCE ✅")
    else:
        print("STATUS: DRIFT DETECTED ❌ Check SIMD rounding.")

if __name__ == "__main__":
    run_audit()
