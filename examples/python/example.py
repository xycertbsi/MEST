import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import numpy as np
import mest

print("MEST Python Example")
print("=" * 40)

devices = mest.list_devices()
for d in devices:
    print(d)

print()

a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
b = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)

result = mest.vector_add(a, b)
print(f"vector_add:   {a} + {b} = {result}")

scaled = mest.vector_scale(a.copy(), 3.0)
print(f"vector_scale: {a} * 3.0 = {scaled}")

matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
transposed = mest.matrix_transpose(matrix)
print(f"\nmatrix_transpose:\n{matrix}\n=>\n{transposed}")
