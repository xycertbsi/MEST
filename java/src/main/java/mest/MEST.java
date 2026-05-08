package mest;

import java.util.ArrayList;
import java.util.List;

public final class MEST {

    private MEST() {}

    public static int deviceCount() {
        return NativeBridge.deviceCount();
    }

    public static String deviceName(int id) {
        return NativeBridge.deviceName(id);
    }

    public static long deviceTotalMemory(int id) {
        return NativeBridge.deviceTotalMemory(id);
    }

    public static List<String> listDevices() {
        int n = deviceCount();
        List<String> result = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            long mem = deviceTotalMemory(i);
            result.add(String.format("[%d] %s  (%.1f GB)", i, deviceName(i), mem / 1e9));
        }
        return result;
    }

    public static float[] vectorAdd(float[] a, float[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Vectors must have equal length");
        try (Buffer ba = Buffer.ofFloats(a);
             Buffer bb = Buffer.ofFloats(b);
             Buffer bc = Buffer.emptyF32(a.length)) {
            NativeBridge.vectorAddF32(ba.handle(), bb.handle(), bc.handle(), a.length);
            return bc.toFloatArray();
        }
    }

    public static float[] vectorScale(float[] data, float scalar) {
        try (Buffer buf = Buffer.ofFloats(data)) {
            NativeBridge.vectorScaleF32(buf.handle(), scalar, data.length);
            return buf.toFloatArray();
        }
    }

    public static float[][] matrixTranspose(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] flat = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            System.arraycopy(matrix[r], 0, flat, r * cols, cols);

        try (Buffer in  = Buffer.ofFloats(flat);
             Buffer out = Buffer.emptyF32(rows * cols)) {
            NativeBridge.matrixTransposeF32(in.handle(), out.handle(), rows, cols);
            float[] result = out.toFloatArray();
            float[][] transposed = new float[cols][rows];
            for (int r = 0; r < cols; r++)
                for (int c = 0; c < rows; c++)
                    transposed[r][c] = result[r * rows + c];
            return transposed;
        }
    }
}
