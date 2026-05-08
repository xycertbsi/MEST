package mest;

public final class NativeBridge {

    static {
        System.loadLibrary("mest");
    }

    private NativeBridge() {}

    static native int    deviceCount();
    static native String deviceName(int deviceId);
    static native long   deviceTotalMemory(int deviceId);

    static native long contextCreate(int deviceId);
    static native void contextDestroy(long handle);
    static native void contextSynchronize(long handle);

    static native long bufferCreateF32(long count);
    static native void bufferDestroy(long handle);
    static native void bufferUploadF32(long buffer, float[] data, long count);
    static native void bufferDownloadF32(long buffer, float[] data, long count);

    static native void vectorAddF32(long a, long b, long c, long n);
    static native void vectorScaleF32(long buffer, float scalar, long n);
    static native void matrixTransposeF32(long input, long output, int rows, int cols);
}
