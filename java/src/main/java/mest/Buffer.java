package mest;

public final class Buffer implements AutoCloseable {

    private long   handle;
    private final int count;
    private boolean destroyed = false;

    Buffer(long handle, int count) {
        this.handle = handle;
        this.count  = count;
    }

    public static Buffer ofFloats(float[] data) {
        long h = NativeBridge.bufferCreateF32(data.length);
        NativeBridge.bufferUploadF32(h, data, data.length);
        return new Buffer(h, data.length);
    }

    public static Buffer emptyF32(int count) {
        long h = NativeBridge.bufferCreateF32(count);
        return new Buffer(h, count);
    }

    public float[] toFloatArray() {
        float[] result = new float[count];
        NativeBridge.bufferDownloadF32(handle, result, count);
        return result;
    }

    public int count() { return count; }

    long handle() { return handle; }

    @Override
    public void close() {
        if (!destroyed) {
            NativeBridge.bufferDestroy(handle);
            destroyed = true;
        }
    }
}
