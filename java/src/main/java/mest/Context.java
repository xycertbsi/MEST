package mest;

public final class Context implements AutoCloseable {

    private long    handle;
    private boolean destroyed = false;

    public Context() {
        this(0);
    }

    public Context(int deviceId) {
        this.handle = NativeBridge.contextCreate(deviceId);
        if (handle == 0) {
            throw new RuntimeException("Failed to create MEST context");
        }
    }

    public void synchronize() {
        NativeBridge.contextSynchronize(handle);
    }

    @Override
    public void close() {
        if (!destroyed) {
            NativeBridge.contextDestroy(handle);
            destroyed = true;
        }
    }
}
