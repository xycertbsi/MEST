#include "mest/jni_bridge.h"
#include "mest/capi.h"
#include <jni.h>
#include <cstring>

extern "C" {

JNIEXPORT jint JNICALL Java_mest_NativeBridge_deviceCount(JNIEnv*, jclass) {
    return mest_device_count();
}

JNIEXPORT jstring JNICALL Java_mest_NativeBridge_deviceName(JNIEnv* env, jclass, jint id) {
    const char* name = mest_device_name(id);
    return env->NewStringUTF(name);
}

JNIEXPORT jlong JNICALL Java_mest_NativeBridge_deviceTotalMemory(JNIEnv*, jclass, jint id) {
    return mest_device_total_memory(id);
}

JNIEXPORT jlong JNICALL Java_mest_NativeBridge_contextCreate(JNIEnv*, jclass, jint device_id) {
    return reinterpret_cast<jlong>(mest_context_create(device_id));
}

JNIEXPORT void JNICALL Java_mest_NativeBridge_contextDestroy(JNIEnv*, jclass, jlong handle) {
    mest_context_destroy(reinterpret_cast<mest_context_t>(handle));
}

JNIEXPORT void JNICALL Java_mest_NativeBridge_contextSynchronize(JNIEnv*, jclass, jlong handle) {
    mest_context_synchronize(reinterpret_cast<mest_context_t>(handle));
}

JNIEXPORT jlong JNICALL Java_mest_NativeBridge_bufferCreateF32(JNIEnv*, jclass, jlong count) {
    return reinterpret_cast<jlong>(mest_buffer_create_f32(static_cast<size_t>(count)));
}

JNIEXPORT void JNICALL Java_mest_NativeBridge_bufferDestroy(JNIEnv*, jclass, jlong handle) {
    mest_buffer_destroy(reinterpret_cast<mest_buffer_t>(handle));
}

JNIEXPORT void JNICALL Java_mest_NativeBridge_bufferUploadF32(JNIEnv* env, jclass,
    jlong buf, jfloatArray arr, jlong count)
{
    jfloat* data = env->GetFloatArrayElements(arr, nullptr);
    mest_buffer_upload_f32(reinterpret_cast<mest_buffer_t>(buf),
                           reinterpret_cast<const float*>(data),
                           static_cast<size_t>(count));
    env->ReleaseFloatArrayElements(arr, data, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_mest_NativeBridge_bufferDownloadF32(JNIEnv* env, jclass,
    jlong buf, jfloatArray arr, jlong count)
{
    jfloat* data = env->GetFloatArrayElements(arr, nullptr);
    mest_buffer_download_f32(reinterpret_cast<mest_buffer_t>(buf),
                              reinterpret_cast<float*>(data),
                              static_cast<size_t>(count));
    env->ReleaseFloatArrayElements(arr, data, 0);
}

JNIEXPORT void JNICALL Java_mest_NativeBridge_vectorAddF32(JNIEnv*, jclass,
    jlong a, jlong b, jlong c, jlong n)
{
    mest_vector_add_f32(
        reinterpret_cast<mest_buffer_t>(a),
        reinterpret_cast<mest_buffer_t>(b),
        reinterpret_cast<mest_buffer_t>(c),
        static_cast<size_t>(n));
}

JNIEXPORT void JNICALL Java_mest_NativeBridge_vectorScaleF32(JNIEnv*, jclass,
    jlong buf, jfloat scalar, jlong n)
{
    mest_vector_scale_f32(reinterpret_cast<mest_buffer_t>(buf), scalar, static_cast<size_t>(n));
}

JNIEXPORT void JNICALL Java_mest_NativeBridge_matrixTransposeF32(JNIEnv*, jclass,
    jlong input, jlong output, jint rows, jint cols)
{
    mest_matrix_transpose_f32(
        reinterpret_cast<mest_buffer_t>(input),
        reinterpret_cast<mest_buffer_t>(output),
        rows, cols);
}

} // extern "C"
