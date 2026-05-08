#pragma once
#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint    JNICALL Java_mest_NativeBridge_deviceCount(JNIEnv*, jclass);
JNIEXPORT jstring JNICALL Java_mest_NativeBridge_deviceName(JNIEnv*, jclass, jint);
JNIEXPORT jlong   JNICALL Java_mest_NativeBridge_deviceTotalMemory(JNIEnv*, jclass, jint);

JNIEXPORT jlong   JNICALL Java_mest_NativeBridge_contextCreate(JNIEnv*, jclass, jint);
JNIEXPORT void    JNICALL Java_mest_NativeBridge_contextDestroy(JNIEnv*, jclass, jlong);
JNIEXPORT void    JNICALL Java_mest_NativeBridge_contextSynchronize(JNIEnv*, jclass, jlong);

JNIEXPORT jlong   JNICALL Java_mest_NativeBridge_bufferCreateF32(JNIEnv*, jclass, jlong);
JNIEXPORT void    JNICALL Java_mest_NativeBridge_bufferDestroy(JNIEnv*, jclass, jlong);
JNIEXPORT void    JNICALL Java_mest_NativeBridge_bufferUploadF32(JNIEnv*, jclass, jlong, jfloatArray, jlong);
JNIEXPORT void    JNICALL Java_mest_NativeBridge_bufferDownloadF32(JNIEnv*, jclass, jlong, jfloatArray, jlong);

JNIEXPORT void    JNICALL Java_mest_NativeBridge_vectorAddF32(JNIEnv*, jclass, jlong, jlong, jlong, jlong);
JNIEXPORT void    JNICALL Java_mest_NativeBridge_vectorScaleF32(JNIEnv*, jclass, jlong, jfloat, jlong);
JNIEXPORT void    JNICALL Java_mest_NativeBridge_matrixTransposeF32(JNIEnv*, jclass, jlong, jlong, jint, jint);

#ifdef __cplusplus
}
#endif
