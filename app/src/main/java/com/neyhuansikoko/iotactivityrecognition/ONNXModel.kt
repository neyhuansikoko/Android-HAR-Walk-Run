package com.neyhuansikoko.iotactivityrecognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

class ONNXModel(private val context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession

    init {
        loadModel()
    }

    private fun loadModel() {
        val modelPath = loadAssetFromCache(context, MODEL_NAME)
        session = env.createSession(modelPath.path, OrtSession.SessionOptions())
    }

    fun predict(data: FloatArray): Int {
        val inputNames = session.inputNames.first()
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(1, data.size.toLong()))

        inputTensor.use { input ->
            val output = session.run(mutableMapOf(inputNames to input))
            output.use { out ->
                val result = (out?.get(0)?.value) as LongArray
                return result.first().toInt()
            }
        }
    }
}