package com.neyhuansikoko.iotactivityrecognition

import android.content.Context
import org.apache.commons.math3.complex.Complex
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation
import org.apache.commons.math3.transform.FastFourierTransformer
import org.apache.commons.math3.transform.DftNormalization
import org.apache.commons.math3.transform.TransformType
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

fun preprocessData(data: Array<FloatArray>): Array<FloatArray> {
    val magnitude = data.map { row ->
        sqrt(row.map { it * it }.sum())
    }.toFloatArray()

    val processedData = data.mapIndexed { index, row ->
        row + magnitude[index]
    }.toTypedArray()

    return processedData
}

fun FloatArray.variance(): Double {
    val mean = average()
    return sumOf { (it - mean).pow(2) } / size
}

fun FloatArray.toDoubleArray(): DoubleArray {
    return map { it.toDouble() }.toDoubleArray()
}

class FFT(data: FloatArray) {
    private val fftResults: Array<out Complex>

    init {
        val inputSize = 1 shl (32 - Integer.numberOfLeadingZeros(data.size - 1))
        val input = data.copyOf(inputSize)
        val transformer = FastFourierTransformer(DftNormalization.STANDARD)
        fftResults = transformer.transform(input.toDoubleArray(), TransformType.FORWARD)
    }

    fun calculateEnergy(): Float {
        return fftResults.sumOf { it.real * it.real }.toFloat()
    }

    fun calculateFrequencyDomainEntropy(): Float {
        val fftAbs = fftResults.map { abs(it.real) }
        val normalizedAmplitudes = fftAbs.map { sqrt(it) / fftResults.size }
        return normalizedAmplitudes.map { -it * ln(it) }.sumOf { it }.toFloat()
    }
}

fun extractFeatures(data: Array<FloatArray>): FloatArray {
    val x = FloatArray(DATA_PER_FRAME)
    val y = FloatArray(DATA_PER_FRAME)
    val z = FloatArray(DATA_PER_FRAME)
    val magnitude = FloatArray(DATA_PER_FRAME)

    data.forEachIndexed { index, row ->
        x[index] = row[0]
        y[index] = row[1]
        z[index] = row[2]
        magnitude[index] = row[3]
    }

    val meanX = x.average().toFloat()
    val meanY = y.average().toFloat()
    val meanZ = z.average().toFloat()
    val meanMagnitude = magnitude.average().toFloat()

    val varX = x.variance().toFloat()
    val varY = y.variance().toFloat()
    val varZ = z.variance().toFloat()
    val varMagnitude = magnitude.variance().toFloat()

    val corrXY = PearsonsCorrelation().correlation(x.toDoubleArray(), y.toDoubleArray()).toFloat()
    val corrYZ = PearsonsCorrelation().correlation(y.toDoubleArray(), z.toDoubleArray()).toFloat()
    val corrXZ = PearsonsCorrelation().correlation(x.toDoubleArray(), z.toDoubleArray()).toFloat()
    val corrMagX = PearsonsCorrelation().correlation(magnitude.toDoubleArray(), x.toDoubleArray()).toFloat()
    val corrMagY = PearsonsCorrelation().correlation(magnitude.toDoubleArray(), y.toDoubleArray()).toFloat()
    val corrMagZ = PearsonsCorrelation().correlation(magnitude.toDoubleArray(), z.toDoubleArray()).toFloat()

    val fftX = FFT(x)
    val fftY = FFT(y)
    val fftZ = FFT(z)
    val fftMagnitude = FFT(magnitude)

    val fftEnergyX = fftX.calculateEnergy()
    val fftEnergyY = fftY.calculateEnergy()
    val fftEnergyZ = fftZ.calculateEnergy()
    val fftEnergyMagnitude = fftMagnitude.calculateEnergy()

    val fftEntropyX = fftX.calculateFrequencyDomainEntropy()
    val fftEntropyY= fftY.calculateFrequencyDomainEntropy()
    val fftEntropyZ = fftZ.calculateFrequencyDomainEntropy()
    val fftEntropyMagnitude = fftMagnitude.calculateFrequencyDomainEntropy()

    return floatArrayOf(
        meanX, meanY, meanZ, meanMagnitude,
        varX, varY, varZ, varMagnitude,
        corrXY, corrYZ, corrXZ, corrMagX, corrMagY, corrMagZ,
        fftEnergyX, fftEnergyY, fftEnergyZ, fftEnergyMagnitude,
        fftEntropyX, fftEntropyY, fftEntropyZ, fftEntropyMagnitude
    )
}

class Evaluation(walk: Int, predictedWalk: Int, run: Int, predictedRun: Int) {
    private val walk = walk.toFloat()
    private val predictedWalk = predictedWalk.toFloat()
    private val run = run.toFloat()
    private val predictedRun = predictedRun.toFloat()

    private fun tp(): Float {
        return min(walk, predictedWalk) + min(run, predictedRun)
    }

    private fun fp(): Float {
        return max(0f, predictedWalk - min(walk, predictedWalk)) + max(0f, predictedRun - min(run, predictedRun))
    }

    private fun fn(): Float {
        return max(0f, walk - min(walk, predictedWalk)) + max(0f, run - min(run, predictedRun))
    }

    fun calculatePrecision(): Float {
        val tp = tp()
        val fp = fp()
        return (tp / (tp + fp)).takeUnless { it.isNaN() } ?: 0F
    }

    fun calculateRecall(): Float {
        val tp = tp()
        val fn = fn()
        return (tp / (tp + fn)).takeUnless { it.isNaN() } ?: 0F
    }

    fun calculateF1Score(): Float {
        val precision = calculatePrecision()
        val recall = calculateRecall()
        return ((2 * precision * recall) / (precision + recall)).takeUnless { it.isNaN() } ?: 0F
    }
}

fun displayTimeFromSeconds(seconds: Int): String {
    val hours = seconds / 3600
    val minutes = (seconds % 3600) / 60
    val remainingSeconds = seconds % 60

    return String.format("%02d:%02d:%02d", hours, minutes, remainingSeconds)
}

fun loadAssetFromCache(context: Context, fileName: String): File {
    val cacheDir = context.cacheDir
    val file = File(cacheDir, fileName)

    if (!file.exists()) {
        context.assets.open(fileName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
    }

    return file
}