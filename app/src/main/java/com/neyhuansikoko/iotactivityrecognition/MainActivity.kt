package com.neyhuansikoko.iotactivityrecognition

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.MutableLiveData
import com.neyhuansikoko.iotactivityrecognition.ui.theme.BootstrapGreen
import com.neyhuansikoko.iotactivityrecognition.ui.theme.BootstrapRed
import com.neyhuansikoko.iotactivityrecognition.ui.theme.DefaultSpacings
import com.neyhuansikoko.iotactivityrecognition.ui.theme.IoTActivityRecognitionTheme
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

var isEvaluating: MutableLiveData<Boolean> = MutableLiveData<Boolean>(false)
var isRecording: MutableLiveData<Boolean> = MutableLiveData<Boolean>(false)
var isWalkingActual: MutableLiveData<Boolean> = MutableLiveData<Boolean>(false)
var isRunningActual: MutableLiveData<Boolean> = MutableLiveData<Boolean>(false)
var totalTime: MutableLiveData<Int> = MutableLiveData<Int>(0)
var walkingTime: MutableLiveData<Int> = MutableLiveData<Int>(0)
var runningTime: MutableLiveData<Int> = MutableLiveData<Int>(0)
var walkingTimeActual: MutableLiveData<Int> = MutableLiveData<Int>(0)
var runningTimeActual: MutableLiveData<Int> = MutableLiveData<Int>(0)
var precision: MutableLiveData<Float> = MutableLiveData<Float>(0F)
var recall: MutableLiveData<Float> = MutableLiveData<Float>(0F)
var f1Score: MutableLiveData<Float> = MutableLiveData<Float>(0F)

private fun linkedUnchecker(switch: MutableLiveData<Boolean>) {
    if (switch !== isWalkingActual) isWalkingActual.postValue(false)
    if (switch !== isRunningActual) isRunningActual.postValue(false)
}

private fun evaluatePrediction() {
    val evaluation = Evaluation(
        walk = walkingTimeActual.value ?: 0,
        predictedWalk = walkingTime.value ?: 0,
        run = runningTimeActual.value ?: 0,
        predictedRun = runningTime.value ?: 0
    )

    precision.postValue(evaluation.calculatePrecision())
    recall.postValue(evaluation.calculateRecall())
    f1Score.postValue(evaluation.calculateF1Score())
}

private fun reset() {
    totalTime.postValue(0)
    walkingTime.postValue(0)
    runningTime.postValue(0)
    walkingTimeActual.postValue(0)
    runningTimeActual.postValue(0)
    precision.postValue(0F)
    recall.postValue(0F)
    f1Score.postValue(0F)
    linkedUnchecker(MutableLiveData())
}

class MainActivity : ComponentActivity(), SensorEventListener {
    private var sensorManager: SensorManager? = null
    private var sensor: Sensor? = null

    private var accumulator = Array(DATA_PER_FRAME) { FloatArray(3) }
    private var accumulatorCount = 0

    private lateinit var model: ONNXModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sensor = sensorManager?.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

        model = ONNXModel(this@MainActivity)

        isRecording.observe(this) { value ->
            if (value) {
                sensorManager?.registerListener(this, sensor, SENSOR_SAMPLING_RATE)
            } else {
                sensorManager?.unregisterListener(this)
            }
        }

        setContent {
            IoTActivityRecognitionTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    App()
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (isRecording.value == true) {
            sensorManager?.registerListener(this, sensor, SENSOR_SAMPLING_RATE)
        }
    }

    override fun onDestroy() {
        if (isRecording.value == true) {
            sensorManager?.unregisterListener(this)
        }
        super.onDestroy()
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_LINEAR_ACCELERATION) {
            // 1. Accumulate data for 1 frame
            accumulator[accumulatorCount] = event.values.copyOf()
            accumulatorCount++

            if (accumulatorCount == accumulator.size) {
                runPrediction(frameData = accumulator.copyOf())
                accumulator = Array(DATA_PER_FRAME) { FloatArray(3) }
                accumulatorCount = 0
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        Log.i("MainActivity", "Sensor: $sensor, Accuracy: $accuracy")
    }

    private fun runPrediction(frameData: Array<FloatArray>) {
        runBlocking {
            launch {
                // 2. Pre-process data
                val processedData = preprocessData(frameData)
                // 3. Extract features
                val features = extractFeatures(processedData)
                // 4. Predict activity
                // 5. Modify time count
                when (model.predict(features)) {
                    1 -> walkingTime.postValue(walkingTime.value?.plus(1))
                    2 -> runningTime.postValue(runningTime.value?.plus(1))
                }
                totalTime.postValue(totalTime.value?.plus(1))
                if (isWalkingActual.value == true) walkingTimeActual.postValue(walkingTimeActual.value?.plus(1))
                if (isRunningActual.value == true) runningTimeActual.postValue(runningTimeActual.value?.plus(1))
            }
        }
    }
}

@Composable
fun App() {
    val evaluating by isEvaluating.observeAsState()
    Column(
        modifier = Modifier
            .padding(DefaultSpacings.default)
            .verticalScroll(rememberScrollState())
    ) {
        Row(
            horizontalArrangement = Arrangement.End,
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(text = "Mode Evaluasi:")
            Spacer(modifier = Modifier.size(DefaultSpacings.default))
            
            Switch(
                checked = evaluating ?: false,
                onCheckedChange = {
                    isEvaluating.postValue(it)
                    linkedUnchecker(isEvaluating)
                }
            )
            Spacer(modifier = Modifier.size(DefaultSpacings.default))
        }
        // Recording time
        val totalTime by totalTime.observeAsState()
        TimeDisplayCard(labelText = "Total waktu aktivitas:", time = displayTimeFromSeconds(totalTime ?: 0), enabled = (evaluating ?: false))
        Spacer(modifier = Modifier.size(DefaultSpacings.default))

        // Walking time
        val walking by isWalkingActual.observeAsState()
        val walkingTime by walkingTime.observeAsState()
        TimeDisplayCard(labelText = "Waktu aktivitas berjalan:", time = displayTimeFromSeconds(walkingTime ?: 0), enabled = (evaluating ?: false), checked = walking) {
            isWalkingActual.postValue(!(isWalkingActual.value ?: false))
            linkedUnchecker(isWalkingActual)
        }
        Spacer(modifier = Modifier.size(DefaultSpacings.default))

        // Running time
        val running by isRunningActual.observeAsState()
        val runningTime by runningTime.observeAsState()
        TimeDisplayCard(labelText = "Waktu aktivitas berlari:", time = displayTimeFromSeconds(runningTime ?: 0), enabled = (evaluating ?: false), checked = running) {
            isRunningActual.postValue(!(isRunningActual.value ?: false))
            linkedUnchecker(isRunningActual)
        }
        Spacer(modifier = Modifier.size(DefaultSpacings.large))

        Row(
            horizontalArrangement = Arrangement.End,
            modifier = Modifier.fillMaxWidth()
        ) {
            StartStopButton()
            Spacer(modifier = Modifier.size(DefaultSpacings.default))
            ResetButton()
        }
        Spacer(modifier = Modifier.size(DefaultSpacings.large))
        if (evaluating == true) {
            Card {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(DefaultSpacings.default)
                ) {
                    val precision by precision.observeAsState()
                    Row {
                        Text(text = "Precision:")
                        Spacer(modifier = Modifier.size(DefaultSpacings.default))
                        Text(text = "${(precision ?: 0F) * 100}%")
                    }
                    val recall by recall.observeAsState()
                    Row {
                        Text(text = "Recall:")
                        Spacer(modifier = Modifier.size(DefaultSpacings.default))
                        Text(text = "${(recall ?: 0F) * 100}%")
                    }
                    val f1Score by f1Score.observeAsState()
                    Row {
                        Text(text = "F1 Score:")
                        Spacer(modifier = Modifier.size(DefaultSpacings.default))
                        Text(text = "${(f1Score ?: 0F) * 100}%")
                    }
                }
            }
        }
    }
}

@Composable
fun TimeDisplayCard(labelText: String, time: String, enabled: Boolean, checked: Boolean? = null, onCheck: () -> Unit = {}) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = if (checked != null && enabled) {
                MaterialTheme.colorScheme.primaryContainer
            } else {
                MaterialTheme.colorScheme.surfaceVariant
            }
        ),
        modifier = Modifier.clickable(
            enabled = (checked != null && enabled)
        ) {
            onCheck()
        }
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(DefaultSpacings.default)
        ) {
            Row(
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Column {
                    Text(
                        text = labelText,
                        style = MaterialTheme.typography.titleLarge
                    )
                    Text(
                        text = time,
                        style = MaterialTheme.typography.displayMedium
                    )
                }
                checked?.let {

                    Switch(
                        enabled = enabled,
                        checked = checked,
                        onCheckedChange = {
                            onCheck()
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun StartStopButton() {
    val recordingState by isRecording.observeAsState()
    if (recordingState == false) {
        Button(
            onClick = { isRecording.postValue(true) },
            colors = ButtonDefaults.buttonColors(containerColor = BootstrapGreen)
        ) {
            Icon(Icons.Filled.PlayArrow, contentDescription = null)
            Spacer(modifier = Modifier.size(ButtonDefaults.IconSpacing))
            Text("Start")
        }
    } else {
        Button(
            colors = ButtonDefaults.buttonColors(containerColor = BootstrapRed),
            onClick = {
                isRecording.postValue(false)
                evaluatePrediction()
            }
        ) {
            Icon(Icons.Filled.Close, contentDescription = null)
            Spacer(modifier = Modifier.size(ButtonDefaults.IconSpacing))
            Text("Stop")
        }
    }
}

@Composable
fun ResetButton() {
    OutlinedButton(
        onClick = { reset() }
    ) {
        Icon(Icons.Filled.Refresh, contentDescription = "Start recording.")
        Spacer(modifier = Modifier.size(ButtonDefaults.IconSpacing))
        Text("Reset")
    }
}

@Preview(
    showBackground = true,
    showSystemUi = true,
)
@Composable
fun MainUIPreview() {
    IoTActivityRecognitionTheme {
        App()
    }
}