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
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.MutableLiveData
import com.neyhuansikoko.iotactivityrecognition.ui.theme.BootstrapGreen
import com.neyhuansikoko.iotactivityrecognition.ui.theme.BootstrapRed
import com.neyhuansikoko.iotactivityrecognition.ui.theme.DefaultSpacings
import com.neyhuansikoko.iotactivityrecognition.ui.theme.IoTActivityRecognitionTheme
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

var isRecording: MutableLiveData<Boolean> = MutableLiveData<Boolean>(false)
var totalTime: MutableLiveData<Int> = MutableLiveData<Int>(0)
var walkingTime: MutableLiveData<Int> = MutableLiveData<Int>(0)
var runningTime: MutableLiveData<Int> = MutableLiveData<Int>(0)

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
            val job = launch {
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
            }
        }
    }
}

@Composable
fun App() {
    Column(
        modifier = Modifier
            .padding(DefaultSpacings.default)
            .verticalScroll(rememberScrollState())
    ) {
        // Recording time
        val totalTime by totalTime.observeAsState()
        TimeDisplayCard(labelText = "Total waktu aktivitas:", time = displayTimeFromSeconds(totalTime ?: 0))
        Spacer(modifier = Modifier.size(DefaultSpacings.default))

        // Walking time
        val walkingTime by walkingTime.observeAsState()
        TimeDisplayCard(labelText = "Waktu aktivitas berjalan:", time = displayTimeFromSeconds(walkingTime ?: 0))
        Spacer(modifier = Modifier.size(DefaultSpacings.default))

        // Running time
        val runningTime by runningTime.observeAsState()
        TimeDisplayCard(labelText = "Waktu aktivitas berlari:", time = displayTimeFromSeconds(runningTime ?: 0))
        Spacer(modifier = Modifier.size(DefaultSpacings.large))

        Row(
            horizontalArrangement = Arrangement.End,
            modifier = Modifier.fillMaxWidth()
        ) {
            StartStopButton()
            Spacer(modifier = Modifier.size(DefaultSpacings.default))
            ResetButton()
        }
    }
}

@Composable
fun TimeDisplayCard(labelText: String, time: String) {
    Card {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(DefaultSpacings.default)
        ) {
            Text(
                text = labelText,
                style = MaterialTheme.typography.titleLarge
            )
            Text(
                text = time,
                style = MaterialTheme.typography.displayMedium
            )
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
            onClick = { isRecording.postValue(false) },
            colors = ButtonDefaults.buttonColors(containerColor = BootstrapRed)
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
        onClick = {
            totalTime.postValue(0)
            walkingTime.postValue(0)
            runningTime.postValue(0)
        }
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