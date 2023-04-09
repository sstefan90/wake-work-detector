package com.example.wakeapp;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.DialogInterface;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.Manifest;
import android.content.pm.PackageManager;

import java.io.IOException;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.io.*;

import com.jlibrosa.audio.JLibrosa;


import com.jlibrosa.audio.exception.FileFormatNotSupportedException;
import com.jlibrosa.audio.wavFile.WavFile;
import com.jlibrosa.audio.wavFile.WavFileException;

import java.lang.*;


public class MainActivity extends AppCompatActivity implements Runnable {

    private Module module = null;
    private final static String TAG = MainActivity.class.getSimpleName();

    JLibrosa audio_processor = new JLibrosa();
    ScalerClass scaler = new ScalerClass();
    private final static int REQUEST_RECORD_AUDIO = 13;
    private final static float AUDIO_LEN_IN_SECONDS = 1.7f;
    private final static int SAMPLE_RATE = 44100;
    private final static float RECORDING_LENGTH_FLOAT = SAMPLE_RATE * AUDIO_LEN_IN_SECONDS;
    private final static int RECORDING_LENGTH = (int) RECORDING_LENGTH_FLOAT; //should be 74970
    private int mStart = 1;
    private HandlerThread mListeningThread;
    private Button mRecordBtn;
    private TextView mWakewordTv;
    private TextView mListeningTv;
    private Handler mListeningHandler;
    private Runnable mListeningRunnable = new Runnable() {
        @Override
        public void run() {
            mListeningHandler.postDelayed(mListeningRunnable, 1000);
            MainActivity.this.runOnUiThread(
                    () -> {
                        mListeningTv.setText(String.format("Listening - %fs left", AUDIO_LEN_IN_SECONDS - mStart));
                        mStart += 1;
                    });
        }
    };


    private void requestAudioPermissions() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.RECORD_AUDIO)) {
            AlertDialog dialog = new AlertDialog.Builder(MainActivity.this).create();
            dialog.setTitle("Permission needed");
            dialog.setMessage("This permission is needed to record audio to process wakeword");
            dialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                    new DialogInterface.OnClickListener() {
                        public void onClick(DialogInterface dialog, int which) {
                            dialog.dismiss();
                        }
                    });
            dialog.show();

        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);

        }
    }

    protected void stopListeningThread() {
        mListeningThread.quitSafely();
        try {
            mListeningThread.join();
            mListeningThread = null;
            mListeningHandler = null;
            mStart = 1;
        } catch (InterruptedException e) {
            Log.e(TAG, "Error on stopping background thread", e);
        }
    }


    public void run() {

        if (!(ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)) {
            requestAudioPermissions();
        }

        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "Audio Record can't initialize!");
            return;
        }
        record.startRecording();

        long shortsRead = 0;
        int recordingOffset = 0;
        short[] audioBuffer = new short[bufferSize / 2];
        short[] recordingBuffer = new short[RECORDING_LENGTH * 2];

        while (shortsRead < RECORDING_LENGTH) {
            int numberOfShort = record.read(audioBuffer, 0, audioBuffer.length);
            shortsRead += numberOfShort;
            System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, numberOfShort);
            recordingOffset += numberOfShort;
        }

        record.stop();
        record.release();
        stopListeningThread();

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mRecordBtn.setText("Recognizing...");
            }
        });

        float[] toprint = new float[10];
        for (int i = 0; i < toprint.length; i++) {
            toprint[i] = recordingBuffer[i];
        }
        Log.d(TAG, "pre raw raw audio " + Arrays.toString(toprint));

        float[] floatInputBuffer = new float[RECORDING_LENGTH];

        // feed in float values between -1.0f and 1.0f by dividing the signed 16-bit inputs.
        for (int i = 0; i < RECORDING_LENGTH; ++i) {
            floatInputBuffer[i] = recordingBuffer[i] / (float) Short.MAX_VALUE;
        }
        float[] to_print = new float[10];
        for (int i = 0; i < to_print.length; i++) {
            to_print[i] = floatInputBuffer[i];
        }

        Log.d(TAG, "raw audio signal" + Arrays.toString(to_print));

        String path = "";

        try {
            path = assetFilePath("test3.wav");
        } catch (Exception e) {
            Log.d(TAG, "could not get wav file path", e);
        }
        Log.d(TAG, "wav file path " + path);

        float[] soundBuffer = readSoundFile(path, 44100, 1.7, 0);

        for (int i = 0; i < to_print.length; i++) {
            to_print[i] = soundBuffer[i];
        }

        Log.d(TAG, "input_audio_file " + Arrays.toString(to_print));


        final int result = detect(soundBuffer);


        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                String message;
                if (result == 1) {
                    message = "Smart Lamp On!";
                } else {
                    message = "Wake Word Not Detected";
                }
                showDetectionResult(message);
                mRecordBtn.setEnabled(true);
                mRecordBtn.setText("Record");
            }
        });
    }

    private void showDetectionResult(String result) {
        mWakewordTv.setText(result);
    }

    private Integer detect(float[] floatInputBuffer) {
        if (module == null) {
            try {
                module = LiteModuleLoader.load(assetFilePath("audiomodel_mobile_lite.ptl"));
            } catch (IOException e) {
                Log.e(TAG, "Unable to load model", e);
            }
        }

        //double audio_model_input[] = new double[RECORDING_LENGTH];
        //for (int n = 0; n < RECORDING_LENGTH; n++)
        //audio_model_input[n] = floatInputBuffer[n];

        //FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(RECORDING_LENGTH);
        //for (double val : audio_model_input)
        //inTensorBuffer.put((float) val);

        float[][] mel_spec = getMelSpectrogram(floatInputBuffer, audio_processor);
        float[] mel_spec_flat = flattenArray(mel_spec);
        float[] mel_spec_db = power_to_db(mel_spec_flat);
        float[] mel_spec_db_scaled = scale_signal(mel_spec_db, scaler);

        FloatBuffer inTensorBufferFinal = Tensor.allocateFloatBuffer(mel_spec_db_scaled.length);
        for (float val : mel_spec_db_scaled)
            inTensorBufferFinal.put((float) val);

        Tensor inTensor = Tensor.fromBlob(inTensorBufferFinal, new long[]{1, 1, 128, 147});
        float[] result = module.forward(IValue.from(inTensor)).toTensor().getDataAsFloatArray();
        Log.d(TAG, String.format("model result %f, %f", result[0], result[1]));

        int largest = 0;
        for (int i = 0; i < result.length; i++) {
            if (result[i] > result[largest]) largest = i;
        }
        return largest;

    }

    public float[][] getMelSpectrogram(float[] inTensorx, JLibrosa audio_processor) {

        float[][] mel_spec = audio_processor.generateMelSpectroGram(inTensorx, 44100, 2048, 128, 512);
        Log.e(TAG, String.format("shape melspec row %d col %d,", mel_spec.length, mel_spec[0].length));
        double[] to_print = new double[10];
        for (int i = 0; i < to_print.length; i++) {
            to_print[i] = mel_spec[0][i];
        }

        Log.e(TAG, String.format("first row mel_spec: " + Arrays.toString(to_print)));
        return mel_spec;

    }

    public float[] scale_signal(float[] flatArray, ScalerClass scaler) {
        for (int i = 0; i < flatArray.length; i++) {
            int index = (i % scaler.scale.length);
            flatArray[i] = (flatArray[i] - scaler.mean[index]) / scaler.scale[index];
        }
        Log.d(TAG, String.format("scaled signal %f, %f, %f", flatArray[0], flatArray[1], flatArray[100]));
        double[] to_print = new double[10];
        for (int i = 0; i < to_print.length; i++) {
            to_print[i] = flatArray[i];
        }

        Log.e(TAG, String.format("first row scale_signal: " + Arrays.toString(to_print)));
        return flatArray;

    }

    public float[] power_to_db(float[] flatArray) {
        float max_ = Float.NEGATIVE_INFINITY;
        float min_ = 1e-10f;
        float top_db = 80;

        //get max of the mel spectogram
        for (int i = 0; i < flatArray.length; i++) {
            if (flatArray[i] > max_) max_ = flatArray[i];
        }

        if (max_ < min_) {
            max_ = min_;
        }
        float reduce = (float) ((float) 10.0 * Math.log10(max_));
        float log_max = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < flatArray.length; i++) {
            if (flatArray[i] < min_) {
                flatArray[i] = (float) ((float) 10.0 * Math.log10(min_) - reduce);
            } else {
                flatArray[i] = (float) (10.0 * Math.log10(flatArray[i]) - reduce);
            }

            if (flatArray[i] > log_max) {
                log_max = flatArray[i];
            }
        }
        Log.e(TAG, String.format("my log_max in power_to_db: %f", log_max));


        for (int i = 0; i < flatArray.length; i++) {
            if (flatArray[i] < log_max - top_db) {
                flatArray[i] = log_max - top_db;
            }
        }

        double[] to_print = new double[10];
        for (int i = 0; i < to_print.length; i++) {
            to_print[i] = flatArray[i];
        }

        Log.e(TAG, String.format("first row power_to_db: " + Arrays.toString(to_print)));
        return flatArray;
    }

    public float[] flattenArray(float[][] inTensor) {
        int length = inTensor.length * inTensor[0].length;
        float[] flattenedArray = new float[length];
        Log.e(TAG, String.format("shape of mel spec rows: %d, cols: %d", inTensor.length, inTensor[0].length));

        for (int row = 0; row < inTensor.length; row++) {
            for (int col = 0; col < inTensor[0].length; col++) {
                int index = row * inTensor.length + col;
                flattenedArray[index] = inTensor[row][col];
            }
        }
        return flattenedArray;
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mRecordBtn = findViewById(R.id.btnRecord);
        mWakewordTv = findViewById(R.id.tvWakeword);
        mListeningTv = findViewById(R.id.tvInstruction);

        mRecordBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mListeningTv.setText(String.format("Listening - %fs left", AUDIO_LEN_IN_SECONDS));
                mWakewordTv.setText("");
                mRecordBtn.setEnabled(false);
                Thread thread = new Thread(MainActivity.this);
                thread.start();
                mListeningThread = new HandlerThread("Timer");
                mListeningThread.start();
                mListeningHandler = new Handler(mListeningThread.getLooper());
                mListeningHandler.postDelayed(mListeningRunnable, 1000);

            }
        });
    }

    public String assetFilePath(String assetName) throws IOException {
        File file = new File(this.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        try (InputStream is = this.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public float[] readSoundFile(String path, int sampleRate, double readDurationInSeconds, int offsetDuration) {


        File sourceFile = new File(path);
        WavFile wavFile = null;
        try {
            wavFile = WavFile.openWavFile(sourceFile);
        } catch (Exception e) {
            Log.d(TAG, "no", e);
        }
        int mNumFrames = (int) (wavFile.getNumFrames());
        int mSampleRate = (int) wavFile.getSampleRate();
        int mChannels = wavFile.getNumChannels();

        int totalNoOfFrames = mNumFrames;
        int frameOffset = offsetDuration * mSampleRate;
        double tobeReadFramesFloat = readDurationInSeconds * mSampleRate;
        int tobeReadFrames = (int) tobeReadFramesFloat;

        if (tobeReadFrames > (totalNoOfFrames - frameOffset)) {
            tobeReadFrames = totalNoOfFrames - frameOffset;
        }

        if (readDurationInSeconds != -1) {
            mNumFrames = tobeReadFrames;
            wavFile.setNumFrames(mNumFrames);
        }


        if (sampleRate != -1) {
            mSampleRate = sampleRate;
        }

        // Read the magnitude values across both the channels and save them as part of
        // multi-dimensional array

        float[][] buffer = new float[mChannels][mNumFrames];
        long readFrameCount = 0;
        //for (int i = 0; i < loopCounter; i++) {
        try {
            readFrameCount = wavFile.readFrames(buffer, mNumFrames, frameOffset);
        } catch (Exception e) {
            Log.d(TAG, "could not read file frame", e);
        }
        //}

        if (wavFile != null) {
            try {
                wavFile.close();
            } catch (Exception e) {
                Log.d(TAG, "could not close wav file", e);
            }
        }
        float[] mono_buffer = buffer[0];

        return mono_buffer;

    }
}