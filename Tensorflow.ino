#include <Arduino.h>
#include "freertos/task.h"
#include "freertos/queue.h"
#include "soc/timer_group_struct.h"
#include "soc/timer_group_reg.h"
#include <math.h>
#include <arduinoFFT.h>  // FFTライブラリ

// TFLite 関連（TFLite を使用する場合）
#include <TensorFlowLite_ESP32.h>
#include "model.h"  // TFLite用モデルデータ（model.h）
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// KNN 用ヘッダ（Pythonで生成したヘッダーファイル）
#include "knn_model.h"

#include <Adafruit_NeoPixel.h>

//-------------------------------------------------------------
// ADC設定、サンプル数等
#define TOUCH_SAMPLE_NUM 128   // FFT用に2のべき乗に設定
#define SAMPLING_FREQ 100      // サンプリング周波数 [Hz]
#define FEATURE_NUM 20         // 10ピーク×(周波数, 振幅) = 20特徴量
#define ADC_IN 34              // ADCピン（例: 34）

//-------------------------------------------------------------
// FreeRTOS 用キュー：特徴量ベクトル（FEATURE_NUM個のfloat）を格納する
QueueHandle_t featureQueue;

//-------------------------------------------------------------
// TFLite 関連変数
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflite_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr int kTensorArenaSize = 8 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

//-------------------------------------------------------------
// NeoPixel 設定
#define NUMPIXELS 24
#define LED_OUT 27
Adafruit_NeoPixel pixels(NUMPIXELS, LED_OUT, NEO_GRB + NEO_KHZ800);

//-------------------------------------------------------------
// FFT 設定
ArduinoFFT<double> FFT = ArduinoFFT<double>();

//-------------------------------------------------------------
// 推論モード
// 0: TFLite、1: KNN
volatile int inferenceMode = 0;

//-------------------------------------------------------------
// ヘルパー関数：上位10ピークの検出
void computeFeatures(double rawData[], int N, float features[FEATURE_NUM]) {
  double real[N];
  double imag[N];
  for (int i = 0; i < N; i++) {
    real[i] = rawData[i];
    imag[i] = 0;
  }
  // ウィンドウ処理（Hamming窓）
  FFT.windowing(real, N, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  // FFT計算
  FFT.compute(real, imag, N, FFT_FORWARD);
  FFT.complexToMagnitude(real, imag, N);
  // 前半分（N/2）を利用
  int nBins = N / 2;
  // 上位10ピーク検出（DC成分は除外：i=1から）
  int topIndices[10];
  for (int i = 0; i < 10; i++) {
    topIndices[i] = -1;
  }
  for (int i = 1; i < nBins; i++) {
    double mag = real[i];
    for (int j = 0; j < 10; j++) {
      if (topIndices[j] == -1 || mag > real[topIndices[j]]) {
        for (int k = 9; k > j; k--) {
          topIndices[k] = topIndices[k - 1];
        }
        topIndices[j] = i;
        break;
      }
    }
  }
  // 各ピークについて周波数と振幅を計算し、features[] に格納
  for (int i = 0; i < 10; i++) {
    int binIndex = topIndices[i];
    float freq = binIndex * ((float)SAMPLING_FREQ / N);
    float amp = real[binIndex];
    features[i * 2] = freq;
    features[i * 2 + 1] = amp;
  }
}

//-------------------------------------------------------------
// LED 制御
void setColor(int red, int green, int blue) {
  for (int i = 0; i < NUMPIXELS; i++) {
    pixels.setPixelColor(i, pixels.Color(red, green, blue));
  }
  pixels.show();
}

//-------------------------------------------------------------
// タスク：センサデータ収集＋特徴量算出
void collectTask(void* param) {
  double rawData[TOUCH_SAMPLE_NUM];
  float features[FEATURE_NUM];
  while (true) {
    for (int i = 0; i < TOUCH_SAMPLE_NUM; i++) {
      rawData[i] = (double)analogRead(ADC_IN);
      vTaskDelay(pdMS_TO_TICKS(10));
    }
    computeFeatures(rawData, TOUCH_SAMPLE_NUM, features);
    xQueueSend(featureQueue, &features, portMAX_DELAY);
  }
}

//-------------------------------------------------------------
// タスク：推論（TFLite または KNN）＋結果表示＆LED制御
void inferenceTask(void* param) {
  float features[FEATURE_NUM];
  while (true) {
    if (xQueueReceive(featureQueue, &features, portMAX_DELAY) == pdTRUE) {
      Serial.println("Input Features:");
      for (int i = 0; i < FEATURE_NUM; i++) {
        Serial.printf("%0.4f ", features[i]);
      }
      Serial.println();

      // モードに応じた推論実行
      if (inferenceMode == 0) {
        // --- TFLite 推論 ---
        size_t num_elements = input->bytes / sizeof(float);
        if (num_elements != FEATURE_NUM) {
          Serial.println("テンソルの要素数が想定と異なります！");
          continue;
        }
        for (size_t i = 0; i < num_elements; i++) {
          input->data.f[i] = features[i];
        }
        if (interpreter->Invoke() != kTfLiteOk) {
          Serial.println("Invoke failed!");
          continue;
        }
        int num_output = output->dims->data[1];
        Serial.print("TFLite Inference Output: ");
        int predicted = 0;
        float maxProb = output->data.f[0];
        for (int i = 0; i < num_output; i++) {
          Serial.printf("%0.4f ", output->data.f[i]);
          if (output->data.f[i] > maxProb) {
            maxProb = output->data.f[i];
            predicted = i;
          }
        }
        Serial.println();
        // LED 制御（例：0: leaf, 1: non, 2: soil, 3: stalk）
        switch (predicted) {
          case 0:
            setColor(255, 0, 0);
            Serial.println("TFLite Predicted: leaf");
            break;
          case 1:
            setColor(0, 255, 0);
            Serial.println("TFLite Predicted: non");
            break;
          case 2:
            setColor(0, 0, 255);
            Serial.println("TFLite Predicted: soil");
            break;
          case 3:
            setColor(128, 0, 128);
            Serial.println("TFLite Predicted: stalk");
            break;
          default:
            setColor(0, 0, 0);
            break;
        }
      } else if (inferenceMode == 1) {
        // --- KNN 推論 ---
        int predicted = knn_predict(features);
        Serial.print("KNN Predicted: ");
        switch (predicted) {
          case 0:
            setColor(255, 0, 0);
            Serial.println("leaf");
            break;
          case 1:
            setColor(0, 255, 0);
            Serial.println("non");
            break;
          case 2:
            setColor(0, 0, 255);
            Serial.println("soil");
            break;
          case 3:
            setColor(128, 0, 128);
            Serial.println("stalk");
            break;
          default:
            setColor(0, 0, 0);
            Serial.println("Unknown");
            break;
        }
      }
    }
  }
}

//-------------------------------------------------------------
// Watchdog timer feed 関数
void feedTheDog() {
  TIMERG0.wdt_wprotect = TIMG_WDT_WKEY_VALUE;
  TIMERG0.wdt_feed = 1;
  TIMERG0.wdt_wprotect = 0;
  TIMERG1.wdt_wprotect = TIMG_WDT_WKEY_VALUE;
  TIMERG1.wdt_feed = 1;
  TIMERG1.wdt_wprotect = 0;
}

//-------------------------------------------------------------
// setup：TFLite初期化、タスク生成、LED初期化
void setup() {
  Serial.begin(921600);
  while (!Serial) {}

  Serial.println("System starting. Default mode: TFLite");

  // --- TFLite モデル初期化 ---
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  tflite_model = tflite::GetModel(e_model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("TFLite Setup complete.");

  // --- NeoPixel 初期化 ---
  pixels.begin();

  // --- キュー作成 ---
  featureQueue = xQueueCreate(10, sizeof(float[FEATURE_NUM]));
  if (featureQueue == NULL) {
    Serial.println("Queue creation failed!");
    while (1);
  }

  // --- タスク生成 ---
  xTaskCreatePinnedToCore(collectTask, "CollectTask", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(inferenceTask, "InferenceTask", 4096, NULL, 1, NULL, 1);
}

//-------------------------------------------------------------
// loop：Serial コマンドの受付とウォッチドッグフィード
void loop() {
  // Serial コマンドによる推論モード切替
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.equalsIgnoreCase("tflite")) {
      inferenceMode = 0;
      Serial.println("Switched to TFLite inference mode.");
    } else if (command.equalsIgnoreCase("knn")) {
      inferenceMode = 1;
      Serial.println("Switched to KNN inference mode.");
    }
  }
  // ウォッチドッグタイマーのフィード
  feedTheDog();
  vTaskDelay(pdMS_TO_TICKS(1));
}