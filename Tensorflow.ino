#include <Arduino.h>
#include "freertos/task.h"
#include "freertos/queue.h"
#include "soc/timer_group_struct.h"
#include "soc/timer_group_reg.h"
#include <math.h>
#include <arduinoFFT.h>  // FFTライブラリのインクルード

#include <TensorFlowLite_ESP32.h>
#include "model.h"  // ここでは model.h に定義したモデルデータを使用
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <Adafruit_NeoPixel.h>

//-------------------------------------------------------------
// ADC設定、サンプル数等
#define TOUCH_SAMPLE_NUM 128   // FFT用に2のべき乗に設定
#define SAMPLING_FREQ 100      // サンプリング周波数 [Hz]（サンプル間隔10msの場合）
#define FEATURE_NUM 20         // 10ピーク×(周波数, 振幅) = 20特徴量
#define ADC_IN 34              // ADCピン（例: 34）

//-------------------------------------------------------------
// FreeRTOS用キュー：特徴量ベクトル（FEATURE_NUM個のfloat）を格納する
QueueHandle_t featureQueue;

//-------------------------------------------------------------
// TFLite関連
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr int kTensorArenaSize = 8 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

//-------------------------------------------------------------
// NeoPixel設定
#define NUMPIXELS 24
#define LED_OUT 27
Adafruit_NeoPixel pixels(NUMPIXELS, LED_OUT, NEO_GRB + NEO_KHZ800);

//-------------------------------------------------------------
// FFT設定
ArduinoFFT<double> FFT = ArduinoFFT<double>();

//-------------------------------------------------------------
// ヘルパー関数：上位10ピークの検出
// FFTを実施し、得られた振幅スペクトルから上位10ピークを検出し、
// 各ピークに対して「周波数（Hz）」と「振幅」を算出してfeatures[]に格納します。
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

  // FFT出力は左右対称なので、前半分（N/2）を利用
  int nBins = N / 2;

  // 簡易的な上位10ピーク検出（DC成分は除外するため、i=1から）
  int topIndices[10];
  for (int i = 0; i < 10; i++) {
    topIndices[i] = -1;
  }
  for (int i = 1; i < nBins; i++) {
    double mag = real[i];
    // 上位10内に入るかチェック
    for (int j = 0; j < 10; j++) {
      if (topIndices[j] == -1 || mag > real[topIndices[j]]) {
        // 後ろをシフトして挿入
        for (int k = 9; k > j; k--) {
          topIndices[k] = topIndices[k - 1];
        }
        topIndices[j] = i;
        break;
      }
    }
  }

  // 検出した各ピークについて、周波数と振幅を算出してfeatures[]に格納
  // 配列の順番は [Peak1_Freq, Peak1_Amp, Peak2_Freq, Peak2_Amp, ...]
  for (int i = 0; i < 10; i++) {
    int binIndex = topIndices[i];
    float freq = binIndex * ((float)SAMPLING_FREQ / N);  // 周波数 = bin_index * (Fs / N)
    float amp = real[binIndex];
    features[i * 2] = freq;
    features[i * 2 + 1] = amp;
  }
}

//-------------------------------------------------------------
// LEDの色を条件に応じて変更する関数
void setColor(int red, int green, int blue) {
  for (int i = 0; i < NUMPIXELS; i++) {
    pixels.setPixelColor(i, pixels.Color(red, green, blue));
  }
  pixels.show();
}

//-------------------------------------------------------------
// タスク：センサデータ収集＋特徴量算出
// ADCからTOUCH_SAMPLE_NUM個のサンプルを取得し、FFTでピーク検出を行い、
// 20個の特徴量を作成してキューに送信します。
void collectTask(void* param) {
  double rawData[TOUCH_SAMPLE_NUM];
  float features[FEATURE_NUM];
  while (true) {
    // ADCからサンプル取得（例：10ms間隔）
    for (int i = 0; i < TOUCH_SAMPLE_NUM; i++) {
      rawData[i] = (double)analogRead(ADC_IN);
      vTaskDelay(pdMS_TO_TICKS(10));
    }
    // FFTを用いて上位10ピークから特徴量を算出
    computeFeatures(rawData, TOUCH_SAMPLE_NUM, features);
    // キューに特徴量ベクトルを送信
    xQueueSend(featureQueue, &features, portMAX_DELAY);
  }
}

//-------------------------------------------------------------
// タスク：TFLiteモデル推論＋結果表示＆LED制御
void inferenceTask(void* param) {
  float features[FEATURE_NUM];
  while (true) {
    // キューから特徴量ベクトルを受信
    if (xQueueReceive(featureQueue, &features, portMAX_DELAY) == pdTRUE) {
      Serial.println("Input Features:");
      for (int i = 0; i < FEATURE_NUM; i++) {
        Serial.printf("%0.4f ", features[i]);
      }
      Serial.println();

      // モデルの入力テンソルのサイズがFEATURE_NUMであることを確認
      size_t num_elements = input->bytes / sizeof(float);
      if (num_elements != FEATURE_NUM) {
        Serial.println("テンソルの要素数が想定と異なります！");
        continue;
      }
      // 入力テンソルへ特徴量コピー
      for (size_t i = 0; i < num_elements; i++) {
        input->data.f[i] = features[i];
      }
      // モデル推論実行
      if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed!");
        continue;
      }
      // 推論結果の出力（例：出力層の次元が 1x4 と仮定）
      int num_output = output->dims->data[1];
      Serial.print("Inference Output: ");
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
      
      // 推論結果に基づいてLEDの色を変更（例：0: leaf=赤, 1: non=緑, 2: soil=青, 3: stalk=紫）
      switch (predicted) {
        case 0:
          setColor(255, 0, 0);  // 赤
          Serial.println("Predicted: leaf");
          break;
        case 1:
          setColor(0, 255, 0);  // 緑
          Serial.println("Predicted: non");
          break;
        case 2:
          setColor(0, 0, 255);  // 青
          Serial.println("Predicted: soil");
          break;
        case 3:
          setColor(128, 0, 128);  // 紫
          Serial.println("Predicted: stalk");
          break;
        default:
          setColor(0, 0, 0);  // 黒（消灯）
          break;
      }
    }
  }
}

//-------------------------------------------------------------
// watchdog timer feed 関数
void feedTheDog() {
  TIMERG0.wdt_wprotect = TIMG_WDT_WKEY_VALUE;
  TIMERG0.wdt_feed = 1;
  TIMERG0.wdt_wprotect = 0;
  TIMERG1.wdt_wprotect = TIMG_WDT_WKEY_VALUE;
  TIMERG1.wdt_feed = 1;
  TIMERG1.wdt_wprotect = 0;
}

//-------------------------------------------------------------
// setup：TFLiteモデル初期化，タスク生成，LED初期化
void setup() {
  Serial.begin(921600);
  while (!Serial) {}

  // TFLiteモデルの初期化
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(e_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("TFLite Setup complete.");

  // NeoPixelの初期化
  pixels.begin();

  // キュー作成（特徴量ベクトルを格納、サイズは10個分）
  featureQueue = xQueueCreate(10, sizeof(float[FEATURE_NUM]));
  if (featureQueue == NULL) {
    Serial.println("Queue creation failed!");
    while (1);
  }

  // 2つのタスクを生成：コア0でデータ収集タスク，コア1で推論タスク
  xTaskCreatePinnedToCore(collectTask, "CollectTask", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(inferenceTask, "InferenceTask", 4096, NULL, 1, NULL, 1);
}

void loop() {
  // ループ内は短く、ウォッチドッグタイマーのフィードのみ実施
  feedTheDog();
  vTaskDelay(pdMS_TO_TICKS(1));
}