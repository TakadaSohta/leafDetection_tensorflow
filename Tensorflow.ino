#include <Arduino.h>
#include "freertos/task.h"
#include "soc/timer_group_struct.h"
#include "soc/timer_group_reg.h"
#include <math.h>
#include <driver/ledc.h>
#include <Adafruit_NeoPixel.h>

//==================================
// TFLite 関連
//==================================
#include <TensorFlowLite_ESP32.h>
#include "model.h"  // TFLite用モデルデータ（model.h）
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

//==================================
// KNN 関連
//==================================
#include "knn_model.h"

//==================================
// ユーザー設定など
//==================================
#define pi 3.141592653589793
#define REFLESH_RATE 1000.0

// DAC ピン
const int Wave = 25;  

// NeoPixel
#define NUMPIXELS 24
#define LED_OUT 27

// PWM 出力ピン
#define PWM_OUT 5

// ADC 入力ピン
#define ADC_IN 34

// 周波数スイープ設定
#define SAMPLE_NUM 100      // 1～100 ステップ
#define INITIAL_FRQ 5000    // 1ステップあたり 5kHz → 最大 500kHz
float results[SAMPLE_NUM + 1];  // 各周波数のADC応答（0～4095）
float freqArr[SAMPLE_NUM + 1];  // （必要なら実際の周波数を保存可能）

// 推論モード: 0=TFLite, 1=KNN
volatile int inferenceMode = 0;

//==================================
// 推論用入力: トップ10ピークの「周波数」
//（※モデルには kHz 単位で入力する）
//==================================
#define FEATURE_NUM 10

//----------------------------------
// ピーク検出用 構造体
//----------------------------------
struct PeakInfo {
  float freqHz;  // 周波数（Hz単位）
  float amp;     // 振幅（ADC値）
};

//----------------------------------
// results[] から上位10ピークを抽出し、
// 周波数を kHz 単位に変換して features[] に格納
//----------------------------------
void getTop10PeakFreq(float features[FEATURE_NUM]) {
  PeakInfo peaks[SAMPLE_NUM];
  for (int d = 1; d <= SAMPLE_NUM; d++) {
    peaks[d - 1].freqHz = (float)(INITIAL_FRQ * d); // 例：5kHz * d (Hz)
    peaks[d - 1].amp    = results[d];
  }
  // 振幅の大きい順にバブルソート
  for (int i = 0; i < SAMPLE_NUM - 1; i++) {
    for (int j = 0; j < SAMPLE_NUM - 1 - i; j++) {
      if (peaks[j].amp < peaks[j + 1].amp) {
        PeakInfo tmp = peaks[j];
        peaks[j]     = peaks[j + 1];
        peaks[j + 1] = tmp;
      }
    }
  }
  // 上位10件の周波数を kHz 単位に変換して features[] にコピー
  for (int i = 0; i < FEATURE_NUM; i++) {
    features[i] = peaks[i].freqHz / 1000.0f;  // 例: 150kHz → 150
  }
}

//==================================
// NeoPixel
//==================================
Adafruit_NeoPixel pixels(NUMPIXELS, LED_OUT, NEO_GRB + NEO_KHZ800);
void setColor(int red, int green, int blue) {
  for (int i = 0; i < NUMPIXELS; i++) {
    pixels.setPixelColor(i, pixels.Color(red, green, blue));
  }
  pixels.show();
}

//==================================
// Watchdog
//==================================
void feedTheDog() {
  TIMERG0.wdt_wprotect = TIMG_WDT_WKEY_VALUE;
  TIMERG0.wdt_feed = 1;
  TIMERG0.wdt_wprotect = 0;
  TIMERG1.wdt_wprotect = TIMG_WDT_WKEY_VALUE;
  TIMERG1.wdt_feed = 1;
  TIMERG1.wdt_wprotect = 0;
}

//==================================
// DAC で任意波形を生成 (pin25)
//==================================
float time_s = 0.0;
float m = 0; // シリアルコマンドで制御する周波数因子
void genWave() {
  time_s += 1.0 / REFLESH_RATE;
  if (time_s > 1.0) {
    time_s = 0.0;
  }
  int dacValue = (int)(127.0 + 10.0 * sin(2.0 * pi * m * time_s));
  dacWrite(Wave, dacValue);
}

//==================================
// 別タスクで DAC 波形生成
//==================================
void task0(void* param) {
  while (1) {
    genWave();
    vTaskDelay(1);
    feedTheDog();
  }
}

//==================================
// TFLite 関連
//==================================
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflite_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 8 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

void initTFLite() {
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
}

//==================================
// setup
//==================================
void setup() {
  // 別タスクで DAC 波形生成
  xTaskCreatePinnedToCore(task0, "Task0", 4096, NULL, 1, NULL, 0);

  // PWM 初期設定（後で周波数スイープに使用）
  ledcSetup(0, 1000, 1);
  ledcAttachPin(PWM_OUT, 0);
  ledcWrite(0, 1);

  pinMode(LED_OUT, OUTPUT);
  pinMode(Wave, OUTPUT);

  Serial.begin(115200);
  while (!Serial) {}

  pixels.begin();
  setColor(0, 0, 0);

  // TFLite 初期化
  initTFLite();

  // 配列初期化
  for (int i = 0; i <= SAMPLE_NUM; i++) {
    results[i] = 0;
    freqArr[i] = 0;
  }

  Serial.println("System starting. Default mode: TFLite");
}

//==================================
// loop
//==================================
void loop() {
  // 1. 周波数スイープ：各ステップのADC値を取得
  for (unsigned int d = 1; d <= SAMPLE_NUM; d++) {
    int frq = INITIAL_FRQ * d;  // 例：5kHz * d
    ledcSetup(0, frq, 1);
    ledcWrite(0, 1);
    int v = analogRead(ADC_IN);
    results[d] = results[d] * 0.5f + (float)v * 0.5f;  // IIRフィルタ
  }
  
  // 2. 現在のトップ10ピーク（周波数）を kHz 単位で取得
  float features[FEATURE_NUM];
  getTop10PeakFreq(features);
  
  // 3. 各ループで取得したピーク周波数を静的に蓄積（10回分）
  static float freqSum[FEATURE_NUM] = {0};
  static int freqCycleCount = 0;
  for (int i = 0; i < FEATURE_NUM; i++) {
    freqSum[i] += features[i];
  }
  freqCycleCount++;
  
  // 4. 10回分の蓄積が完了したら平均値を計算して推論に利用
  if (freqCycleCount >= 10) {
    float avgFeatures[FEATURE_NUM];
    for (int i = 0; i < FEATURE_NUM; i++) {
      avgFeatures[i] = freqSum[i] / freqCycleCount;
    }
    
    // デバッグ用：平均化した周波数ピーク値を表示（kHz単位）
    Serial.print("Averaged Frequency Peaks (kHz): ");
    for (int i = 0; i < FEATURE_NUM; i++) {
      Serial.printf("%0.2f ", avgFeatures[i]);
    }
    Serial.println();
    
    // 5. 推論処理：平均値を入力として TFLite または KNN で判定
    if (inferenceMode == 0) {
      // --- TFLite ---
      size_t numElements = input->bytes / sizeof(float);
      if (numElements != FEATURE_NUM) {
        Serial.println("ERROR: TFLite model expects different input size!");
      } else {
        for (int i = 0; i < FEATURE_NUM; i++) {
          input->data.f[i] = avgFeatures[i];
        }
        if (interpreter->Invoke() != kTfLiteOk) {
          Serial.println("Invoke failed!");
        } else {
          int num_output = output->dims->data[1];  // 例：4クラス
          Serial.print("TFLite Output: ");
          for (int i = 0; i < num_output; i++) {
            Serial.printf("%0.4f ", output->data.f[i]);
          }
          Serial.println();
          
          // 最も大きい出力確率のクラスを判定
          float maxProb = output->data.f[0];
          int predicted = 0;
          for (int i = 0; i < num_output; i++) {
            if (output->data.f[i] > maxProb) {
              maxProb = output->data.f[i];
              predicted = i;
            }
          }
          // LED 表示更新
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
              Serial.println("TFLite Predicted: ???");
              break;
          }
        }
      }
    } else {
      // --- KNN ---
      int predicted = knn_predict(avgFeatures);
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
    
    // 蓄積用変数をリセット
    freqCycleCount = 0;
    for (int i = 0; i < FEATURE_NUM; i++) {
      freqSum[i] = 0;
    }
  }
  
  // 6. 現在のADC値取得結果（トップ10ピーク）を表示（デバッグ用）
  {
    PeakInfo peaks[SAMPLE_NUM];
    for (int d = 1; d <= SAMPLE_NUM; d++) {
      peaks[d - 1].freqHz = (float)(INITIAL_FRQ * d);
      peaks[d - 1].amp    = results[d];
    }
    // バブルソートで上位10件抽出
    for (int i = 0; i < SAMPLE_NUM - 1; i++) {
      for (int j = 0; j < SAMPLE_NUM - 1 - i; j++) {
        if (peaks[j].amp < peaks[j + 1].amp) {
          PeakInfo tmp = peaks[j];
          peaks[j] = peaks[j + 1];
          peaks[j + 1] = tmp;
        }
      }
    }
    Serial.println("Current Top 10 Peaks (Frequency [Hz] : ADC Value):");
    for (int i = 0; i < FEATURE_NUM; i++) {
      Serial.print(peaks[i].freqHz);
      Serial.print(" Hz : ");
      Serial.println(peaks[i].amp);
    }
  }
  
  // 7. シリアルコマンドで m の変更および推論モードの切替
  if (Serial.available() > 0) {
    int incomingByte = Serial.read();
    switch (incomingByte) {
      case 'N':  // No Touch
        m = 0;
        setColor(0, 0, 0);
        Serial.println("Set m=0 (No Touch)");
        break;
      case 'L':  // Leaf
        m = 60;
        setColor(255, 0, 0);
        Serial.println("Set m=60 (Leaf)");
        break;
      case 'S':  // Stalk
        m = 120;
        setColor(0, 255, 0);
        Serial.println("Set m=120 (Stalk)");
        break;
      case 'G':  // Ground
        m = 240;
        setColor(0, 0, 255);
        Serial.println("Set m=240 (Ground)");
        break;
      case 'T':  // TFLite モード
        inferenceMode = 0;
        Serial.println("Switched to TFLite mode.");
        break;
      case 'K':  // KNN モード
        inferenceMode = 1;
        Serial.println("Switched to KNN mode.");
        break;
      default:
        break;
    }
  }
  
  // 8. Watchdog のフィード
  feedTheDog();
}


