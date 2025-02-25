# Leafdetection_tensorflow
![TensorFlow Logo](https://www.tensorflow.org/images/tf_logo_social.png)

## 概要 / Overview
本プロジェクトは、複数のCSVファイルからセンサデータのピーク情報を抽出し、TensorFlow/Kerasを用いてニューラルネットワークを構築・学習するものです。  
The project extracts peak information from sensor data stored in multiple CSV files and uses TensorFlow/Keras to build and train a neural network model.

学習済みモデルはTensorFlow Lite形式に変換され、ESP32（またはESP32-S3）上で推論を実行するため、C言語のヘッダーファイルに変換して利用します。  
The trained model is converted to TensorFlow Lite format and then transformed into a C header file for inference on ESP32 (or ESP32-S3).

ESP32側では、ADCから取得したセンサデータに対してFFTを実施し、上位10ピーク（各ピークの周波数と振幅）の合計20個の特徴量を抽出。  
On the ESP32 side, an FFT is applied to the sensor data acquired via ADC, extracting 20 features (frequency and amplitude pairs for the top 10 peaks).

その後、TensorFlow Liteモデルにより入力データを分類し、分類結果に応じたNeoPixel LEDの制御を行います。  
The model then classifies the input data, and the classification results are used to control the NeoPixel LED colors.

## 特徴 / Features
- **データ前処理と統合 / Data Preprocessing and Integration**  
  CSVファイル（例：`leaf.csv`, `non.csv`, `stalk.csv`, `soil.csv`）からピーク情報を抽出し、`combined_peak_dataset.csv` を生成。  
  Extracts peak information from CSV files (e.g., `leaf.csv`, `non.csv`, `stalk.csv`, `soil.csv`) and generates a combined dataset.

- **ニューラルネットワークの学習 / Neural Network Training**  
  TensorFlow/Kerasを用いて、抽出した特徴量（ピークの周波数と振幅）から分類モデルを学習。  
  Trains a classification model using the extracted features (frequency and amplitude of peaks) with TensorFlow/Keras.

- **TensorFlow Lite変換 / TensorFlow Lite Conversion**  
  学習済みモデルをTensorFlow Lite形式に変換し、ESP32向けのヘッダーファイル（model.h）を生成。  
  Converts the trained model to TensorFlow Lite format and generates a C header file (model.h) for ESP32.

- **ESP32での推論とLED制御 / Inference and LED Control on ESP32**  
  ADCからのセンサデータをFFTで処理し、TensorFlow Liteモデルにより分類。分類結果に応じたNeoPixel LED制御を実施。  
  Processes sensor data from the ADC via FFT, performs inference with the TensorFlow Lite model, and controls the NeoPixel LED based on the classification result.

- **Modelの作成に関しては以下のColabを参考にしてください**
  https://colab.research.google.com/drive/1cXcK1S6Bp6tIH6z4TiKSWayCkn_2JPR2?usp=sharing


## ファイル構成 / File Structure
- **data_processing.py**  
  複数のCSVファイルからピーク情報を抽出し、統合データセット (`combined_peak_dataset.csv`) を生成するスクリプト。  
  A script that extracts peak information from multiple CSV files and generates the combined dataset.

- **model_training.py**  
  統合データセットを使用してニューラルネットワークの構築、学習、評価を行うスクリプト。  
  A script to build, train, and evaluate the neural network model using the combined dataset.

- **model_conversion.py**  
  学習済みKerasモデルをTensorFlow Lite形式に変換し、`.tflite` ファイルとして保存するスクリプト。  
  A script that converts the trained Keras model to TensorFlow Lite format and saves it as a `.tflite` file.

- **convert_tflite_to_header.py**  
  `.tflite` ファイルをC言語のヘッダーファイル（`model.h`）に変換するスクリプト。  
  A script to convert the `.tflite` file into a C header file (`model.h`).

- **Tensorflow.ino**  
  ESP32/ESP32-S3向けのArduinoコード。ADCデータの取得、TensorFlow Liteによる推論、NeoPixel LED制御を実装。  
  Arduino code for ESP32/ESP32-S3. Implements ADC data acquisition, FFT-based feature extraction, TensorFlow Lite inference, and NeoPixel LED control.

- **model.h**  
  `convert_tflite_to_header.py` により生成された、TensorFlow Liteモデルデータを格納するヘッダーファイル。  
  The header file containing the TensorFlow Lite model data, generated by `convert_tflite_to_header.py`.

## 環境と要件 / Environment and Requirements
- **Python**  
  - ライブラリ: pandas, numpy, tensorflow, scikit-learn 等  
  - Libraries: pandas, numpy, tensorflow, scikit-learn, etc.
  
- **Arduino IDE**  
  - ESP32/ESP32-S3用のArduinoコア（最新版推奨）  
  - 使用ライブラリ: TensorFlow Lite for Microcontrollers, Adafruit NeoPixel, ArduinoFFT  
  - Arduino IDE with ESP32/ESP32-S3 core (latest version recommended)  
  - Libraries: TensorFlow Lite for Microcontrollers, Adafruit NeoPixel, ArduinoFFT

- **ハードウェア / Hardware**  
  - ESP32 または ESP32-S3 開発ボード  
  - ADCセンサ（例: タッチセンサ）  
  - NeoPixel LEDストリップ  
  - ESP32 or ESP32-S3 development board  
  - ADC sensor (e.g., touch sensor)  
  - NeoPixel LED strip

## 使用方法 / How to Use

### 1. データ前処理とモデル学習 / Data Preprocessing and Model Training
1. 各CSVファイル（例：`leaf.csv`, `non.csv`, `stalk.csv`, `soil.csv`）をプロジェクトディレクトリに配置。  
   Place the CSV files (e.g., `leaf.csv`, `non.csv`, `stalk.csv`, `soil.csv`) in the project directory.
2. `data_processing.py` を実行し、ピーク情報を抽出して `combined_peak_dataset.csv` を生成。  
   Run `data_processing.py` to extract peak information and generate `combined_peak_dataset.csv`.
3. `model_training.py` を実行し、ニューラルネットワークの学習・評価を実施。  
   Run `model_training.py` to train and evaluate the neural network.
4. `model_conversion.py` を実行し、TensorFlow Lite形式のモデル（`model.tflite`）を生成。  
   Run `model_conversion.py` to generate the TensorFlow Lite model (`model.tflite`).
5. `convert_tflite_to_header.py` を実行し、`model.tflite` を Cヘッダーファイル（`model.h`）に変換。  
   Run `convert_tflite_to_header.py` to convert `model.tflite` into the C header file (`model.h`).

### 2. ESP32への書き込みと推論実行 / Uploading to ESP32 and Running Inference
1. Arduino IDEで `Tensorflow.ino` を開く。  
   Open `Tensorflow.ino` in the Arduino IDE.
2. ESP32またはESP32-S3のボード設定および必要ライブラリのインストールを確認。  
   Ensure that the ESP32/ESP32-S3 board is selected and the necessary libraries are installed.
3. プログラムをコンパイルし、開発ボードに書き込む。  
   Compile and upload the program to your development board.
4. シリアルモニターで推論結果を確認し、NeoPixel LEDの色が変化することを確認。  
   Check the serial monitor for inference results and verify that the NeoPixel LED changes color accordingly.

### 3. ESP32-S3向け最適化 / Optimizations for ESP32-S3
ESP32-S3では、ハードウェアアクセラレーション（SIMD/DSP拡張）により推論が高速化されます。  
On ESP32-S3, hardware acceleration (SIMD/DSP extensions) can speed up inference.
Arduino IDEの `platform.local.txt` を使用して、追加のコンパイラフラグ（例：`-O3 -mve`）を設定することで最適化可能です。  
You can optimize by setting additional compiler flags (e.g., `-O3 -mve`) using `platform.local.txt` in the Arduino IDE.

## ライセンス / License
本プロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご確認ください。  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## お問い合わせ / Contact
ご質問や改善案がありましたら、IssueやPull Requestでお知らせください。  
For any questions or suggestions, please open an issue or submit a pull request.

---

## 自己紹介 / About Me
**高田崇天**  
Email: takada[at].iit.tsukuba.ac.jp

私は、IoTシステムや組み込み機器を対象とした機械学習やセンサデータ処理の研究・開発を行っています。  
I work on research and development in the field of machine learning and sensor data processing for IoT systems and embedded devices.

お気軽にご連絡ください！  
Feel free to contact me!
