## 1st solution

Discussion: https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/97815  

Github: https://github.com/qrfaction/1st-Freesound-Audio-Tagging-2019

- feature
  - log-mel (shape=(441, 64)) の他にglobal features (shape=(128, 12))とlength
  - global featuresは音声データを時間方向に等分割して基礎統計量を計算し作成

- model
  - global featuresはRNNに投げて（電線の時と似たRNNアーキテクチャ）、log-melはCNNに投げる。
  - RNNとCNNからの出力をconcatしてDenseで出力
  - log-melの各軸はそれぞれ物理的意味の異なる（時間軸と周波数軸）であるため、最終層付近で時間方向にmax-poolingした後、周波数方向にavg-pooling
- training
  - noisyデータのみを使ってpetrained modelを作成

- ensemble
  - NNでstacking
  - LocallyConnected1Dの重みを各モデルのアンサンブル時の重みと見立てることができる



## 2nd solution

Discussion: https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/95924  

Github: https://github.com/lRomul/argus-freesound  

- PadToSizeでパディング、SpecAugmentも使用（PytorchのAugmentationコードの書き方が参考になる）
- mihiro2さんkernelのsimple cnnをベースとして、attentionやskip connectionを導入
- curatedデータのhand relabeling



## 3rd solution

Discussion: https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/97926  

Github: https://github.com/ex4sperans/freesound-classification/tree/master  

- Max-poolingの代わりにRNNを使うケースも試した。精度は悪くなったがアンサンブル要員として良く機能した。
- ランクに基づく損失関数を使いたかったので、LSEP loss(https://arxiv.org/abs/1704.03135)を採用した。CV0.015分の改善が得られた。

- curatedデータのみで学習したモデルでnoisyデータをpredictし、noisyの元のラベルと予測結果が一致するものを採用しcuratedデータセットに加える。採用されたnoisyデータの数が5k分に達するまで繰り返した。



## 4th solution

Discussion: https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/96440  

Kernel: https://www.kaggle.com/osciiart/freesound2019-solution-links?scriptVersionId=16047723  

- モデルはlog-melをinputとしたResnet34と、raw waveをinputとしたEnvNetV2
- 音源の長さがバラバラだがスライスせずに全長使うのが良かった。理由は分類に必要な部分が冒頭に集中しているから。実装は、全ファイルを音源の長さにソートして長さの近いもの同士でグループ化、グループ内で長さ揃えてミニバッチに変換。
- noisyデータに関してはマルチタスク学習させることで、特徴量をいい感じに抽出
- アンサンブル時に、時間幅の異なるモデルを混ぜるのが効く（4secと8sec）



## 7th solution

Discussion: https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/97812

Kernel: https://www.kaggle.com/hidehisaarai1213/freesound-7th-place-solution  

- augmentationにsoxを使用
- Cropする際、dBの合計値が大きい部分を優先的に選択する。
- RandomResizedCropを使うとvalidationスコアのバラつきが大きいため、validationスコアの計算時にもttaを適用する。