[简体中文](./README_cn.md) | [English](./README_en.md) | **日本語**

<p align="center">
  <img src="docs/static/images/logo/paddlets-readme-logo.png" align="middle" width=500>
<p>

------------------------------------------------------------------------------------------

<p align="center">
  <a href="https://github.com/PaddlePaddle/PaddleTS/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleNLP?color=9ea"></a>
  <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/paddlepaddle-2.3.0+-aff.svg"></a>
  <a href="https://github.com/PaddlePaddle/PaddleTS/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleTS?color=3af"></a>
  <a href="https://github.com/PaddlePaddle/PaddleTS/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleTS?color=9cc"></a>
</p>

--------------------------------------------------------------------------------


PaddleTS - Python による PaddlePaddle ベースの時系列モデリング

PaddleTS は、PaddlePaddle ディープラーニングフレームワークに基づく
    最先端のディープニューラルネットワークモデルに焦点を当てた、
    ディープ時系列モデリングのための使いやすい Python ライブラリです。
    実務家や専門家に優れた柔軟性と優れたユーザー体験を提供することを目的としています。特徴は以下の通りです:

* TSDataset という名前の統一されたデータ構造は、1 つまたは複数のターゲット変数と、オプションで異なる種類の共変量を持つ時系列データを表現するためのものです
    TSDataset という統一されたデータ構造
    （例えば、既知の共変量、観測された共変量、静的な共変量など）
* PaddleBaseModelImpl という基底モデルクラスは PaddleBaseModel を継承し、さらにいくつかのルーチン手続き（データロード、コールバックセットアップ）をカプセル化します
    を継承し、さらにいくつかのルーチン手続き（例えば、データのロード、コールバックのセットアップ、
    損失計算、トレーニングループ制御など）をカプセル化し、開発者がネットワークアーキテクチャの実装に集中できるようにします
    新しいモデルを開発する際、開発者はネットワークアーキテクチャの実装に集中することができます
* 最先端のディープラーニング・モデルのセット
    NBEATS、NHiTS、LSTNet、TCN、Transformer、DeepAR、Informer など、
    表現には TS2Vec、CoST など、
    異常検知のための AutoEncoder、VAE、AnomalyTransformer など
* データ前処理のための変換演算子一式（欠損値／外れ値の処理など、
    ワンホットエンコーディング，正規化，日付/時間に関連した共変量の自動生成など）
* 迅速なデータ探索のための分析演算子セット（基本統計やサマリーなど）
* 自動時系列モデリングモジュール (AutoTS) は、主流のハイパーパラメータ最適化アルゴリズムをサポートし、複数のモデルやデータセットで大幅な改善を示します
* サードパーティ(例えば scikit-learn、[pyod](https://github.com/yzhao062/pyod)) の ML モデルとデータ変換の統合
* 時系列モデルのアンサンブル

最近の更新:

* PaddleTS が時系列分類をサポート
* PaddleTS は 6 つの新しい時系列モデルをリリースします。
  USAD(UnSupervised Anomaly Detection) と MTAD-GAT(Multivariate Time-series Anomaly Detection via Graph Attention Network)による異常検知、
  時系列分類のための CNN と Inception Time、
  予測には SCINet(Sample Convolution and Interaction Network) と TFT(Temporal Fusion Transformer)
* [Paddle 推論](https://www.paddlepaddle.org.cn/paddle/paddleinference)が PaddleTS の時系列予測と異常検出に利用可能になりました
* PaddleTS はモデルに依存しない説明とモデルに依存する説明の両方をサポートします
* PaddleTS は表現ベースの時系列クラスタと分類をサポートします

[リリースノート](https://github.com/PaddlePaddle/PaddleTS/wiki/Release-Notes)も参照してください。

将来的には、以下のような高度な機能が追加される予定です:

* より多くの時系列モデル
* 実際のビジネス問題を解決するためのエンド・ツー・エンドのソリューションを提供することを目的としたシナリオ別のパイプライン
* その他


## PaddleTS について

具体的には、PaddleTS は以下のモジュールで構成されています:


| モジュール                                                                                                                  | 説明                                                                          |
|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [**paddlets.datasets**](https://paddlets.readthedocs.io/en/latest/source/modules/datasets/overview.html)                  | 統一された時系列表現（TSDataset）と、事前に構築された TSDataset を含むデータリポジトリ。 |
| [**paddlets.autots**](https://paddlets.readthedocs.io/en/latest/source/modules/autots/overview.html)                      | ハイパーパラメーターの自動チューニング。                                             |
| [**paddlets.transform**](https://paddlets.readthedocs.io/en/latest/source/modules/transform/overview.html)                | データの前処理とデータ変換。                                                       |
| [**paddlets.models.forecasting**](https://paddlets.readthedocs.io/en/latest/source/modules/models/overview.html)          | PaddlePaddle ベースの時系列予測のためのディープニューラルネットワークモデル 。           |
| [**paddlets.models.representation**](https://paddlets.readthedocs.io/en/latest/source/modules/models/representation.html) | PaddlePaddle ベースの時系列表現のためのディープニューラルネットワークモデル 。           |
| [**paddlets.models.anomaly**](https://paddlets.readthedocs.io/en/latest/source/modules/models/anomaly.html)               | 時系列異常検出のための PaddlePaddle ベースのディープニューラルネットワークモデル         |
| [**paddlets.models.classify**](https://paddlets.readthedocs.io/en/latest/source/api/paddlets.models.classify.html)        | 時系列分類のための PaddlePaddle ベースのディープニューラルネットワークモデル。           |
| [**paddlets.pipeline**](https://paddlets.readthedocs.io/en/latest/source/modules/pipeline/overview.html)                  | 時系列分析とモデリングのワークフローを構築するためのパイプライン。                       |
| [**paddlets.metrics**](https://paddlets.readthedocs.io/en/latest/source/modules/metrics/overview.html)                    | モデルのパフォーマンスを測定するための指標。                                          |
| [**paddlets.analysis**](https://paddlets.readthedocs.io/en/latest/source/modules/analysis/overview.html)                  | 迅速なデータ探索と高度なデータ分析。                                                |
| [**paddlets.ensemble**](https://paddlets.readthedocs.io/en/latest/source/modules/ensemble/overview.html)                  | 時系列アンサンブル法。                                                            |
| [**paddlets.xai**](https://paddlets.readthedocs.io/en/latest/source/api/paddlets.xai.html)                                | 時系列モデリングのモデル非依存的説明とモデル特異的説明。                               |
| [**paddlets.utils**](https://paddlets.readthedocs.io/en/latest/source/modules/backtest/overview.html)                     | ユーティリティ機能。                                                             |


## インストール

### 前提条件

* python >= 3.7
* paddlepaddle >= 2.3

pip で paddlets をインストールする:
```bash
pip install paddlets
```

より詳細な情報は、[インストール](https://paddlets.readthedocs.io/en/latest/source/installation/overview.html)を参照してください。


## ドキュメント

* [始める](https://paddlets.readthedocs.io/en/latest/source/get_started/get_started.html)

* [API リファレンス](https://paddlets.readthedocs.io/en/latest/source/api/paddlets.analysis.html)


## コミュニティ

以下の WeChat QR コードをスキャンして、PaddleTS コミュニティに参加し、PaddleTS メンテナーやコミュニティメンバーと技術的な議論をしてください:
<p align="center">
    <img src="docs/static/images/wechat_qrcode/wechat_qrcode.jpg" align="middle" height=300 width=300>
</p>

## コントリビュート

私たちはあらゆる種類の貢献に感謝します。バグがありましたら、[Filing an issue](https://github.com/PaddlePaddle/PaddleTS/issues) までお知らせください。

もしバグフィックスにご協力いただけるのであれば、それ以上の議論は必要ありません。

新しい機能、ユーティリティ機能、拡張機能などをコアに貢献する予定がある場合は、まず issue を発行し、私たちと議論してください。
議論なしに PR を送ると、拒絶されてしまうかもしれません。なぜなら、私たちがコアをあなたの認識とは違う方向に持っていってしまうかもしれないからです。


## ライセンス
[LICENSE](LICENSE)ファイルにあるように、PaddleTS は Apache スタイルのライセンスを持っています。
