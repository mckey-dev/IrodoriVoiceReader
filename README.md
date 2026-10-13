# Irodori Voice Reader

Version: **v1.0.0**  
更新日: 2026-09-18

ローカルの [Irodori-TTS V4.1 Anime](https://huggingface.co/phasefield-audio/Irodori-TTS-v4.1-Anime) と Chrome 拡張で、Web ページの日本語を読み上げるプロジェクトです。構成は OmniVoice Reader を参考にしています。

- サーバー: 手元の GPU（NVIDIA CUDA / AMD ROCm）または CPU で音声を生成する
- 拡張: ツールバーから操作ウィンドウを開き、ページ本文または選択テキストを読み上げる
- 再生中は、操作ウィンドウに現在の文を出す。ページ上のハイライトは設定でオンにできる
- ChatGPT / Gemini では、完成した解答を自動で読む設定がある（既定はオフ）
- 声は未選択でも読める（既定のアニメ声）。操作ウィンドウの「声」から参照音声を登録するとクローンできる
- モデルは日本語入力専用です

開発環境では、以下の GPU を想定しています。

- NVIDIA GPU（推奨。`start_cuda.bat`。公式 extra は CUDA 12.8。仮想環境は `venv_cuda`）
- AMD GPU / ROCm（`start_amd.bat`。Windows は RX 7900 シリーズ / gfx1100 と Radeon 780M / gfx1103 の ROCm 7.14。仮想環境は `venv_amd`）

## 必要なもの

- Windows（Linux 用の起動スクリプトもあります）
- Python 3.10 以上
- [uv](https://docs.astral.sh/uv/)（無ければ起動スクリプトが `pip install uv` を試します）
- Google Chrome
- GPU を使う場合は、対応する PyTorch（CUDA 12.8 または ROCm）
- FFmpeg は不要（WAV / OGG / FLAC は `soundfile`、MP3 は `miniaudio`）

## ディレクトリの見取り

| パス | 内容 |
| --- | --- |
| `server/` | ローカル TTS サーバー（`python -m server` → 既定 `http://127.0.0.1:8000`） |
| `server/config.py` | `HOST` / `PORT` と生成既定値 `DEFAULT_*` |
| `server/app.py` | FastAPI の HTTP ルート |
| `server/runtime.py` | GPU・モデル・参照音声の実行時状態 |
| `server/sampling/` | 参照音声の WAV 変換と声ライブラリ |
| `extension/` | Chrome 拡張（MV3） |
| `extension/sentence.js` | 読み上げ単位への分割 |
| `extension/settings.json` | 再生速度・音量・ハイライト／チャット自動・生成オプション |
| `models/Irodori-TTS-v4.1-Anime` | TTS モデル。無ければ起動時にダウンロード |
| `models/Semantic-DACVAE-Japanese-32dim` | 音声コーデック。無ければ起動時にダウンロード |
| `voice/` | 登録した参照音声（元ファイル・wav） |
| `test/test_client.html` | 拡張を使わないブラウザ単体の試験 UI |
| `test/test_sentence_split.mjs` | 文分割の確認用 |

## 1. 仮想環境と依存パッケージ

起動スクリプトは [uv](https://docs.astral.sh/uv/) で依存関係を入れます。初回は Irodori-TTS 本体と PyTorch の取得に時間がかかります。

| 起動 | extra | 仮想環境 | 用途 |
| --- | --- | --- | --- |
| `start.bat` / `start.sh` | `cpu` | `venv` | CPU |
| `start_amd.bat` / `start_amd.sh` | `rocm` | `venv_amd` | AMD GPU / ROCm |
| `start_cuda.bat` / `start_cuda.sh` | `cu128` | `venv_cuda` | NVIDIA CUDA 12.8 |

手動で入れる場合:

```powershell
cd C:\app\IrodoriVoiceReader
uv sync --extra cu128
uv run --no-sync python -m server
```

注意:

- PyTorch の backend extra は同時に使えません。`cpu` / `cu128` / `rocm` から 1 つだけ選んでください。仮想環境も分かれています（`venv`、`venv_amd`、`venv_cuda`）。
- Windows の `rocm` extra は AMD の wheel（ROCm 7.14）です。対象は RX 7900 XTX（gfx1100）と Radeon 780M（gfx1103）です。Linux は従来どおり PyTorch 公式の ROCm 7.1 です。

## 2. モデル

モデルはリポジトリ直下の `models` に置きます。Hugging Face のユーザーキャッシュは使いません。

| フォルダ | 用途 |
| --- | --- |
| `models/Irodori-TTS-v4.1-Anime` | TTS。無ければ起動時に `phasefield-audio/Irodori-TTS-v4.1-Anime` をダウンロード |
| `models/Semantic-DACVAE-Japanese-32dim` | 波形の encode / decode。無ければ起動時にダウンロード |

## 3. 声クローン

参照音声は任意です。選ばなければ、モデル既定のアニメ声で読みます。操作ウィンドウの「声」からアップロードすると、その声でクローンします。

保存先は `voice/{id}/` です。アップロード名 `自分の声.mp3` なら、同じ stem で WAV も残します。

```
voice/{id}/
  自分の声.mp3
  自分の声.wav
```

元が WAV なら変換コピーは作りません。書き起こしや Whisper は使いません。消すときは拡張の削除だけです。

## 4. TTS サーバーの起動

Windows:

```bat
start.bat
start_amd.bat
start_cuda.bat
```

Linux:

```bash
chmod +x start.sh start_amd.sh start_cuda.sh
./start.sh
./start_amd.sh
./start_cuda.sh
```

手動で起動する場合:

```powershell
uv sync --extra cu128
uv run --no-sync python -m server
```

待ち受けは既定で `127.0.0.1:8000` です。`--host` と `--port` で変えられます。起動スクリプトにも同じ引数を渡せます。

```powershell
uv run --no-sync python -m server --host 0.0.0.0 --port 8080
start_cuda.bat --port 8080
```

Chrome 拡張と `test/test_client.html` は `http://127.0.0.1:8000` 固定です。ポートを変える場合は、そちらも合わせてください。

起動後:

| URL | 内容 |
| --- | --- |
| http://127.0.0.1:8000/ | 稼働確認 |
| http://127.0.0.1:8000/health | ヘルスチェック |
| `POST /tts` | 読み上げ。本文は `{"text": "こんにちは。"}`。声が未選択なら既定のアニメ声 |
| http://127.0.0.1:8000/tts/defaults | 生成オプションの既定値 |
| http://127.0.0.1:8000/extension/settings | 拡張設定の読み書き |
| http://127.0.0.1:8000/voices | 声の一覧・追加・選択・削除 |

`POST /tts` で省略できる項目と、サーバー既定値は次のとおりです。

| 項目 | 既定 |
| --- | --- |
| `caption` | 空（未指定。下の「caption の書き方」を参照） |
| `speed` | 1.0（下の「speed の意味」を参照） |
| `num_steps` | 40 |
| `cfg_scale_text` | 3.0 |
| `cfg_scale_caption` | 3.0 |
| `cfg_scale_speaker` | 5.0 |
| `t_schedule_mode` | `linear`（`sway` も可） |
| `sway_coeff` | -1.0（`t_schedule_mode=sway` のとき使う） |

既定値は `server/config.py` の `DEFAULT_*` で変えます。操作ウィンドウの「生成設定」と `test/test_client.html` からも同じ項目を送れます。

操作ウィンドウ上部の「速度」は、生成済み音声の再生速度です。生成時の話す速さは `speed` です。

### speed の意味

`speed` は Irodori-TTS が音声を作るときの話す速さです。再生用の「速度」スライダーとは別です。内部では `duration_scale = 1 / speed` に変換します。

| 値 | 効果 |
| --- | --- |
| `1.0` | 既定 |
| `1` より大きい | 短く、速く話す |
| `1` より小さい | 長く、遅く話す |

例: `0.8` はゆっくり、`1.2` は速めです。

### caption の書き方

`caption` は Voice Design 用の自由文です。空、または未指定ならキャプション条件は使いません。参照音声と併用することもできます。

この Anime モデルは、元モデルとは別の注釈で追加学習しているため、caption や絵文字の効き方はベースモデルと違うことがあります。

例:

| caption | ねらい |
| --- | --- |
| `明るく短いアニメ声で、自然に読んでください。` | 明るい短めの口調 |
| `落ち着いた低めの声。丁寧で穏やかな話し方。` | 落ち着いた読み |
| `少し急ぎ気味に、テンポよく話してください。` | テンポを上げる |

入力テキスト側の絵文字でも、話し方や非言語的な発声が変わることがあります。

## 5. Chrome 拡張の読み込み

1. Chrome で `chrome://extensions` を開く
2. 右上の「デベロッパーモード」をオンにする
3. 「パッケージ化されていない拡張機能を読み込む」
4. このリポジトリの `extension` フォルダを選ぶ

ツールバーの Irodori Web Reader をクリックすると、操作ウィンドウが開きます（ツールバー下の小さなポップアップではありません）。位置は、直前のマウス位置に近いところです。ページをクリックしてもこの窓は閉じません。すでに開いていれば、その窓を前面へ出します。

拡張を更新したあとは、拡張の再読み込みに加えて、読み上げ対象のページも再読み込みしてください。

## 6. 使い方

1. サーバーを起動する
2. 読み上げたいページを開く
3. ツールバーの Irodori Web Reader をクリックして操作ウィンドウを開く

| 操作 | 内容 |
| --- | --- |
| 本文を抽出 | ページの記事本文だけを表示する |
| このページを読み上げる | 抽出した本文を順に読み上げる |
| 選択範囲を読み上げる | ページ上で選択したテキストだけを読み上げる |
| 一時停止 / 再開 / 停止 | 再生を制御する |
| 速度 / 音量 / 次の文まで | 再生側。速度と音量は再生中でも変更できる。次の文までは、前の文が終わってから次を始めるまでの待ち時間（0〜2000ms、既定 500ms）。待ちのあいだも次の文の音声変換は続ける |
| 読み上げ中の文をハイライト | ページ上で現在の文を黄色く囲む。既定はオフ |
| チャットの解答を自動で読む | ChatGPT / Gemini で、オンにしたあとに完成した解答を読む。既定はオフ |
| 声 | 参照なし、または参照音声のアップロード・選択・削除 |
| 生成設定 | `caption`、`speed`、`num_steps` など、サーバーへ送る生成オプション |

ページ上でテキストを選択して右クリックし、「選択テキストをIrodori Web Readerで読み上げ」でも同じことができます。

読み上げ中は、操作ウィンドウに現在の文が出ます。ページ上の黄色いハイライトは、設定でオンにしたときだけです。操作ウィンドウを閉じても再生は続きます。

「チャットの解答を自動で読む」は `chatgpt.com` / `chat.openai.com` / `gemini.google.com` だけです。オンにした時点ですでに画面にあるログは読まず、そのあと完成したアシスタント解答だけを読みます。生成中は待って、テキストが止まってから開始します。コードブロックは除きます。次の解答が完成したら、いま読んでいるものを止めて切り替えます。ハイライト設定とは独立です。

文の切り方は `extension/sentence.js` で、次の順です。

1. HTML から取るとき、`<p>` `<div>` `<li>` `<br>` などのブロックの切れ目と、テキスト中の改行を改行として残す。画面幅での折り返しは改行にしない。テキストファイルの改行もそのまま使う
2. 改行ごとに 1 文とする。空行は捨て、行をまたいで結合しない
3. 50 文字未満の行は、句読点があってもそのまま 1 単位
4. 50 文字以上の行は、先頭 50 文字以内で区切りを繰り返す（区切り文字は直前の単位に残す）。第一条件としてその窓内の最後の `。` または `. `（ピリオド+半角スペース）を使う。第一条件が無いときだけ、同じ窓内の最後の `、` `！` `？` または `, `（カンマ+半角スペース）を使う。`1,000` のような数字内の `, ` は第二条件にしない
5. 区切ったあとも 50 文字を超える塊は、先頭 50 文字に `…以下略` を付け、続きは読まない
6. 各読み上げ単位の末尾に `。` が無ければ付与する（末尾が `.` の英文単位はそのまま。省略した単位は `…以下略。` で終わる）

## 7. ブラウザ単体のテスト UI

拡張を使わず、テキストを直接読み上げたい場合:

1. サーバーを起動する
2. `test/test_client.html` をブラウザで開く
3. 文章を入力して「読み上げ開始」

文分割だけを確認する場合:

```powershell
node test/test_sentence_split.mjs
```

## 8. 調査用ログ

読み上げが止まったように見えるときは、次を見ます。

| 場所 | 見られるもの |
| --- | --- |
| サーバー窓 | リクエスト ID、文字数、生成秒数、サンプル数 |
| Offscreen のコンソール | `[playback]` `[tts]` `[audio]` |
| service worker のコンソール | 抽出文字数・行数（`background.js`） |
| ページのコンソール | 抽出とハイライトの成否。チャット自動は `chat_auto.js` |
| 操作ウィンドウ | 失敗時の「読み上げエラー」 |

Offscreen と service worker のコンソールは、`chrome://extensions` の拡張詳細から開けます。

## ライセンス

- このリポジトリのコード: MIT License
- モデル重み: [Irodori-TTS-v4.1-Anime](https://huggingface.co/phasefield-audio/Irodori-TTS-v4.1-Anime) とベースモデルの MIT / 倫理的制限に従う
- 推論実装: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
