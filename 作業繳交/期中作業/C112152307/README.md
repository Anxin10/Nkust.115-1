# BXJudge：Beyblade X 本機 AI 裁判

學號：C112152307

BXJudge 讀取 Beyblade X 比賽影片，自動判斷每一回合的勝方與得分方式（Spin、Over、Burst、Xtreme）。所有運算都在本機 GPU 上進行，影片不會上傳到雲端。

## 做法

1. **找出場地**：用 YOLO-pose 關鍵點模型找出場地上的 10 個特徵點，算出影片畫面和場地模板之間的對應關係，再把戰鬥區和各個得分袋套到畫面上。綠色軌道只用來輔助檢查。
2. **偵測與追蹤陀螺**：用 RF-DETR／YOLO 偵測器框出每顆陀螺，用 ByteTrack 追蹤。陀螺碰撞、互相遮擋時會暫停身分判定，不硬猜。
3. **分辨雙方（X／B）**：從發射器的位置和雙方分開後的連續證據，判斷每顆陀螺屬於哪一方。
4. **判定**：依官方規則檢查發射、落地、停轉、入袋，產生勝方與得分方式。證據不足時標成「需要人工覆核」，不輸出錯誤的判決。
5. **覆核與評估**：用本機網頁介面修正結果，再拿人工答案計算準確率。

## 目前成果

- 場地自動定位：在 434 支實拍影片中找到 369 支（85.0%）。找到的影片裡，四個判定區相對軌道的平均位移都不超過 40 像素。
- 判決採保守設計：證據不足就交人工，不追求全部自動判定。

## 專案結構

| 資料夾 | 內容 |
|---|---|
| `src/bxjudge/` | 主程式：場地定位、偵測追蹤、身分判定、規則判決、覆核介面 |
| `config/` | 各階段的門檻與規則設定（YAML） |
| `scripts/` | 評估、標註、匯出等輔助腳本 |
| `tests/` | pytest 測試 |

## 執行環境

- WSL2 Ubuntu、Python 3.12、NVIDIA GPU（開發時使用 RTX 3090）
- 用 [uv](https://docs.astral.sh/uv/) 管理套件

```bash
uv sync
uv run pytest -m "not gpu"
```

## 主要指令

```bash
uv run bxjudge scan <影片資料夾>        # 列出新的或變更過的影片
uv run bxjudge calibrate <影片>.mp4     # 手動標定場地區域（自動定位失敗時使用）
uv run bxjudge train --kind yolo26 --dataset <資料集> --out <輸出資料夾>
uv run bxjudge run <影片資料夾>         # 判定影片中的每一回合
uv run bxjudge run --rejudge <影片資料夾> # 重新判定已處理過的影片
uv run bxjudge review                   # 開啟本機覆核介面
uv run bxjudge eval                     # 用人工答案計算準確率
```

比賽影片、標註資料和模型權重都沒有放進這個資料夾，原因是影片含比賽畫面，權重檔案太大。
