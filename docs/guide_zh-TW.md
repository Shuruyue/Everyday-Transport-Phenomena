# 使用說明（繁體中文）

這個 repo 對應輸送現象(二) 兩份自主學習作業：Week 1 熱傳（金屬棒加熱與冷卻）、Week 2 質傳（乙醇蒸發）。
程式、驗證與完整分析流程都已完成；`results/synthetic/` 裡的圖是用合成數據跑的流程測試（圖上有浮水印），
**不能當成實驗結果繳交**。

## 一、做實驗前

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps
pytest -q                            # 47 個測試全過，代表程式正常
python scripts/design_estimates.py   # 設計估算表 → docs/design_estimates.md
```

設計估算決定了幾件事：不鏽鋼筷子的熱只傳 3 cm 左右，所以感測點集中在水面上 1、2、4、7、12 cm；
開風扇冷卻的時間常數只有 40–70 s，要錄影逐格讀溫度（≤ 5 s 一點）；乙醇液面會冷到 6–11 °C，所以一定要量液溫。

## 二、記錄數據

依 `data/templates/` 的欄位格式存成 CSV，放進 `data/raw/`：

| 實驗 | 檔案欄位 | 注意 |
|---|---|---|
| Week 1 加熱 | `t_s, T_base_C, T1_C … T5_C` | T1…T5 的順序要和設定檔的 `sensor_positions_cm` 一致 |
| Week 1 冷卻 | `t_s, T_C` | 取樣間隔固定（錄影讀值），否則瞬時 h 算不出來 |
| Week 2 Stefan tube | `t_h, mass_g` 或 `t_h, z_mm` | 秤重用 0.01 g 秤；讀液面就用 `z_mm` |
| Week 2 開放皿 | `t_min, mass_g, T_liquid_C` | `mass_g` 是皿加液體，皿重填在設定檔 `tare_g` |

## 三、填設定檔

打開 `config/week1.toml`、`config/week2.toml`，把所有 `"MEASURE"` 換成你的量測值（單位寫在欄位名稱裡）。
還有 `"MEASURE"` 沒填，分析程式會直接停下並列出缺哪些，不會用假值硬跑。

幾個容易填錯的：

- `material`：先做磁鐵測試。304 幾乎不吸、430 會吸，兩者 k 差約 75 %。
- `emissivity`：貼黑膠帶的地方約 0.95，裸不鏽鋼 0.1–0.4；`emissivity_range` 給上下限做不確定度。
- `steady_window_s`：加熱曲線已經平的那段時間。
- `fit_start_s`：冷卻剛夾出、還在擦水膜的前幾秒要跳過。
- `fan_label`：Week 1 和 Week 2 用同一台風扇、同一檔、同距離時填一樣的名字，Week 2 會自動做跨作業的熱質類比驗證。
- `rate_window_min`：開放皿質量曲線前段接近直線的時間範圍。

## 四、跑分析

```bash
python scripts/week1_analysis.py config/week1.toml
python scripts/week2_analysis.py config/week2.toml   # 要先跑 Week 1，才有風速做類比
```

結果在 `results/week1/`、`results/week2/`：每個 run 一張圖、`summary.md`（可直接抄進簡報的表）、
`*_results.json`（所有數字與信賴區間）。

## 五、結果對應簡報哪一頁

| 簡報內容 | 從哪裡來 |
|---|---|
| Fourier 熱傳率、fin 參數 m | `summary.md` Heating 表：`q_base`、`m` |
| h 與 k 分開求、模型形式不確定度 | 兩種 h 模型擬合出的 k 之差 |
| 冷卻時間常數、到室溫時間、h 與相關式比較 | Cooling 表與每個冷卻 run 的右圖 |
| 風速反推 U_eff | forced 冷卻 run 的 `U_eff_m_s` |
| D_AB 與 Fuller、Chapman–Enskog 比較 | Week 2 Stefan 表 |
| k_c、膜厚、蒸發冷卻、完全蒸發時間（含 95 % 範圍） | Week 2 Open dish 表與圖 |
| 75 % 酒精、吸水效應 | Aqueous 表、`water_uptake_max_g` |
| 跨作業熱質類比 | `summary.md` 最後一節 |
| 程式正確性 | `figures/verification/`（放附錄） |

推導與每個假設的理由在 `docs/derivations.md`。
