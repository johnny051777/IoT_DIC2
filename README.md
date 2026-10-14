# CRISP-DM Regression Studio (50 Startups & Boston Housing)

🌐 **Live Demo**: [https://iot-dic2-regression.vercel.app](https://iot-dic2-regression.vercel.app)
📦 **GitHub**: [https://github.com/johnny051777/IoT_DIC2](https://github.com/johnny051777/IoT_DIC2)

多元線性迴歸 × 5 種特徵選擇演算法互動式分析平台 (基於 CRISP-DM 資料探勘標準流程)。

## 專案亮點

- **下拉選單呈現兩種經典問題**：
  - 💼 **50 Startups (Kaggle)**：預測新創公司淨利潤 (Profit)，自動將 `State` 類別欄位進行 One-Hot 編碼。
  - 🏡 **Boston Housing (Kaggle)**：預測波士頓房價中位數 (MEDV)，評估 13 項城市與經濟特徵。
- **支援任意輸入自訂 Dataset**：
  - 支援輸入公開 CSV 網址（例如：`https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`）。
  - 支援直接拖曳 / 上傳本機 CSV 檔案。
  - 自動偵測特徵型態、中位數缺失值填補、標準化縮放。
- **5 種特徵選擇機制評估 (Occam's Razor 簡約原則)**：
  1. **Lasso (1-SE 規則)**：L1 正則化與係數路徑收縮。
  2. **Forward SFS (循序前向選擇)**：逐步搜尋最佳 R² 特徵組合。
  3. **RFE (遞迴特徵消除 / RFECV)**：以 5 折交叉驗證決定最適維度。
  4. **ElasticNet 係數排序**：結合 L1/L2 正則化並依 1% 誤差簡約原則截斷。
  5. **Exhaustive Search (全子集窮舉搜尋)**：搜尋 $2^p - 1$ 所有可能特徵子集，找出全域最佳 CV-RMSE 組合。
  6. **All Features Baseline**：完整特徵基準對照。
- **即時預測模擬器 (What-If Playground)**：
  - 依選定之特徵選擇模型動態生成特徵滑桿與輸入框。
  - 調整輸入即可即時計算預測結果並顯示迴歸計算式。

---

## 本地開發與測試

```bash
# 1. 安裝 Python 相依套件
pip install -r requirements.txt

# 2. 啟動本機測試伺服器
python dev_server.py
```

開啟瀏覽器前往：`http://localhost:8000`

---

## CRISP-DM 六大階段對應

1. **Business Understanding (商業理解)**：確定預測標的（50 Startups 之淨利潤、Boston Housing 之房價）。
2. **Data Understanding (資料理解)**：檢視前 8 筆原始資料、欄位型態、統計分佈與特徵相關係數熱圖 (Correlation Matrix)。
3. **Data Preparation (資料準備)**：自動進行數值清洗、中位數補齊、類別變數 One-Hot Encoding，以及 80/20 訓練/測試切分。
4. **Modeling (模型建構)**：執行 5 種特徵篩選演算法，繪製 Lasso 係數路徑圖與全子集維度評估散布圖。
5. **Evaluation (模型評估)**：綜合比較 Train/Test/5-Fold CV 之 RMSE、R² 與 Adjusted R²，產出 Leaderboard 排行榜與殘差分佈圖。
6. **Deployment (模型部署)**：以 Serverless 架構部署於 Vercel，提供即時互動預測模擬器。

---

## 部署至 Vercel

本專案支援一鍵部署至 Vercel：
- 前端：靜態 `index.html`（含 Plotly.js 互動視覺化）。
- 後端：`api/analyze.py`（Vercel Python Serverless Function）。

```bash
# 登入並發布至 Vercel
npx vercel --prod
```
