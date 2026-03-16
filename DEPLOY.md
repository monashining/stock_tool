# 股票工具 - 上線部署指南

## 方式一：Streamlit Community Cloud（推薦，免費）

### 前置條件
1. 將專案推送到 **GitHub** 公開或私有倉庫
2. 擁有 [share.streamlit.io](https://share.streamlit.io) 帳號（可用 GitHub 登入）

### 部署步驟

1. **登入** [share.streamlit.io](https://share.streamlit.io)，點右上角 **「Create app」**

2. **選擇「Yup, I have an app」**，填寫：
   - **Repository**：`你的用戶名/stock_tool`（或你的倉庫路徑）
   - **Branch**：`main` 或 `master`
   - **Main file path**：`app.py`
   - **App URL**（選填）：例如 `stock-tool`，網址會是 `https://stock-tool.streamlit.app`

3. **進階設定（Advanced settings）**
   - **Python version**：建議 3.11 或 3.12
   - **Secrets**：貼上以下內容（將 `your_token` 換成實際的 LINE Channel Access Token）：

   ```toml
   LINE_CHANNEL_ACCESS_TOKEN = "your_line_channel_access_token_here"
   ```

4. 點 **「Deploy」**，等待幾分鐘完成部署

5. 部署完成後，你的 App 網址為：`https://你的子網域.streamlit.app`

### 注意事項
- LINE 推播為選用功能，未設定 `LINE_CHANNEL_ACCESS_TOKEN` 時，推播按鈕會顯示「缺少 LINE 環境變數」
- 若使用 FinMind，需確認其 API 在雲端環境可正常連線
- 首次部署可能需 5–10 分鐘安裝依賴

---

## 方式二：本地或自架伺服器

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 設定環境變數
複製 `.env.example` 為 `.env`，並填入 `LINE_CHANNEL_ACCESS_TOKEN`（選填）

### 3. 啟動 App
```bash
streamlit run app.py
```

預設會在 `http://localhost:8501` 開啟。

### 4. 背景執行（Linux/Mac）
```bash
nohup streamlit run app.py --server.port 8501 &
```

---

## 方式三：Docker（自架或雲端 VM）

建立 `Dockerfile`：

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

建置與執行：
```bash
docker build -t stock-tool .
docker run -p 8501:8501 -e LINE_CHANNEL_ACCESS_TOKEN=你的token stock-tool
```

---

## 疑難排解

| 問題 | 可能原因 | 解法 |
|------|----------|------|
| 部署失敗 | 依賴衝突 | 檢查 `requirements.txt` 版本，必要時鎖定版本 |
| 資料抓不到 | 網路限制 | FinMind / yfinance 需可連外網 |
| LINE 推播失敗 | Token 錯誤或過期 | 到 LINE Developers 確認 Channel Access Token |
| 記憶體不足 | 資料量過大 | 考慮減少快取 TTL 或分批載入 |
