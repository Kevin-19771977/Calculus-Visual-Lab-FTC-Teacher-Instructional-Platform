# FTC Teaching Tool Web Clear

這是一個使用 Streamlit 製作的「微積分基本定理（FTC）」互動教學工具，適合放在 GitHub 並部署到 Streamlit Community Cloud，讓學生直接在網頁上操作。

## 功能特色

- 模組 1：累積函數動態生成
- 模組 2：導數與累積同步
- 模組 3：變上限積分符號辨識
- 模組 4：FTC Part 2 幾何意義
- 清楚版介面，適合學生操作與課堂展示

## 在本機執行

請先安裝套件：

```bash
pip install -r requirements.txt
```

然後執行：

```bash
python -m streamlit run app.py
```

## 上傳到 GitHub

請把這個資料夾中的所有檔案一起上傳到 GitHub repository 根目錄：

- app.py
- requirements.txt
- README.md
- .gitignore

## 部署到 Streamlit Community Cloud

1. 登入 Streamlit Community Cloud
2. 選擇你的 GitHub repository
3. Main file path 設定為 `app.py`
4. 按下 Deploy

## 適用場景

- 大學微積分課堂教學
- 學生個別操作學習
- 微積分基本定理概念研究實驗
- 教學展示與試教
