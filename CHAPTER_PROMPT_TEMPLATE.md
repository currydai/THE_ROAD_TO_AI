# 章节生成提示词模板（复制后替换花括号内容）

本模板用于在仓库中为“监督学习”或其他模块新增一个完整章节，包含：
- 中/英两份 LaTeX 教程（.tex）
- 带注释的 Python 脚本（.py），生成本章配图
- figures 目录下的图片（.png）
- 编译后的 PDF（.pdf）

仅提交上述文件类型；LaTeX 中间产物（.aux/.log/.out/.synctex.gz 等）由 .gitignore 忽略。

---

## 1) 章节路径与命名
- 章节目录：`0_Machine Learning/0_Supervised Learning/{Chapter_Index}_{Chapter_Slug}/`
  - `{Chapter_Index}`：章节序号（整数，例如 2、3）
  - `{Chapter_Slug}`：英文短名（例如 `Support Vector Regression`）
- 文件命名（统一风格）：
  - 英文 LaTeX：`{slug}_tutorial_english.tex`
  - 中文 LaTeX：`{slug}_tutorial_chinese.tex`
  - 生成配图脚本：`gen_{slug}_figures.py`
  - 图片目录：`figures/`（脚本输出必须保存到该目录）
  - PDF：与 .tex 同名的 .pdf（编译产物）

> 注：`{slug}` 建议用 `{Chapter_Slug}` 的下划线/小写变体，例如 `support_vector_regression`。

---

## 2) LaTeX 统一要求（中英文各一份）
- 编译工具：XeLaTeX 或 LuaLaTeX
  - 中文：`ctexart` 文档类；英文：`article` 文档类
- 常用宏包（保持一致）：
  - `geometry, amsmath, amssymb, amsthm, bm, hyperref, graphicx, caption, listings, xcolor, float, placeins`
  - `\graphicspath{{figures/}}`
- 代码样式（统一）：
  ```tex
  % Code style
  \lstdefinestyle{code}{
    basicstyle=\ttfamily\small,
    numbers=left,
    numberstyle=\tiny,
    numbersep=8pt,
    keywordstyle=\color{blue},
    commentstyle=\color{teal!70!black},
    stringstyle=\color{orange!70!black},
    showstringspaces=false,
    breaklines=true,
    frame=single,
    framerule=0.3pt,
    rulecolor=\color{black!15}
  }
  \lstset{style=code}
  ```
- 章节结构（需中英一致，仅语言不同）：
  1. Introduction / 引言（动机、优缺点、适用场景）
  2. Theory and Formulas / 原理与公式（模型、目标函数、推导要点）
     - 模型/记号；损失与正则；闭式解或优化方法（坐标下降/梯度法/KKT/数值稳定性建议）
     - 标准化与截距处理（不惩罚截距，居中/标准化的注意事项）
  3. Applications and Tips / 应用场景与要点（多重共线性、特征选择、超参/CV、预处理）
  4. Python Practice / Python 实战（说明脚本功能与输出位置）
     - 用 `lstlisting` 展示与 .py 一致的核心代码片段
  5. Result / 运行效果（插图、图例、`\ref` 引用）
  6. Summary / 小结（择优建议与工程注意事项）
- 插图规范：
  - `\caption` 后紧接 `\label`；正文用 `\ref{fig:...}` 引用
  - 多图后加 `\FloatBarrier`，必要时 `\begin{figure}[H]` 固定位置

---

## 3) Python 脚本统一要求
- 文件：`gen_{slug}_figures.py`
- 顶部添加简要 docstring，逐步注释关键操作
- 生成可复现的合成数据或读入公开数据；不使用需联网的依赖
- 输出图片路径（确保被 LaTeX `\graphicspath` 找到）：
  ```python
  fig_dir = os.path.join(
      "0_Machine Learning", "0_Supervised Learning",
      f"{Chapter_Index}_{Chapter_Slug}", "figures"
  )
  os.makedirs(fig_dir, exist_ok=True)
  out_path = os.path.join(fig_dir, "{figure_file}.png")
  ```
- 库兼容性：避免使用旧版不支持的参数（如 `lasso_path` 不带 `fit_intercept`）；必要时在注释中说明
- 图例适度：可仅显示部分图例以避免拥挤

---

## 4) 本章信息占位符（请替换）
- 章节：`{Chapter_Index}_{Chapter_Slug}`
- 中文标题：`{Chinese_Title}`
- 英文标题：`{English_Title}`
- 关键算法/概念：`{Key_Algorithms}`（示例：Ridge/Lasso/Elastic Net 或 SVR/核方法 等）
- 推导/公式要点：`{Derivation_Points}`（示例：目标函数、KKT、闭式解、软阈值/坐标下降、数值稳定性）
- 实战图列表：`{Figure_List}`（每幅图名称与含义，例如：`coef_path.png`：系数路径）
- 脚本目标：`{Script_Goals}`（例如：生成系数路径、学习曲线、决策边界对比等）

---

## 5) 提示词模板（粘贴到对话里使用）
```
请在仓库路径 `0_Machine Learning/0_Supervised Learning/{Chapter_Index}_{Chapter_Slug}/` 下创建新章节，满足以下要求：

- 生成两份 LaTeX：
  - 英文：`{slug}_tutorial_english.tex`
  - 中文：`{slug}_tutorial_chinese.tex`
  - 统一导言：geometry, amsmath, amssymb, amsthm, bm, hyperref, graphicx, caption, listings, xcolor, float, placeins；以及统一 `lstdefinestyle{code}`；`\graphicspath{{figures/}}`
  - 结构：Introduction / Theory and Formulas / Applications and Tips / Python Practice / Result / Summary
  - 插图：`\caption` + `\label`；必要时 `[H]` 与 `\FloatBarrier`

- 生成 Python 脚本：`gen_{slug}_figures.py`，带完整注释
  - 生成 `{Figure_List}`，保存到 `figures/`
  - 注意兼容性，避免使用旧版库未支持的参数

- 仅提交：.py、.tex、.pdf 与 figures 下的图片

- 本章主题：
  - 中文标题：{Chinese_Title}
  - 英文标题：{English_Title}
  - 关键算法：{Key_Algorithms}
  - 推导要点：{Derivation_Points}
  - 实战脚本目标：{Script_Goals}