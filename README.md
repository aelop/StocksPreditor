<img width="920" height="637" alt="图片" src="https://github.com/user-attachments/assets/ec94a0cd-781f-4fe5-9da0-69728db2edc4" /># StocksPreditor
By training ai model,to cal the signal

Dependency

# 核心Web框架
Flask==2.3.3
Flask-CORS==4.0.0

# 数据获取和处理
yfinance==0.2.18
pandas==2.0.3
numpy==1.24.3

# 机器学习
scikit-learn==1.3.0

# 技术分析指标
TA-Lib==0.4.26

# 可选：数据可视化和增强功能
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# 可选：性能优化
numba==0.57.1
requests==2.31.0


🎯 系统核心功能
预测能力

机器学习模型: 使用随机森林算法分析股票数据
技术指标集成: 包含20+种技术分析指标（RSI, MACD, 布林带, 移动平均线等）
概率预测: 提供上涨/下跌概率和95%置信区间
实时数据: 通过Yahoo Finance获取最新股票数据

数学方法应用

正态分布: 用于计算置信区间和风险评估
中心极限定理: 通过多个决策树的预测来降低预测方差
K线采样: 使用1小时K线数据进行短期预测
统计学指标: RSI、随机指标、威廉指标等

技术实现

后端: Python Flask API，提供RESTful接口
前端: 现代化HTML5网页应用，响应式设计
数据处理: 集成pandas和numpy进行高效数据处理
可视化: 动态概率条、实时数据显示

# 开发和测试工具
pytest==7.4.0
jupyter==1.0.0


