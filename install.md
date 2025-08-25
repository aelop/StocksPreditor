# 🎯 AI股票预测系统 - 完整安装指南

## 📋 系统概述

这是一个基于机器学习和技术分析的智能股票预测系统，能够分析美股股票并预测下一小时的价格变化趋势。

### 🔧 技术特性
- **机器学习模型**: 使用随机森林算法进行预测
- **技术指标**: 集成20+种技术分析指标 (RSI, MACD, 布林带等)
- **实时数据**: 通过Yahoo Finance获取实时股票数据
- **概率预测**: 提供涨跌概率和置信区间
- **响应式UI**: 现代化的Web界面

## 🚀 快速开始

### 第一步：环境准备

确保你的系统已安装Python 3.7+:

```bash
python --version
```

### 第二步：安装依赖包

创建项目文件夹并安装所需的Python包：

```bash
# 创建项目目录
mkdir stock-predictor
cd stock-predictor

# 安装依赖包
pip install flask flask-cors yfinance pandas numpy scikit-learn TA-Lib

# 如果TA-Lib安装有问题，可以尝试：
# Windows: 下载whl文件 https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# macOS: brew install ta-lib
# Linux: sudo apt-get install libta-lib-dev
```

### 第三步：创建文件

创建以下文件结构：
```
stock-predictor/
├── app.py (后端API服务器)
├── index.html (前端网页)
└── README.md (说明文档)
```

将前面提供的Python代码保存为 `app.py`，HTML代码保存为 `index.html`。

### 第四步：启动系统

1. **启动后端API服务器**:
```bash
python app.py
```

你应该看到类似输出：
```
股票预测API服务器启动中...
访问 http://localhost:5000 来使用API
API端点:
  POST /api/predict - 预测股票
  POST /api/stock-info - 获取股票信息
  GET /health - 健康检查
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://[::1]:5000
```

2. **打开前端网页**:
双击 `index.html` 文件在浏览器中打开，或者通过浏览器打开该文件。

## 📖 使用方法

### 基本操作

1. **输入股票代码**: 在输入框中输入美股代码，如 `AAPL`, `TSLA`, `MSFT`
2. **点击预测**: 点击"预测下一小时"按钮
3. **查看结果**: 系统会显示预测结果、概率分析和股票信息

### 支持的股票

系统支持所有在Yahoo Finance上可查询的美股股票，包括但不限于：

- **科技股**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META
- **金融股**: JPM, BAC, WFC, GS, MS
- **消费品**: KO, PG, JNJ, WMT, HD
- **ETF**: SPY, QQQ, IWM, VTI

## 🔍 预测原理

### 机器学习模型
- **算法**: 随机森林回归器
- **特征**: 20+种技术指标和价格特征
- **训练数据**: 6个月的历史数据
- **预测目标**: 下一小时价格变化百分比

### 技术指标
1. **趋势指标**: SMA, EMA, MACD
2. **动量指标**: RSI, 随机指标
3. **波动指标**: 布林带, ATR
4. **成交量指标**: 成交量比率分析
5. **价格模式**: 价格相对位置, 高低比率

### 概率计算
- 使用Sigmoid函数转换预测值为概率
- 通过随机森林的多个决策树估计不确定性
- 提供95%置信区间

## ⚙️ API接口文档

### 1. 预测股票价格
```http
POST /api/predict
Content-Type: application/json

{
    "symbol": "AAPL"
}
```

**响应示例**:
```json
{
    "symbol": "AAPL",
    "current_price": 175.43,
    "predicted_change": 0.0123,
    "predicted_price": 177.59,
    "probability_up": 0.687,
    "probability_down": 0.313,
    "confidence_interval": {
        "lower": -0.0089,
        "upper": 0.0335
    },
    "model_performance": {
        "train_score": 0.734,
        "test_score": 0.612
    },
    "prediction_time": "2024-01-15T10:30:00"
}
```

### 2. 获取股票信息
```http
POST /api/stock-info
Content-Type: application/json

{
    "symbol": "AAPL"
}
```

### 3. 健康检查
```http
GET /health
```

## 🛠️ 自定义配置

### 修改预测参数

在 `app.py` 中可以调整以下参数：

```python
# 随机森林模型参数
self.model = RandomForestRegressor(
    n_estimators=100,      # 决策树数量
    random_state=42,       # 随机种子
    max_depth=10,          # 最大深度
    min_samples_split=5    # 最小分割样本数
)

# 数据获取参数
period="6mo"              # 训练数据时间范围
interval="1h"             # 数据时间间隔
```

### 添加新的技术指标

在 `calculate_technical_indicators` 方法中添加新指标：

```python
# 添加新指标
df['new_indicator'] = talib.NEW_FUNCTION(df['Close'])

# 在特征列表中包含新指标
feature_columns.append('new_indicator')
```

## 📊 性能优化建议

### 提高预测准确性
1. **增加训练数据**: 使用更长时间的历史数据
2. **特征工程**: 添加更多相关的技术指标
3. **模型集成**: 结合多种机器学习模型
4. **参数调优**: 使用网格搜索优化模型参数

### 系统性能优化
1. **缓存机制**: 缓存股票数据减少API调用
2. **异步处理**: 使用异步框架如FastAPI
3. **数据库**: 存储历史数据和预测结果
4. **负载均衡**: 多实例部署处理高并发

## 🚨 重要注意事项

### 投资风险警告
- ⚠️ **本系统仅供学习和研究使用**
- ⚠️ **预测结果不构成投资建议**
- ⚠️ **股票投资存在重大风险，可能损失本金**
- ⚠️ **任何投资决策应基于您自己的研究和专业建议**

### 技术限制
- 预测模型基于历史数据，无法预测突发事件
- 市场情绪和基本面因素未完全考虑
- 小时级预测的准确性有限
- 需要稳定的网络连接获取实时数据

### 数据来源
- 使用Yahoo Finance提供的免费数据
- 数据可能存在延迟或不准确
- 建议在交易前验证数据的准确性

## 🔧 故障排除

### 常见问题

**1. TA-Lib安装失败**
```bash
# Windows解决方案
pip install https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp39-cp39-win_amd64.whl

# macOS解决方案
brew install ta-lib
pip install TA-Lib

# Linux解决方案
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

**2. CORS错误**
确保后端API启动并且前端页面通过HTTP访问（而非file://协议）

**3. 数据获取失败**
- 检查网络连接
- 确认股票代码正确
- Yahoo Finance可能暂时不可用

**4. 预测准确率低**
- 增加训练数据量
- 调整模型参数
- 考虑市场波动期间预测困难

## 📈 扩展功能建议

### 功能增强
- [ ] 多时间框架预测（5分钟、15分钟、日线）
- [ ] 批量股票分析
- [ ] 预测结果历史记录
- [ ] 邮件/短信预警功能
- [ ] 投资组合分析
- [ ] 实时数据流处理

### 界面改进
- [ ] 交互式K线图表
- [ ] 技术指标可视化
- [ ] 移动端适配
- [ ] 暗色主题
- [ ] 多语言支持

## 📞 技术支持

如果遇到问题或需要帮助：

1. **检查系统要求**: 确保Python版本和依赖包正确安装
2. **查看日志**: 检查终端输出的错误信息
3. **网络测试**: 确认能够访问Yahoo Finance
4. **版本兼容**: 确保所有依赖包版本兼容

---

**免责声明**: 本系统仅用于教育和研究目的。使用本系统进行任何投资决策的风险完全由用户承担。作者不对任何投资损失负责。