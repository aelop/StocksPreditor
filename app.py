from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import talib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


class StockPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def fetch_stock_data(self, symbol, period="3mo"):
        """获取股票数据"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval="1h")
            if data.empty:
                return None
            return data
        except Exception as e:
            print(f"获取数据错误: {e}")
            return None

    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        df = data.copy()

        # 基础价格指标
        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['volume_change'] = df['Volume'].pct_change()

        # 移动平均线
        df['sma_5'] = talib.SMA(df['Close'], 5)
        df['sma_20'] = talib.SMA(df['Close'], 20)
        df['ema_12'] = talib.EMA(df['Close'], 12)
        df['ema_26'] = talib.EMA(df['Close'], 26)

        # MACD
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['Close'])

        # RSI
        df['rsi'] = talib.RSI(df['Close'], 14)

        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['Close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 随机指标
        df['slowk'], df['slowd'] = talib.STOCH(df['High'], df['Low'], df['Close'])

        # ATR (平均真实波幅)
        df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'])

        # 威廉指标
        df['willr'] = talib.WILLR(df['High'], df['Low'], df['Close'])

        # 价格相对位置
        df['price_position'] = (df['Close'] - df['Low'].rolling(20).min()) / (
                    df['High'].rolling(20).max() - df['Low'].rolling(20).min())

        # 成交量指标
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']

        return df

    def prepare_features(self, data):
        """准备特征数据"""
        df = self.calculate_technical_indicators(data)

        # 选择特征列
        feature_columns = [
            'price_change', 'high_low_ratio', 'volume_change',
            'sma_5', 'sma_20', 'ema_12', 'ema_26',
            'macd', 'macdsignal', 'macdhist',
            'rsi', 'bb_width', 'bb_position',
            'slowk', 'slowd', 'atr', 'willr',
            'price_position', 'volume_ratio'
        ]

        # 添加滞后特征
        for col in ['Close', 'Volume', 'rsi', 'macd']:
            for lag in [1, 2, 3]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                feature_columns.append(f'{col}_lag_{lag}')

        # 添加时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        feature_columns.extend(['hour', 'day_of_week'])

        # 创建目标变量 (下一小时的价格变化)
        df['target'] = df['Close'].shift(-1) / df['Close'] - 1

        return df, feature_columns

    def train_model(self, symbol):
        """训练预测模型"""
        # 获取更多历史数据进行训练
        data = self.fetch_stock_data(symbol, period="6mo")
        if data is None:
            return False, "无法获取股票数据"

        df, feature_columns = self.prepare_features(data)

        # 移除缺失值
        df = df.dropna()

        if len(df) < 50:
            return False, "数据量不足，无法训练模型"

        X = df[feature_columns]
        y = df['target']

        # 分割训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练模型
        self.model.fit(X_train_scaled, y_train)

        # 评估模型
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        return True, {
            'train_score': train_score,
            'test_score': test_score,
            'feature_columns': feature_columns
        }

    def predict_next_hour(self, symbol):
        """预测下一小时价格变化"""
        # 获取最新数据
        data = self.fetch_stock_data(symbol, period="3mo")
        if data is None:
            return None, "无法获取股票数据"

        # 训练模型
        success, result = self.train_model(symbol)
        if not success:
            return None, result

        df, feature_columns = self.prepare_features(data)
        df = df.dropna()

        if len(df) == 0:
            return None, "数据处理后为空"

        # 获取最新的特征数据
        latest_features = df[feature_columns].iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)

        # 进行预测
        prediction = self.model.predict(latest_features_scaled)[0]

        # 计算预测概率和置信区间
        # 使用随机森林的多个决策树来估计不确定性
        tree_predictions = []
        for tree in self.model.estimators_:
            tree_pred = tree.predict(latest_features_scaled)[0]
            tree_predictions.append(tree_pred)

        pred_std = np.std(tree_predictions)
        confidence_interval = {
            'lower': prediction - 1.96 * pred_std,
            'upper': prediction + 1.96 * pred_std
        }

        # 转换为涨跌概率
        prob_up = 1 / (1 + np.exp(-prediction * 10))  # sigmoid转换
        prob_down = 1 - prob_up

        current_price = data['Close'].iloc[-1]

        return {
            'symbol': symbol,
            'current_price': float(current_price),
            'predicted_change': float(prediction),
            'predicted_price': float(current_price * (1 + prediction)),
            'probability_up': float(prob_up),
            'probability_down': float(prob_down),
            'confidence_interval': {
                'lower': float(confidence_interval['lower']),
                'upper': float(confidence_interval['upper'])
            },
            'model_performance': {
                'train_score': result['train_score'],
                'test_score': result['test_score']
            },
            'prediction_time': datetime.now().isoformat()
        }, None


# 创建预测器实例
predictor = StockPredictor()


@app.route('/api/predict', methods=['POST'])
def predict_stock():
    """预测股票API"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()

        if not symbol:
            return jsonify({'error': '请提供股票代码'}), 400

        # 确保股票代码格式正确
        if not symbol.endswith(('.US', '.HK')) and len(symbol) <= 5:
            # 对于美股，不需要后缀
            pass

        result, error = predictor.predict_next_hour(symbol)

        if error:
            return jsonify({'error': error}), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'预测出错: {str(e)}'}), 500


@app.route('/api/stock-info', methods=['POST'])
def get_stock_info():
    """获取股票基本信息"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()

        if not symbol:
            return jsonify({'error': '请提供股票代码'}), 400

        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1d", interval="1h")

        if hist.empty:
            return jsonify({'error': '无法获取股票数据'}), 404

        latest_data = hist.iloc[-1]

        return jsonify({
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'current_price': float(latest_data['Close']),
            'volume': int(latest_data['Volume']),
            'high_24h': float(hist['High'].max()),
            'low_24h': float(hist['Low'].min()),
            'sector': info.get('sector', '未知'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', None)
        })

    except Exception as e:
        return jsonify({'error': f'获取股票信息出错: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    print("股票预测API服务器启动中...")
    print("访问 http://localhost:5000 来使用API")
    print("API端点:")
    print("  POST /api/predict - 预测股票")
    print("  POST /api/stock-info - 获取股票信息")
    print("  GET /health - 健康检查")
    app.run(debug=True, host='0.0.0.0', port=5000)