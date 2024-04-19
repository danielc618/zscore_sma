from datetime import datetime, timezone, timedelta
from http import client
from logging import INFO
from logging.handlers import TimedRotatingFileHandler
from typing import List
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.runtime import Runtime, StrategyTrader
from cybotrade.models import (
    Exchange,
    OrderSide,
    RuntimeConfig,
    RuntimeMode,
    Symbol
)
from cybotrade.permutation import Permutation
import talib
import numpy as np
import asyncio
import logging
import colorlog
import math

class Strategy(BaseStrategy):
    symbol = [Symbol(base="BTC", quote="USDT")]
    sma_length = 50
    z_score_threshold = 0.75
    entry_candle_length = 0
    entry_time = {}
    
    async def set_param(self, identifier, value):
        logging.info(f"Setting{identifier} to {value}")
        if identifier == "sma":
            self.sma_length = float(value)
        elif identifier == "z_score":
            self.z_score_threshold = float(value)
        else:
            logging.info(f"Could not set {identifier}, not found")

    def convert_ms_to_datetime(self, milliseconds):
        seconds = milliseconds / 1000.0
        return datetime.fromtimestamp(seconds)
    
    def get_mean(self, array):
        total = 0
        for i in range(0, len(array)):
            total += array[i]
        return total / len(array)
    
    def get_stddev(self, array):
        total = 0
        mean = self.get_mean(array)
        for i in range(0, len(array)):
            minus_mean = math.pow(array[i] - mean, 2)
            total += minus_mean
        return math.sqrt(total / (len(array)- 1))
    
    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}"))
        file_handler = TimedRotatingFileHandler (filename="z_score_sma.log", when="h",)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    # Check the limit orders and cancel them if the timeframe has passed over 5 candles.
    # async def on_active_order_interval(self, strategy: StrategyTrader, active_orders: List[ActiveOrder]):
    #     if self.entry_candle_length >= 5:
    #         print("Cancel old orders")
    #     # check order id (all the orders will have a special id) status from exchange and if the order placed time + 30 mins is more than the datetime now    
    #     for order in active_orders:
    #         self.entry_time["client_order_id"] == order.client_order_id
    #         and self.entry_time["entry_time"] + timedelta(minutes=30) >= datetime.now():


    async def on_candle_closed(self, strategy, topic, symbol):
        # Retrieve list of candle data for corresponding symbol that candle closed.
        candles = self.data_map[topic]
        # high = np.array(list(map(lambda c: float(c["high"]), candles)))
        # low = np.array(list(map(lambda row: float(c["low"]), candles)))
        # volume = np.array(list(map(lambda c: float(c["volume"]), candles)))
        # Retrieve close data from list of candle data.
        close = []
        start_time = []

        for candle in candles:
            close.append(float(candle["close"]))
            start_time.append(float(candle["start_time"]))

        close = np.array(close) 
        start_time = np.array(start_time)
        
        sma_forty = talib.SMA(close, self.sma_length)
        #logging.info(f"close: {close[-1]}, sma_forty: {sma_forty}")
        # price_changes = (float(close[-1]) / sma_forty[-1] - 1.0)
        std = self.get_stddev(close[-50:])
        z_score = (close[-1] - sma_forty[-1]) / std

        current_pos = await strategy.position(exchange=Exchange.BybitLinear,symbol=symbol)
        logging.info(f"close: {close[-1]}, sma: {sma_forty[-1]}, std: {std}, z_score: {z_score} current_pos: {current_pos} at {self.convert_ms_to_datetime}")

        if z_score >= 2:
            print("Short")
            await strategy.open(exchange=Exchange.BybitLinear,symbol= self.symbol[0], side=OrderSide.Sell, quantity=0.01, is_hedge_mode=False)
        elif z_score <= -2:
            print("Long")
            await strategy.open(exchange=Exchange.BybitLinear,symbol= self.symbol[0], side=OrderSide.Buy, quantity=0.01, is_hedge_mode=False)

        else:
            if current_pos.long.quantity > 0:
                if z_score >= 0:
                    await strategy.close(exchange=Exchange.BybitLinear,symbol= self.symbol[0], side=OrderSide.Buy, quantity=0.01, is_hedge_mode=False)
                    print("Exit")
               
            elif current_pos.short.quantity > 0:
                if z_score <= 0:
                    await strategy.close(exchange=Exchange.BybitLinear,symbol= self.symbol[0], side=OrderSide.Sell, quantity=0.01, is_hedge_mode=False)
                    print("Exit")
               


        new_pos = await strategy.position(exchange=Exchange.BybitLinear, symbol=symbol)
        logging.info(f"new_pos: {new_pos}")


config = RuntimeConfig(
    mode=RuntimeMode.Backtest,
    datasource_topics=[],
    candle_topics=["candles-1m-BTC/USDT-bybit"],
    active_order_interval=1,
    initial_capital=10000.0,
    #exchange keys="./exchange-keys.json",
    start_time=datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc),
    data_count=150,
    api_key="test",
    api_secret="notest",
)

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters["sma"] = [50]
hyper_parameters["z_score"] = [0.75]
# hyper_parameters["sma"] = np.arange(10,60,10)
# hyper_parameters["z_score"] = np.arange(0.7, 0.8, 0.9)

async def start():
    await permutation.run(hyper_parameters, Strategy)

asyncio.run(start())


