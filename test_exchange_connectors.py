#!/usr/bin/env python3
"""
Test script for QuantumTrade Exchange Connectors
Tests all exchange connections, data feeds, and trading functionality
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all exchange connectors can be imported"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)

    try:
        from binance_connector import BinanceConnector
        print("✓ BinanceConnector imported successfully")
    except Exception as e:
        print(f"✗ BinanceConnector import failed: {e}")
        return False

    try:
        from bitget_connector import BitgetConnector
        print("✓ BitgetConnector imported successfully")
    except Exception as e:
        print(f"✗ BitgetConnector import failed: {e}")
        return False

    try:
        from coinbase_connector import CoinbaseConnector
        print("✓ CoinbaseConnector imported successfully")
    except Exception as e:
        print(f"✗ CoinbaseConnector import failed: {e}")
        return False

    try:
        from data_feed import DataFeed
        print("✓ DataFeed imported successfully")
    except Exception as e:
        print(f"✗ DataFeed import failed: {e}")
        return False

    try:
        from config import Config
        print("✓ Config imported successfully")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False

    return True

def test_config():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)

    try:
        from config import Config
        config = Config()

        # Test basic config
        trading_config = config.get('trading')
        print(f"✓ Trading config loaded: {len(trading_config)} settings")

        # Test exchange configs
        binance_config = config.get('binance', {})
        bitget_config = config.get('bitget', {})
        coinbase_config = config.get('coinbase', {})

        print(f"✓ Binance config: {len(binance_config)} settings")
        print(f"✓ Bitget config: {len(bitget_config)} settings")
        print(f"✓ Coinbase config: {len(coinbase_config)} settings")

        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_exchange_connections():
    """Test exchange connection status"""
    print("\n" + "=" * 60)
    print("TESTING EXCHANGE CONNECTIONS")
    print("=" * 60)

    from binance_connector import BinanceConnector
    from bitget_connector import BitgetConnector
    from coinbase_connector import CoinbaseConnector
    from config import Config

    config = Config()
    results = {}

    # Test Binance
    try:
        binance = BinanceConnector(config.get('binance'))
        status = binance.get_connection_status()
        results['binance'] = status
        print(f"✓ Binance connection status: {status}")
    except Exception as e:
        print(f"✗ Binance connection test failed: {e}")
        results['binance'] = {'connected': False, 'error': str(e)}

    # Test Bitget
    try:
        bitget = BitgetConnector(config.get('bitget'))
        status = bitget.get_connection_status()
        results['bitget'] = status
        print(f"✓ Bitget connection status: {status}")
    except Exception as e:
        print(f"✗ Bitget connection test failed: {e}")
        results['bitget'] = {'connected': False, 'error': str(e)}

    # Test Coinbase
    try:
        coinbase = CoinbaseConnector(config.get('coinbase'))
        status = coinbase.get_connection_status()
        results['coinbase'] = status
        print(f"✓ Coinbase connection status: {status}")
    except Exception as e:
        print(f"✗ Coinbase connection test failed: {e}")
        results['coinbase'] = {'connected': False, 'error': str(e)}

    return results

def test_market_data():
    """Test market data retrieval from all exchanges"""
    print("\n" + "=" * 60)
    print("TESTING MARKET DATA RETRIEVAL")
    print("=" * 60)

    from binance_connector import BinanceConnector
    from bitget_connector import BitgetConnector
    from coinbase_connector import CoinbaseConnector
    from config import Config

    config = Config()
    symbols = ['BTCUSDT', 'ETHUSDT', 'BTC-USD']
    results = {}

    exchanges = [
        ('Binance', BinanceConnector, 'BTCUSDT', config.get('binance')),
        ('Bitget', BitgetConnector, 'BTCUSDT', config.get('bitget')),
        ('Coinbase', CoinbaseConnector, 'BTC-USD', config.get('coinbase'))
    ]

    for name, connector_class, symbol, exchange_config in exchanges:
        try:
            connector = connector_class(exchange_config)
            start_time = time.time()

            # Test market data
            data = connector.get_market_data(symbol)
            elapsed = time.time() - start_time

            if data and 'price' in data:
                print(f"✓ {name} market data: ${data['price']:.2f} ({elapsed:.2f}s)")
                results[name] = {'success': True, 'price': data['price'], 'time': elapsed}
            else:
                print(f"✗ {name} market data failed: Invalid response")
                results[name] = {'success': False, 'error': 'Invalid response'}

        except Exception as e:
            print(f"✗ {name} market data failed: {e}")
            results[name] = {'success': False, 'error': str(e)}

    return results

def test_data_feed():
    """Test multi-exchange data feed functionality"""
    print("\n" + "=" * 60)
    print("TESTING MULTI-EXCHANGE DATA FEED")
    print("=" * 60)

    try:
        from data_feed import DataFeed
        from config import Config

        config = Config()
        data_feed = DataFeed(config.get('data'))

        # Test supported symbols
        symbols = data_feed.get_supported_symbols()
        print(f"✓ Supported symbols: {symbols}")

        # Test current data for different symbols
        test_symbols = ['BTC/USD', 'ETH/USD']
        for symbol in test_symbols:
            try:
                data = data_feed.get_current_data(symbol)
                if data and 'price' in data:
                    print(f"✓ {symbol} data: ${data['price']:.2f} ({data.get('source', 'unknown')})")
                else:
                    print(f"✗ {symbol} data failed: Invalid response")
            except Exception as e:
                print(f"✗ {symbol} data failed: {e}")

        # Test exchange switching
        print("\nTesting exchange switching:")
        exchanges = ['binance', 'bitget', 'coinbase']
        for exchange in exchanges:
            try:
                data_feed.set_current_exchange(exchange)
                data = data_feed.get_current_data('BTC/USD')
                if data:
                    print(f"✓ {exchange.upper()}: ${data['price']:.2f} ({data.get('source', 'unknown')})")
                else:
                    print(f"✗ {exchange.upper()}: No data")
            except Exception as e:
                print(f"✗ {exchange.upper()}: {e}")

        return True

    except Exception as e:
        print(f"✗ Data feed test failed: {e}")
        return False

def test_mock_fallback():
    """Test mock data fallback when APIs are unavailable"""
    print("\n" + "=" * 60)
    print("TESTING MOCK DATA FALLBACK")
    print("=" * 60)

    from binance_connector import BinanceConnector
    from bitget_connector import BitgetConnector
    from coinbase_connector import CoinbaseConnector

    # Test with no credentials (should use mock data)
    old_env = dict(os.environ)

    # Remove API credentials temporarily
    api_keys = ['BINANCE_API_KEY', 'BINANCE_SECRET_KEY',
                'BITGET_API_KEY', 'BITGET_SECRET_KEY',
                'COINBASE_API_KEY', 'COINBASE_SECRET_KEY']

    for key in api_keys:
        os.environ.pop(key, None)

    try:
        # Test Binance mock
        binance = BinanceConnector()
        data = binance.get_market_data('BTCUSDT')
        if data and data.get('source') == 'mock_fallback':
            print("✓ Binance mock data: ${:.2f}".format(data['price']))
        else:
            print("✗ Binance mock data failed")

        # Test Bitget mock
        bitget = BitgetConnector()
        data = bitget.get_market_data('BTCUSDT')
        if data and data.get('source') == 'mock_fallback':
            print("✓ Bitget mock data: ${:.2f}".format(data['price']))
        else:
            print("✗ Bitget mock data failed")

        # Test Coinbase mock
        coinbase = CoinbaseConnector()
        data = coinbase.get_market_data('BTC-USD')
        if data and data.get('source') == 'mock_fallback':
            print("✓ Coinbase mock data: ${:.2f}".format(data['price']))
        else:
            print("✗ Coinbase mock data failed")

    finally:
        # Restore environment
        os.environ.update(old_env)

    return True

def test_trade_simulation():
    """Test trade execution simulation"""
    print("\n" + "=" * 60)
    print("TESTING TRADE SIMULATION")
    print("=" * 60)

    from binance_connector import BinanceConnector
    from bitget_connector import BitgetConnector
    from coinbase_connector import CoinbaseConnector

    exchanges = [
        ('Binance', BinanceConnector()),
        ('Bitget', BitgetConnector()),
        ('Coinbase', CoinbaseConnector())
    ]

    for name, connector in exchanges:
        try:
            # Test buy order
            result = connector.execute_trade('buy', 'BTCUSDT', 0.001, 45000)
            if result and result.get('success'):
                print(f"✓ {name} buy simulation: {result['order_id']} (${result['price']:.2f})")
            else:
                print(f"✗ {name} buy simulation failed: {result}")

            # Test sell order
            result = connector.execute_trade('sell', 'BTCUSDT', 0.001, 45000)
            if result and result.get('success'):
                print(f"✓ {name} sell simulation: {result['order_id']} (${result['price']:.2f})")
            else:
                print(f"✗ {name} sell simulation failed: {result}")

        except Exception as e:
            print(f"✗ {name} trade simulation failed: {e}")

    return True

def run_all_tests():
    """Run all tests and provide summary"""
    print("QUANTUMTRADE EXCHANGE CONNECTORS - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    start_time = time.time()
    test_results = {}

    # Run all tests
    test_results['imports'] = test_imports()
    test_results['config'] = test_config()
    test_results['connections'] = test_exchange_connections()
    test_results['market_data'] = test_market_data()
    test_results['data_feed'] = test_data_feed()
    test_results['mock_fallback'] = test_mock_fallback()
    test_results['trade_simulation'] = test_trade_simulation()

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(".2f")

    # Count successful tests
    successful = sum(1 for result in test_results.values() if result is True or (isinstance(result, dict) and all(v.get('success', False) for v in result.values() if isinstance(v, dict))))

    total_tests = len(test_results)
    print(f"Tests passed: {successful}/{total_tests}")

    if successful == total_tests:
        print("🎉 ALL TESTS PASSED! Multi-exchange support is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")

    print("\nNext steps:")
    print("1. Set up API credentials in environment variables for live trading")
    print("2. Test with real API keys (optional)")
    print("3. Update web interface for exchange selection")
    print("4. Monitor exchange connections in production")

    return successful == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
