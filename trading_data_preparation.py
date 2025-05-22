import pandas as pd
import numpy as np # Needed for np.nan
from collections import deque # For Equal Highs/Lows lookback

# --- Configuration ---
ohlc_file_path = 'EURUSD_OHLC.csv'
news_file_path = 'news_calendar.csv' # Corrected from new_calendarCSV
news_column_names = ['Timestamp (UTC)', 'Date', 'Time (Local)', 'Currency', 'Impact', 'Event', 'Actual', 'Forecast', 'Previous']
big_red_flag_events = ['CPI', 'Non-Farm Payrolls', 'Interest Rate Decision', 'FOMC', 'ECB Press Conference', 'Fed Chair Powell Speaks'] # Added some common ones
SWING_N = 2 # Number of candles on each side for swing point identification
LIQUIDITY_LOOKBACK = 20 # For Equal Highs/Lows detection
PRICE_TOLERANCE_PIPS = 5 # For Equal Highs/Lows
PIP_VALUE = 0.0001 # For EURUSD, adjust if necessary
SL_BUFFER_PIPS = 5 # Stop loss buffer in pips
MIN_RR_RATIO = 1.5 # Minimum Risk/Reward ratio for a trade

# --- Swing Point Identification Functions ---
def identify_swing_high(df, N=2):
    """
    Identifies Swing Highs in OHLC data.
    A candle is a Swing High if its High is >= High of N preceding candles AND >= High of N succeeding candles.
    """
    df_copy = df.copy()
    df_copy['is_swing_high'] = True # Assume True, then filter out

    for i in range(1, N + 1):
        df_copy['is_swing_high'] &= (df_copy['High'] >= df_copy['High'].shift(i))
        df_copy['is_swing_high'] &= (df_copy['High'] >= df_copy['High'].shift(-i))
    
    df_copy['is_swing_high'] = df_copy['is_swing_high'].fillna(False)
    return df_copy

def identify_swing_low(df, N=2):
    """
    Identifies Swing Lows in OHLC data.
    A candle is a Swing Low if its Low is <= Low of N preceding candles AND <= Low of N succeeding candles.
    """
    df_copy = df.copy()
    df_copy['is_swing_low'] = True # Assume True, then filter out

    for i in range(1, N + 1):
        df_copy['is_swing_low'] &= (df_copy['Low'] <= df_copy['Low'].shift(i))
        df_copy['is_swing_low'] &= (df_copy['Low'] <= df_copy['Low'].shift(-i))
        
    df_copy['is_swing_low'] = df_copy['is_swing_low'].fillna(False)
    return df_copy

# --- Market Structure Detection Function ---
def detect_market_structure(df):
    df_out = df.copy()
    df_out['bos_bullish'] = False
    df_out['bos_bearish'] = False
    df_out['choch_bullish'] = False
    df_out['choch_bearish'] = False

    # Structural points for uptrend
    uptrend_HH_price = np.nan
    uptrend_HL_price = np.nan
    # Structural points for downtrend
    downtrend_LH_price = np.nan
    downtrend_LL_price = np.nan

    # Potentials for next structural points
    potential_HL_price = np.nan # Last SL price that could become an HL
    potential_HL_idx = -1
    potential_LH_price = np.nan # Last SH price that could become an LH
    potential_LH_idx = -1
    
    trend = "undetermined"

    for i in range(len(df_out)):
        idx = df_out.index[i]
        row = df_out.iloc[i] # Use iloc for positional access

        is_sh = row['is_swing_high']
        sh_price = row['High'] if is_sh else np.nan
        is_sl = row['is_swing_low']
        sl_price = row['Low'] if is_sl else np.nan
        close_price = row['Close']

        if trend == "undetermined":
            if is_sl:
                potential_HL_price = sl_price
                potential_HL_idx = i # Using integer index for sequence check
            if is_sh:
                potential_LH_price = sh_price
                potential_LH_idx = i

            if not pd.isna(potential_HL_price) and not pd.isna(potential_LH_price):
                if potential_LH_idx > potential_HL_idx: # SH after SL
                    if potential_LH_price > potential_HL_price: # SH is higher
                        trend = "uptrend"
                        uptrend_HH_price = potential_LH_price
                        uptrend_HL_price = potential_HL_price
                        potential_HL_price, potential_LH_price = np.nan, np.nan
                        potential_HL_idx, potential_LH_idx = -1, -1
                elif potential_HL_idx > potential_LH_idx: # SL after SH
                    if potential_HL_price < potential_LH_price: # SL is lower
                        trend = "downtrend"
                        downtrend_LH_price = potential_LH_price
                        downtrend_LL_price = potential_HL_price
                        potential_HL_price, potential_LH_price = np.nan, np.nan
                        potential_HL_idx, potential_LH_idx = -1, -1
        
        elif trend == "uptrend":
            if is_sl:
                potential_HL_price = sl_price # Record latest SL as potential next HL
                # Check for Bearish CHoCH
                if not pd.isna(uptrend_HL_price) and close_price < uptrend_HL_price:
                    df_out.loc[idx, 'choch_bearish'] = True
                    trend = "downtrend"
                    downtrend_LH_price = uptrend_HH_price
                    downtrend_LL_price = sl_price # This SL is the new LL
                    uptrend_HH_price, uptrend_HL_price, potential_HL_price, potential_LH_price = np.nan, np.nan, np.nan, np.nan
                    potential_HL_idx, potential_LH_idx = -1, -1
            
            elif is_sh: # Only evaluate SH if not an SL that caused CHoCH
                if not pd.isna(uptrend_HH_price) and sh_price > uptrend_HH_price:
                    # Potential Bullish BOS
                    if close_price > uptrend_HH_price:
                        df_out.loc[idx, 'bos_bullish'] = True
                    # Confirm new structure
                    if not pd.isna(potential_HL_price): # Lock in the HL that preceded this new HH
                        uptrend_HL_price = potential_HL_price
                    uptrend_HH_price = sh_price
                    potential_HL_price = np.nan # Reset for next leg
                    potential_HL_idx = -1

        elif trend == "downtrend":
            if is_sh:
                potential_LH_price = sh_price # Record latest SH as potential next LH
                # Check for Bullish CHoCH
                if not pd.isna(downtrend_LH_price) and close_price > downtrend_LH_price:
                    df_out.loc[idx, 'choch_bullish'] = True
                    trend = "uptrend"
                    uptrend_HL_price = downtrend_LL_price
                    uptrend_HH_price = sh_price # This SH is the new HH
                    downtrend_LH_price, downtrend_LL_price, potential_HL_price, potential_LH_price = np.nan, np.nan, np.nan, np.nan
                    potential_HL_idx, potential_LH_idx = -1, -1
            
            elif is_sl: # Only evaluate SL if not an SH that caused CHoCH
                if not pd.isna(downtrend_LL_price) and sl_price < downtrend_LL_price:
                    # Potential Bearish BOS
                    if close_price < downtrend_LL_price:
                        df_out.loc[idx, 'bos_bearish'] = True
                    # Confirm new structure
                    if not pd.isna(potential_LH_price): # Lock in the LH that preceded this new LL
                        downtrend_LH_price = potential_LH_price
                    downtrend_LL_price = sl_price
                    potential_LH_price = np.nan # Reset for next leg
                    potential_LH_idx = -1
    return df_out

# --- Order Block Detection Function ---
def detect_order_blocks(df):
    df_out = df.copy()
    df_out['bullish_ob_low'] = np.nan
    df_out['bullish_ob_high'] = np.nan
    df_out['bearish_ob_low'] = np.nan
    df_out['bearish_ob_high'] = np.nan

    # Bullish OB: Last bearish candle before a strong bullish move
    # Strong bullish move: next candle's Low > current bearish candle's High
    is_bearish_candle = df_out['Close'] < df_out['Open']
    strong_bullish_move_follows = df_out['Low'].shift(-1) > df_out['High']
    is_bullish_ob_condition = is_bearish_candle & strong_bullish_move_follows
    
    df_out.loc[is_bullish_ob_condition, 'bullish_ob_low'] = df_out.loc[is_bullish_ob_condition, 'Low']
    df_out.loc[is_bullish_ob_condition, 'bullish_ob_high'] = df_out.loc[is_bullish_ob_condition, 'High']

    # Bearish OB: Last bullish candle before a strong bearish move
    # Strong bearish move: next candle's High < current bullish candle's Low
    is_bullish_candle = df_out['Close'] > df_out['Open']
    strong_bearish_move_follows = df_out['High'].shift(-1) < df_out['Low']
    is_bearish_ob_condition = is_bullish_candle & strong_bearish_move_follows

    df_out.loc[is_bearish_ob_condition, 'bearish_ob_low'] = df_out.loc[is_bearish_ob_condition, 'Low']
    df_out.loc[is_bearish_ob_condition, 'bearish_ob_high'] = df_out.loc[is_bearish_ob_condition, 'High']
    
    return df_out

# --- Fair Value Gap Detection Function ---
def detect_fair_value_gaps(df):
    df_out = df.copy()
    df_out['bullish_fvg_low'] = np.nan
    df_out['bullish_fvg_high'] = np.nan
    df_out['bearish_fvg_low'] = np.nan
    df_out['bearish_fvg_high'] = np.nan

    # Bullish FVG: Candle_1.High < Candle_3.Low (FVG on Candle_2)
    # Candle_1 is df.shift(1), Candle_3 is df.shift(-1) relative to Candle_2 (current row)
    is_bullish_fvg = df_out['High'].shift(1) < df_out['Low'].shift(-1)
    df_out.loc[is_bullish_fvg, 'bullish_fvg_low'] = df_out['High'].shift(1)[is_bullish_fvg]
    df_out.loc[is_bullish_fvg, 'bullish_fvg_high'] = df_out['Low'].shift(-1)[is_bullish_fvg]

    # Bearish FVG: Candle_1.Low > Candle_3.High (FVG on Candle_2)
    is_bearish_fvg = df_out['Low'].shift(1) > df_out['High'].shift(-1)
    df_out.loc[is_bearish_fvg, 'bearish_fvg_low'] = df_out['High'].shift(-1)[is_bearish_fvg]
    df_out.loc[is_bearish_fvg, 'bearish_fvg_high'] = df_out['Low'].shift(1)[is_bearish_fvg]
    
    return df_out

# --- Liquidity Feature Detection Function ---
def detect_liquidity_features(df, lookback_eql=20, price_tolerance_pips=5, pip_val=0.0001):
    df_out = df.copy()
    price_tolerance_abs = price_tolerance_pips * pip_val

    df_out['liquidity_sweep_bullish'] = False
    df_out['liquidity_sweep_bearish'] = False
    df_out['equal_highs_level'] = np.nan
    df_out['equal_lows_level'] = np.nan
    df_out['inducement_sweep_price'] = np.nan # Price of the SH/SL that was swept

    last_confirmed_sh_price = np.nan
    last_confirmed_sh_idx = -1
    last_confirmed_sl_price = np.nan
    last_confirmed_sl_idx = -1

    # For EQL H/L detection (storing indices of SH/SLs within lookback)
    recent_sh_indices = deque(maxlen=lookback_eql)
    recent_sl_indices = deque(maxlen=lookback_eql)

    for i in range(len(df_out)):
        current_idx_label = df_out.index[i]
        row = df_out.iloc[i]
        current_high = row['High']
        current_low = row['Low']
        current_close = row['Close']

        # Update last confirmed SH/SL based on current candle's swing status
        if row['is_swing_high']:
            last_confirmed_sh_price = current_high
            last_confirmed_sh_idx = i
            # Check for Equal Highs
            for sh_prev_idx_pos in recent_sh_indices: # sh_prev_idx_pos is integer index
                if abs(current_high - df_out['High'].iloc[sh_prev_idx_pos]) <= price_tolerance_abs:
                    df_out.loc[current_idx_label, 'equal_highs_level'] = current_high # Mark current SH
                    # Mark previous SH as well, if not already part of another EQLH at same level
                    # This can get complex if multiple EQL highs form. Storing the level is key.
                    # df_out.loc[df_out.index[sh_prev_idx_pos], 'equal_highs_level'] = df_out['High'].iloc[sh_prev_idx_pos]
                    break
            recent_sh_indices.append(i)
            
        if row['is_swing_low']:
            last_confirmed_sl_price = current_low
            last_confirmed_sl_idx = i
            # Check for Equal Lows
            for sl_prev_idx_pos in recent_sl_indices:
                if abs(current_low - df_out['Low'].iloc[sl_prev_idx_pos]) <= price_tolerance_abs:
                    df_out.loc[current_idx_label, 'equal_lows_level'] = current_low
                    break
            recent_sl_indices.append(i)

        # Liquidity Sweeps & Inducement
        # Bullish Sweep (sweeping a prior SL)
        if not pd.isna(last_confirmed_sl_price) and i > last_confirmed_sl_idx: # Ensure SL is in the past
            if current_low < last_confirmed_sl_price and current_close > last_confirmed_sl_price:
                df_out.loc[current_idx_label, 'liquidity_sweep_bullish'] = True
                df_out.loc[current_idx_label, 'inducement_sweep_price'] = last_confirmed_sl_price
        
        # Bearish Sweep (sweeping a prior SH)
        if not pd.isna(last_confirmed_sh_price) and i > last_confirmed_sh_idx: # Ensure SH is in the past
            if current_high > last_confirmed_sh_price and current_close < last_confirmed_sh_price:
                df_out.loc[current_idx_label, 'liquidity_sweep_bearish'] = True
                df_out.loc[current_idx_label, 'inducement_sweep_price'] = last_confirmed_sh_price
                
    return df_out

# --- Higher Timeframe Narrative/Bias Determination Function ---
def determine_higher_tf_narrative(df_htf, col_name_prefix):
    narrative_col_name = f'{col_name_prefix}_Narrative' # Consistent naming, can be interpreted as bias for 15M
    narratives = pd.Series(index=df_htf.index, dtype=str, name=narrative_col_name)
    current_narrative = 'Undetermined'

    for idx, row in df_htf.iterrows():
        if row['bos_bullish'] or row['choch_bullish']:
            current_narrative = 'Bullish'
        elif row['bos_bearish'] or row['choch_bearish']:
            current_narrative = 'Bearish'
        # If no event, narrative persists
        narratives.loc[idx] = current_narrative
    
    return narratives

# --- Entry Signal Detection Function ---
def detect_entry_signals(df):
    """
    Detects 5-minute entry signals based on HTF narrative, bias, POI interaction,
    5M CHoCH, and (optionally) preceding liquidity sweeps.
    Filters for London session and avoids red news.
    """
    df['long_signal'] = False
    df['short_signal'] = False

    # Ensure necessary columns exist, fill with False or appropriate neutral value if not
    # This is important if, for example, liquidity sweep features are optional or not always computed
    for col in ['is_near_red_news', 'choch_bullish', 'choch_bearish', 
                'liquidity_sweep_bearish', 'liquidity_sweep_bullish']:
        if col not in df.columns:
            df[col] = False # Assuming boolean, adjust if different

    # Time filter: London session (8:00-16:59 UTC)
    london_session = (df.index.hour >= 8) & (df.index.hour < 17)

    # News filter
    # Ensure 'is_near_red_news' exists, if not, assume not near news (or handle as error)
    if 'is_near_red_news' in df.columns:
        not_near_red_news = ~df['is_near_red_news']
    else:
        print("Warning: 'is_near_red_news' column not found in df_5m. Assuming no news filter.")
        not_near_red_news = True


    base_filter = london_session & not_near_red_news

    # --- Long Conditions ---
    if '4H_Narrative' in df.columns and '15M_Daily_Bias' in df.columns:
        long_cond1_narrative_bias = (df['4H_Narrative'] == 'Bullish') & (df['15M_Daily_Bias'] == 'Bullish')

        # POI Interaction Checks
        interact_4h_bull_ob = pd.Series(False, index=df.index)
        if '4H_bullish_ob_low' in df.columns and '4H_bullish_ob_high' in df.columns:
            interact_4h_bull_ob = (df['Low'] <= df['4H_bullish_ob_high']) & \
                                  (df['High'] >= df['4H_bullish_ob_low']) & \
                                  df['4H_bullish_ob_low'].notna()

        interact_4h_bull_fvg = pd.Series(False, index=df.index)
        if '4H_bullish_fvg_low' in df.columns and '4H_bullish_fvg_high' in df.columns:
            interact_4h_bull_fvg = (df['Low'] <= df['4H_bullish_fvg_high']) & \
                                   (df['High'] >= df['4H_bullish_fvg_low']) & \
                                   df['4H_bullish_fvg_low'].notna()

        interact_15m_bull_ob = pd.Series(False, index=df.index)
        if '15M_bullish_ob_low' in df.columns and '15M_bullish_ob_high' in df.columns:
            interact_15m_bull_ob = (df['Low'] <= df['15M_bullish_ob_high']) & \
                                   (df['High'] >= df['15M_bullish_ob_low']) & \
                                   df['15M_bullish_ob_low'].notna()
        
        interact_15m_bull_fvg = pd.Series(False, index=df.index)
        if '15M_bullish_fvg_low' in df.columns and '15M_bullish_fvg_high' in df.columns:
            interact_15m_bull_fvg = (df['Low'] <= df['15M_bullish_fvg_high']) & \
                                    (df['High'] >= df['15M_bullish_fvg_low']) & \
                                    df['15M_bullish_fvg_low'].notna()

        long_cond2_poi = interact_4h_bull_ob | interact_4h_bull_fvg | \
                         interact_15m_bull_ob | interact_15m_bull_fvg
                         
        long_cond3_choch = df['choch_bullish']
        long_cond4_sweep = df['liquidity_sweep_bearish'] # Assumes current candle

        df.loc[base_filter & long_cond1_narrative_bias & long_cond2_poi & long_cond3_choch & long_cond4_sweep, 'long_signal'] = True
    else:
        print("Warning: '4H_Narrative' or '15M_Daily_Bias' not found. Skipping long signal calculation.")


    # --- Short Conditions ---
    if '4H_Narrative' in df.columns and '15M_Daily_Bias' in df.columns:
        short_cond1_narrative_bias = (df['4H_Narrative'] == 'Bearish') & (df['15M_Daily_Bias'] == 'Bearish')

        interact_4h_bear_ob = pd.Series(False, index=df.index)
        if '4H_bearish_ob_low' in df.columns and '4H_bearish_ob_high' in df.columns:
            interact_4h_bear_ob = (df['Low'] <= df['4H_bearish_ob_high']) & \
                                  (df['High'] >= df['4H_bearish_ob_low']) & \
                                  df['4H_bearish_ob_low'].notna()

        interact_4h_bear_fvg = pd.Series(False, index=df.index)
        if '4H_bearish_fvg_low' in df.columns and '4H_bearish_fvg_high' in df.columns:
            interact_4h_bear_fvg = (df['Low'] <= df['4H_bearish_fvg_high']) & \
                                   (df['High'] >= df['4H_bearish_fvg_low']) & \
                                   df['4H_bearish_fvg_low'].notna()

        interact_15m_bear_ob = pd.Series(False, index=df.index)
        if '15M_bearish_ob_low' in df.columns and '15M_bearish_ob_high' in df.columns:
            interact_15m_bear_ob = (df['Low'] <= df['15M_bearish_ob_high']) & \
                                   (df['High'] >= df['15M_bearish_ob_low']) & \
                                   df['15M_bearish_ob_low'].notna()

        interact_15m_bear_fvg = pd.Series(False, index=df.index)
        if '15M_bearish_fvg_low' in df.columns and '15M_bearish_fvg_high' in df.columns:
            interact_15m_bear_fvg = (df['Low'] <= df['15M_bearish_fvg_high']) & \
                                    (df['High'] >= df['15M_bearish_fvg_low']) & \
                                    df['15M_bearish_fvg_low'].notna()
                            
        short_cond2_poi = interact_4h_bear_ob | interact_4h_bear_fvg | \
                          interact_15m_bear_ob | interact_15m_bear_fvg
                          
        short_cond3_choch = df['choch_bearish']
        short_cond4_sweep = df['liquidity_sweep_bullish'] # Assumes current candle

        df.loc[base_filter & short_cond1_narrative_bias & short_cond2_poi & short_cond3_choch & short_cond4_sweep, 'short_signal'] = True
    else:
        print("Warning: '4H_Narrative' or '15M_Daily_Bias' not found. Skipping short signal calculation.")
        
    return df

# --- Trade Management (SL/TP) Calculation Function ---
def calculate_trade_management(df_5m_data, df_4h_full, df_15m_full):
    df = df_5m_data.copy()
    df['sl_price'] = np.nan
    df['tp_price'] = np.nan

    sl_buffer_abs = SL_BUFFER_PIPS * PIP_VALUE

    # Iterate over a list of indices where a signal is initially true
    # This is important because we might modify the signal flags within the loop
    signal_indices = df.index[(df['long_signal']) | (df['short_signal'])].tolist()

    for signal_idx in signal_indices:
        current_candle_5m = df.loc[signal_idx]
        entry_price = current_candle_5m['Close']
        calculated_sl = np.nan
        calculated_tp = np.nan
        
        is_long_trade = current_candle_5m['long_signal']
        is_short_trade = current_candle_5m['short_signal']

        if is_long_trade:
            # Stop Loss for Long
            calculated_sl = current_candle_5m['Low'] - sl_buffer_abs

            # Take Profit for Long: look forward for targets
            potential_tps = []
            df_5m_future = df[df.index > signal_idx]
            df_15m_future = df_15m_full[df_15m_full.index > signal_idx]
            df_4h_future = df_4h_full[df_4h_full.index > signal_idx]

            # Target 1: 4H Swing High
            future_4h_sh = df_4h_future[df_4h_future['is_swing_high']]
            if not future_4h_sh.empty:
                potential_tps.append(future_4h_sh['High'].iloc[0])
            
            # Target 2: 15M Swing High
            future_15m_sh = df_15m_future[df_15m_future['is_swing_high']]
            if not future_15m_sh.empty:
                potential_tps.append(future_15m_sh['High'].iloc[0])

            # Target 3: 5M Equal Highs Level
            future_5m_eqh = df_5m_future[df_5m_future['equal_highs_level'].notna()]
            if not future_5m_eqh.empty:
                potential_tps.append(future_5m_eqh['equal_highs_level'].iloc[0])

            # Target 4: Lower boundary of 4H Bearish FVG
            future_4h_bear_fvg = df_4h_future[df_4h_future['bearish_fvg_low'].notna()]
            if not future_4h_bear_fvg.empty:
                potential_tps.append(future_4h_bear_fvg['bearish_fvg_low'].iloc[0])

            # Target 5: Lower boundary of 15M Bearish FVG
            future_15m_bear_fvg = df_15m_future[df_15m_future['bearish_fvg_low'].notna()]
            if not future_15m_bear_fvg.empty:
                potential_tps.append(future_15m_bear_fvg['bearish_fvg_low'].iloc[0])

            valid_tps = [tp for tp in potential_tps if tp > entry_price]
            if valid_tps:
                calculated_tp = min(valid_tps)

        elif is_short_trade:
            # Stop Loss for Short
            calculated_sl = current_candle_5m['High'] + sl_buffer_abs

            # Take Profit for Short: look forward for targets
            potential_tps = []
            df_5m_future = df[df.index > signal_idx]
            df_15m_future = df_15m_full[df_15m_full.index > signal_idx]
            df_4h_future = df_4h_full[df_4h_full.index > signal_idx]

            # Target 1: 4H Swing Low
            future_4h_sl = df_4h_future[df_4h_future['is_swing_low']]
            if not future_4h_sl.empty:
                potential_tps.append(future_4h_sl['Low'].iloc[0])

            # Target 2: 15M Swing Low
            future_15m_sl = df_15m_future[df_15m_future['is_swing_low']]
            if not future_15m_sl.empty:
                potential_tps.append(future_15m_sl['Low'].iloc[0])

            # Target 3: 5M Equal Lows Level
            future_5m_eql = df_5m_future[df_5m_future['equal_lows_level'].notna()]
            if not future_5m_eql.empty:
                potential_tps.append(future_5m_eql['equal_lows_level'].iloc[0])

            # Target 4: Upper boundary of 4H Bullish FVG
            future_4h_bull_fvg = df_4h_future[df_4h_future['bullish_fvg_high'].notna()]
            if not future_4h_bull_fvg.empty:
                potential_tps.append(future_4h_bull_fvg['bullish_fvg_high'].iloc[0])

            # Target 5: Upper boundary of 15M Bullish FVG
            future_15m_bull_fvg = df_15m_future[df_15m_future['bullish_fvg_high'].notna()]
            if not future_15m_bull_fvg.empty:
                potential_tps.append(future_15m_bull_fvg['bullish_fvg_high'].iloc[0])
            
            valid_tps = [tp for tp in potential_tps if tp < entry_price]
            if valid_tps:
                calculated_tp = max(valid_tps)

        # Risk-Reward Check
        if not pd.isna(calculated_sl) and not pd.isna(calculated_tp):
            risk = 0
            reward = 0
            if is_long_trade:
                risk = entry_price - calculated_sl
                reward = calculated_tp - entry_price
            elif is_short_trade:
                risk = calculated_sl - entry_price
                reward = entry_price - calculated_tp
            
            if risk > 0 and (reward / risk) >= MIN_RR_RATIO:
                df.loc[signal_idx, 'sl_price'] = calculated_sl
                df.loc[signal_idx, 'tp_price'] = calculated_tp
            else:
                # R:R not met or invalid risk, invalidate signal
                if is_long_trade:
                    df.loc[signal_idx, 'long_signal'] = False
                elif is_short_trade:
                    df.loc[signal_idx, 'short_signal'] = False
        else:
            # SL or TP could not be determined, invalidate signal
            if is_long_trade:
                df.loc[signal_idx, 'long_signal'] = False
            elif is_short_trade:
                df.loc[signal_idx, 'short_signal'] = False
                
    return df

# --- Backtesting Engine ---
def run_backtest(df_5m_data):
    print("\nStarting Backtest...")
    trades_list = []
    in_trade = False
    current_trade = {}
    trade_id_counter = 0

    # Ensure the DataFrame index is a DatetimeIndex for row.Index access
    if not isinstance(df_5m_data.index, pd.DatetimeIndex):
        # This should not happen if setup correctly, but as a safeguard:
        df_5m_data.index = pd.to_datetime(df_5m_data.index)

    for row in df_5m_data.itertuples(index=True): # index=True makes row.Index available
        current_candle_timestamp = row.Index
        current_candle_open = row.Open
        current_candle_high = row.High
        current_candle_low = row.Low
        current_candle_close = row.Close
        is_big_news_candle = row.is_big_red_flag_news_candle

        if in_trade:
            # PRIORITY 1: Big Red Flag News Exit
            if is_big_news_candle:
                # Use Open of news candle for exit to simulate immediate reaction
                exit_price = current_candle_open 
                current_trade['exit_time'] = current_candle_timestamp
                current_trade['exit_price'] = exit_price
                current_trade['exit_reason'] = 'Big News Exit'
                
                if current_trade['trade_type'] == 'long':
                    current_trade['pnl'] = exit_price - current_trade['entry_price']
                else: # short
                    current_trade['pnl'] = current_trade['entry_price'] - exit_price
                
                trades_list.append(current_trade.copy()) # Use .copy() important for dicts
                in_trade = False
                current_trade = {}
                continue # Skip SL/TP check for this candle

            # PRIORITY 2: SL/TP Check (if not exited by news)
            sl = current_trade['sl_price']
            tp = current_trade['tp_price']
            exit_reason = None
            exit_p = None

            if current_trade['trade_type'] == 'long':
                if current_candle_low <= sl:
                    exit_p = sl
                    exit_reason = 'Stop Loss'
                elif current_candle_high >= tp:
                    exit_p = tp
                    exit_reason = 'Take Profit'
            else: # short
                if current_candle_high >= sl:
                    exit_p = sl
                    exit_reason = 'Stop Loss'
                elif current_candle_low <= tp:
                    exit_p = tp
                    exit_reason = 'Take Profit'
            
            if exit_reason:
                current_trade['exit_time'] = current_candle_timestamp
                current_trade['exit_price'] = exit_p
                current_trade['exit_reason'] = exit_reason
                if current_trade['trade_type'] == 'long':
                    current_trade['pnl'] = exit_p - current_trade['entry_price']
                else:
                    current_trade['pnl'] = current_trade['entry_price'] - exit_p
                
                trades_list.append(current_trade.copy())
                in_trade = False
                current_trade = {}
                continue # Move to next candle after closing trade
        
        # If NOT in trade, check for new entry signals
        if not in_trade:
            # Ensure sl_price and tp_price are valid before entering a trade
            if row.long_signal and not pd.isna(row.sl_price) and not pd.isna(row.tp_price):
                in_trade = True
                trade_id_counter += 1
                current_trade = {
                    'id': trade_id_counter,
                    'entry_time': current_candle_timestamp,
                    'entry_price': current_candle_close, # Entry at close of signal candle
                    'sl_price': row.sl_price,
                    'tp_price': row.tp_price,
                    'trade_type': 'long'
                }
            elif row.short_signal and not pd.isna(row.sl_price) and not pd.isna(row.tp_price):
                in_trade = True
                trade_id_counter += 1
                current_trade = {
                    'id': trade_id_counter,
                    'entry_time': current_candle_timestamp,
                    'entry_price': current_candle_close, # Entry at close of signal candle
                    'sl_price': row.sl_price,
                    'tp_price': row.tp_price,
                    'trade_type': 'short'
                }

    # After the loop, if still in trade, close it at the last candle's close
    if in_trade:
        last_candle = df_5m_data.iloc[-1]
        exit_price = last_candle['Close']
        current_trade['exit_time'] = df_5m_data.index[-1]
        current_trade['exit_price'] = exit_price
        current_trade['exit_reason'] = 'End of Backtest'
        if current_trade['trade_type'] == 'long':
            current_trade['pnl'] = exit_price - current_trade['entry_price']
        else:
            current_trade['pnl'] = current_trade['entry_price'] - exit_price
        trades_list.append(current_trade.copy())

    print(f"Backtest Complete. Total trades considered: {trade_id_counter}")
    if not trades_list:
        print("No trades were executed.")
        return pd.DataFrame() # Return empty DataFrame

    trades_df = pd.DataFrame(trades_list)
    
    # Calculate PnL in pips for reporting (optional, but good for fixed pip value assets)
    # trades_df['pnl_pips'] = trades_df['pnl'] / PIP_VALUE

    print("\n--- Backtest Results Summary ---")
    print(f"Total Trades Executed: {len(trades_df)}")
    if not trades_df.empty:
        print(f"Total PnL (price units): {trades_df['pnl'].sum():.5f}")
        # print(f"Total PnL (pips): {trades_df['pnl_pips'].sum():.2f}")
        print("\nExit Reasons Distribution:")
        print(trades_df['exit_reason'].value_counts())
        print("\nSample Trades:")
        print(trades_df.head())
        if len(trades_df) > 5:
            print("\nLast 5 Trades:")
            print(trades_df.tail())
    else:
        print("No trades were executed to summarize.")
        
    return trades_df

# --- 1. Load EURUSD OHLC Data ---
try:
    df_ohlc = pd.read_csv(ohlc_file_path)
except FileNotFoundError:
    print(f"Error: OHLC file not found at {ohlc_file_path}")
    exit()

# Combine Date and Time columns into a new DateTime column
if 'Date' in df_ohlc.columns and 'Time' in df_ohlc.columns:
    df_ohlc['DateTime_str'] = df_ohlc['Date'] + ',' + df_ohlc['Time']
    # Parse DateTime_str, set as UTC index
    df_ohlc['DateTime'] = pd.to_datetime(df_ohlc['DateTime_str'], format='%Y.%m.%d,%H:%M')
    # Drop the intermediate and original date/time columns if they are no longer needed and not the index
    df_ohlc = df_ohlc.drop(columns=['Date', 'Time', 'DateTime_str'])
elif 'DateTime' in df_ohlc.columns:
    # Fallback for the original single 'DateTime' column case, though less likely now
    df_ohlc['DateTime'] = pd.to_datetime(df_ohlc['DateTime'], format='%Y.%m.%d,%H:%M')
else:
    print(f"Error: CSV file {ohlc_file_path} must contain either 'Date' and 'Time' columns or a 'DateTime' column.")
    exit()

# Crucial Timezone Step for OHLC: Assuming DateTime is UTC naive
df_ohlc['DateTime'] = df_ohlc['DateTime'].dt.tz_localize('UTC')
df_ohlc = df_ohlc.set_index('DateTime')

# Store raw dataframe
df_raw = df_ohlc.copy()

# --- 2. Resample OHLC Data ---
ohlc_dict = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last'
}

df_4h = df_ohlc.resample('4h').apply(ohlc_dict).dropna()
df_15m = df_ohlc.resample('15min').apply(ohlc_dict).dropna() # Changed from 15M to 15min
df_5m = df_ohlc.resample('5min').apply(ohlc_dict).dropna()   # Changed from 5M to 5min

# --- 3. Load News Calendar Data ---
try:
    df_news = pd.read_csv(news_file_path, usecols=news_column_names) # Ensure only specified columns are used
except FileNotFoundError:
    print(f"Error: News calendar file not found at {news_file_path}")
    exit()
except ValueError as e:
    print(f"Error loading news_calendar.csv. Make sure it contains the expected columns: {news_column_names}. Details: {e}")
    # Attempt to load with all columns to see what's there if usecols fails
    try:
        df_news_all_cols = pd.read_csv(news_file_path)
        print(f"Actual columns in news_calendar.csv: {df_news_all_cols.columns.tolist()}")
    except Exception as read_e:
        print(f"Could not even read the news file to check columns: {read_e}")
    exit()


# Parse Timestamp (UTC), set as UTC index
df_news['Timestamp (UTC)'] = pd.to_datetime(df_news['Timestamp (UTC)'])
if df_news['Timestamp (UTC)'].dt.tz is None:
    df_news['Timestamp (UTC)'] = df_news['Timestamp (UTC)'].dt.tz_localize('UTC')
else:
    df_news['Timestamp (UTC)'] = df_news['Timestamp (UTC)'].dt.tz_convert('UTC')
df_news = df_news.set_index('Timestamp (UTC)')

# --- 4. Filter News Data ---
df_news_filtered = df_news[
    df_news['Currency'].isin(['USD', 'EUR'])
]
df_red_news = df_news_filtered[
    df_news_filtered['Impact'].isin(['High', 'Red']) # Assuming 'Red' is a possible value for high impact
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Add a flag for big red flag events
df_red_news['is_big_event'] = df_red_news['Event'].apply(lambda x: any(event_kw in str(x) for event_kw in big_red_flag_events))

# --- 5. Merge News Data into df_5m ---
df_5m['is_near_red_news'] = False
df_5m['is_big_red_flag_news_candle'] = False

pre_news_window = pd.Timedelta(minutes=30)
post_news_window_end = pd.Timedelta(minutes=5) - pd.Timedelta(seconds=1) # up to the end of the 5-min candle news occurs in

for news_time, news_event_row in df_red_news.iterrows():
    # Define the window for 'is_near_red_news'
    start_window = news_time - pre_news_window
    end_window = news_time + post_news_window_end

    # Mark candles within this window
    df_5m.loc[
        (df_5m.index >= start_window) & (df_5m.index <= end_window),
        'is_near_red_news'
    ] = True

    # For 'is_big_red_flag_news_candle'
    if news_event_row['is_big_event']:
        # Find the 5-minute candle that *contains* the news_time
        # The candle starts at or before news_time and ends after news_time
        # Candle start time <= news_time < Candle end time (start_time + 5 minutes)
        
        # To find the specific candle:
        # It's the candle whose index is the floor of news_time to 5min interval
        target_candle_start_time = news_time.floor('5min')

        if target_candle_start_time in df_5m.index:
            df_5m.loc[target_candle_start_time, 'is_big_red_flag_news_candle'] = True
        else:
            # If the exact floored time is not in index (e.g. due to market closure, or sparse data)
            # try to find the closest preceding candle.
            # This might happen if news is exactly at market open before first 5m candle forms.
            # We will use .get_loc with method='ffill' for this case, but it's more robust to ensure
            # the news_time is within the candle's [start, end) interval.
            # An alternative, if `target_candle_start_time` is not in the index, would be to find the candle
            # whose `index <= news_time` and `index + pd.Timedelta(minutes=5) > news_time`.
            
            # Using `loc` with a slice up to news_time and taking the last one.
            # This gets the candle that starts at or before the news.
            potential_candles = df_5m.loc[df_5m.index <= news_time]
            if not potential_candles.empty:
                closest_preceding_candle_time = potential_candles.index[-1]
                # Check if this candle actually contains the news time
                if closest_preceding_candle_time <= news_time < closest_preceding_candle_time + pd.Timedelta(minutes=5):
                     df_5m.loc[closest_preceding_candle_time, 'is_big_red_flag_news_candle'] = True
                # else:
                    # print(f"Warning: Big red flag event at {news_time} for '{news_event_row['Event']}' did not align perfectly with a 5M candle start. Closest preceding is {closest_preceding_candle_time}, but news is not within its interval.")

# --- 6. Identify Swing Points ---
print("\nIdentifying swing points...")
df_4h = identify_swing_high(df_4h, N=SWING_N)
df_4h = identify_swing_low(df_4h, N=SWING_N)

df_15m = identify_swing_high(df_15m, N=SWING_N)
df_15m = identify_swing_low(df_15m, N=SWING_N)

df_5m = identify_swing_high(df_5m, N=SWING_N)
df_5m = identify_swing_low(df_5m, N=SWING_N)
print("Swing points identified.")

# --- 7. Detect Market Structure ---
print("\nDetecting market structure (BOS/CHoCH)...")
df_4h = detect_market_structure(df_4h)
df_15m = detect_market_structure(df_15m)
df_5m = detect_market_structure(df_5m)
print("Market structure detection complete.")

# --- 8. Detect Order Blocks and FVGs ---
print("\nDetecting Order Blocks and Fair Value Gaps...")
df_4h = detect_order_blocks(df_4h)
df_4h = detect_fair_value_gaps(df_4h)

df_15m = detect_order_blocks(df_15m)
df_15m = detect_fair_value_gaps(df_15m)

df_5m = detect_order_blocks(df_5m)
df_5m = detect_fair_value_gaps(df_5m)
print("Order Blocks and FVGs detection complete.")

# --- 9. Detect Liquidity Features ---
print("\nDetecting Liquidity Features (Sweeps, EQL H/L, Inducement)...")
df_4h = detect_liquidity_features(df_4h, lookback_eql=LIQUIDITY_LOOKBACK, price_tolerance_pips=PRICE_TOLERANCE_PIPS, pip_val=PIP_VALUE)
df_15m = detect_liquidity_features(df_15m, lookback_eql=LIQUIDITY_LOOKBACK, price_tolerance_pips=PRICE_TOLERANCE_PIPS, pip_val=PIP_VALUE)
# For 5M, perhaps a smaller tolerance for EQL H/L might be better, e.g., 2-3 pips
df_5m = detect_liquidity_features(df_5m, lookback_eql=LIQUIDITY_LOOKBACK, price_tolerance_pips=3, pip_val=PIP_VALUE) 
print("Liquidity features detection complete.")

# --- 10. Determine Multi-Timeframe Narrative and Bias ---
print("\nDetermining Multi-Timeframe Narrative and Bias...")

# Determine 4H Narrative
df_4h_narrative = determine_higher_tf_narrative(df_4h, "4H")
df_4h = df_4h.join(df_4h_narrative) # Add to df_4h for completeness if needed elsewhere

# Determine 15M Bias (using the same logic for its own timeframe structure)
df_15m_bias_series = determine_higher_tf_narrative(df_15m, "15M") # Output will be "15M_Narrative"
df_15m_bias_series.name = "15M_Daily_Bias" # Rename for clarity as per user request
df_15m = df_15m.join(df_15m_bias_series) # df_15m_bias_series already has the correct name

# Merge/Align with df_5m
# Align 4H Narrative to 5M timeframe
df_5m['4H_Narrative'] = df_4h_narrative.reindex(df_5m.index, method='ffill').fillna('Undetermined')

# Align 15M Bias to 5M timeframe
df_5m['15M_Daily_Bias'] = df_15m_bias_series.reindex(df_5m.index, method='ffill').fillna('Undetermined') # Use df_15m_bias_series

print("Multi-Timeframe Narrative and Bias determination complete.")

# --- 10.5 Propagate HTF POIs to 5M DataFrame ---
print("\nPropagating Higher Timeframe POIs (OBs/FVGs/EQLs) to 5M data...") # Updated print
poi_columns_base = [
    'bullish_ob_low', 'bullish_ob_high', 'bearish_ob_low', 'bearish_ob_high', 
    'bullish_fvg_low', 'bullish_fvg_high', 'bearish_fvg_low', 'bearish_fvg_high',
    'equal_highs_level', 'equal_lows_level' # Added Equal Highs/Lows
]

for col_base in poi_columns_base:
    # 4H POIs
    if col_base in df_4h.columns: # Ensure column exists before trying to access
        df_5m[f'4H_{col_base}'] = df_4h[col_base].reindex(df_5m.index, method='ffill')
    else:
        df_5m[f'4H_{col_base}'] = np.nan # Add column with NaNs if not found in HTF
    # 15M POIs
    if col_base in df_15m.columns: # Ensure column exists
        df_5m[f'15M_{col_base}'] = df_15m[col_base].reindex(df_5m.index, method='ffill')
    else:
        df_5m[f'15M_{col_base}'] = np.nan

print("HTF POI propagation complete.")

# --- 11. Detect Entry Signals on 5M ---
print("\nDetecting Entry Signals on 5M timeframe...")
df_5m = detect_entry_signals(df_5m)
print("Entry signal detection complete.")

# --- 11.5 Calculate Trade Management (SL/TP) and Filter by R:R ---
print("\nCalculating SL/TP and filtering signals by R:R...")
# We need to pass the full df_4h and df_15m dataframes for TP lookups
df_5m = calculate_trade_management(df_5m, df_4h, df_15m) 
print("SL/TP calculation and R:R filtering complete.")

# --- 12. Output ---
print("\n--- df_raw (EURUSD OHLC Raw) ---")
print(df_raw.head())
print(f"Index timezone: {df_raw.index.tz}")
print("\n")

print("--- df_4h (EURUSD 4-Hour with all features) ---")
print(df_4h.head())
ob_fvg_liq_4h_sample = df_4h[
    df_4h['bullish_ob_low'].notna() | df_4h['bearish_ob_low'].notna() | 
    df_4h['bullish_fvg_low'].notna() | df_4h['bearish_fvg_low'].notna() |
    df_4h['liquidity_sweep_bullish'] | df_4h['liquidity_sweep_bearish'] |
    df_4h['equal_highs_level'].notna() | df_4h['equal_lows_level'].notna() |
    df_4h['inducement_sweep_price'].notna()
].copy() # Use .copy() to avoid SettingWithCopyWarning on potential future modifications
if not ob_fvg_liq_4h_sample.empty:
    print("Sample of 4h OB/FVG/Liquidity/Narrative events:")
    cols_to_show_4h = ob_fvg_liq_4h_sample.columns.tolist()
    if '4H_Narrative' not in cols_to_show_4h: cols_to_show_4h.append('4H_Narrative') 
    # defensive if sample itself doesn't have the narrative due to filtering
    sample_data_4h = df_4h.loc[ob_fvg_liq_4h_sample.index, cols_to_show_4h].head()
    print(sample_data_4h)
else:
    print("No OB/FVG/Liquidity events found in df_4h to show with narrative.")
print("\n")

print("--- df_15m (EURUSD 15-Minute with all features) ---")
print(df_15m.head())
ob_fvg_liq_15m_sample = df_15m[
    df_15m['bullish_ob_low'].notna() | df_15m['bearish_ob_low'].notna() | 
    df_15m['bullish_fvg_low'].notna() | df_15m['bearish_fvg_low'].notna() |
    df_15m['liquidity_sweep_bullish'] | df_15m['liquidity_sweep_bearish'] |
    df_15m['equal_highs_level'].notna() | df_15m['equal_lows_level'].notna() |
    df_15m['inducement_sweep_price'].notna()
].copy()
if not ob_fvg_liq_15m_sample.empty:
    print("Sample of 15m OB/FVG/Liquidity/Bias events:")
    cols_to_show_15m = ob_fvg_liq_15m_sample.columns.tolist()
    if '15M_Daily_Bias' not in cols_to_show_15m: cols_to_show_15m.append('15M_Daily_Bias') # Changed to 15M_Daily_Bias
    sample_data_15m = df_15m.loc[ob_fvg_liq_15m_sample.index, cols_to_show_15m].head()
    print(sample_data_15m)
else:
    print("No OB/FVG/Liquidity events found in df_15m to show with bias.")
print("\n")

print("--- df_5m (EURUSD 5-Minute with all features) ---")
# Updated columns to show for df_5m head, including 15M_Daily_Bias and sample propagated POIs
cols_for_5m_head = [
    'Open', 'High', 'Low', 'Close', '4H_Narrative', '15M_Daily_Bias', 
    '4H_bullish_ob_low', '15M_bearish_fvg_high', '4H_equal_highs_level', '15M_equal_lows_level',
    'long_signal', 'short_signal', 'sl_price', 'tp_price'
]
# Ensure all requested columns exist before trying to print, to avoid KeyErrors if some POIs are not generated
existing_cols_for_5m_head = [col for col in cols_for_5m_head if col in df_5m.columns]
print(df_5m[existing_cols_for_5m_head].head())

ob_fvg_liq_5m_sample = df_5m[
    df_5m['bullish_ob_low'].notna() | df_5m['bearish_ob_low'].notna() | 
    df_5m['bullish_fvg_low'].notna() | df_5m['bearish_fvg_low'].notna() |
    df_5m['liquidity_sweep_bullish'] | df_5m['liquidity_sweep_bearish'] |
    df_5m['equal_highs_level'].notna() | df_5m['equal_lows_level'].notna() |
    df_5m['inducement_sweep_price'].notna() |
    df_5m['long_signal'] | df_5m['short_signal'] |
    df_5m['sl_price'].notna() | df_5m['tp_price'].notna()
].copy()
if not ob_fvg_liq_5m_sample.empty:
    print("Sample of 5m OB/FVG/Liquidity/Narrative/Signal/SLTP events:") # Updated title
    cols_to_show_5m = ob_fvg_liq_5m_sample.columns.tolist()
    if '4H_Narrative' not in cols_to_show_5m: cols_to_show_5m.append('4H_Narrative')
    if '15M_Daily_Bias' not in cols_to_show_5m: cols_to_show_5m.append('15M_Daily_Bias') # Changed to 15M_Daily_Bias
    if 'long_signal' not in cols_to_show_5m: cols_to_show_5m.append('long_signal')
    if 'short_signal' not in cols_to_show_5m: cols_to_show_5m.append('short_signal')
    # Ensure essential columns are present for context if a signal is true
    essential_cols = ['Open', 'High', 'Low', 'Close', 'long_signal', 'short_signal', 'sl_price', 'tp_price', '4H_Narrative', '15M_Daily_Bias'] # Changed to 15M_Daily_Bias
    for col in essential_cols:
        if col not in cols_to_show_5m: cols_to_show_5m.insert(0, col) # Add to front if missing
    cols_to_show_5m = list(dict.fromkeys(cols_to_show_5m)) # Remove duplicates, preserve order

    sample_data_5m = df_5m.loc[ob_fvg_liq_5m_sample.index, cols_to_show_5m].head(10) # Show more rows
    print(sample_data_5m)
else:
    print("No OB/FVG/Liquidity/Signal/SLTP events found in df_5m to show with narrative and bias.")
print("\n")

print("--- df_5m Interesting Rows (all features including HTF Narrative/Bias & Signals & SL/TP) ---") # Updated title
interesting_rows_5m = df_5m[
    df_5m['is_near_red_news'] | df_5m['is_big_red_flag_news_candle'] |
    df_5m['bos_bullish'] | df_5m['bos_bearish'] | 
    df_5m['choch_bullish'] | df_5m['choch_bearish'] |
    df_5m['bullish_ob_low'].notna() | df_5m['bearish_ob_low'].notna() | 
    df_5m['bullish_fvg_low'].notna() | df_5m['bearish_fvg_low'].notna() |
    df_5m['liquidity_sweep_bullish'] | df_5m['liquidity_sweep_bearish'] |
    df_5m['equal_highs_level'].notna() | df_5m['equal_lows_level'].notna() |
    df_5m['inducement_sweep_price'].notna() |
    df_5m['4H_Narrative'].notna() | df_5m['15M_Daily_Bias'].notna() | # Changed to 15M_Daily_Bias
    df_5m['long_signal'] | df_5m['short_signal'] |
    df_5m['sl_price'].notna() | df_5m['tp_price'].notna()
].copy()
if not interesting_rows_5m.empty:
    cols_to_show_5m_interesting = [
        'Open', 'High', 'Low', 'Close', 'is_near_red_news', 'is_big_red_flag_news_candle', 
        'is_swing_high', 'is_swing_low', 'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
        'bullish_ob_low', 'bullish_ob_high', 'bearish_ob_low', 'bearish_ob_high',
        'bullish_fvg_low', 'bullish_fvg_high', 'bearish_fvg_low', 'bearish_fvg_high',
        '4H_bullish_ob_low', '4H_bullish_ob_high', '4H_bearish_ob_low', '4H_bearish_ob_high',
        '4H_bullish_fvg_low', '4H_bullish_fvg_high', '4H_bearish_fvg_low', '4H_bearish_fvg_high',
        '15M_bullish_ob_low', '15M_bullish_ob_high', '15M_bearish_ob_low', '15M_bearish_ob_high',
        '15M_bullish_fvg_low', '15M_bullish_fvg_high', '15M_bearish_fvg_low', '15M_bearish_fvg_high',
        'liquidity_sweep_bullish', 'liquidity_sweep_bearish', 'equal_highs_level', 'equal_lows_level', 'inducement_sweep_price',
        '4H_Narrative', '15M_Daily_Bias', 'long_signal', 'short_signal', 'sl_price', 'tp_price' # Changed to 15M_Daily_Bias
    ]
    # Ensure all columns exist in the dataframe before selecting
    cols_to_show_5m_interesting = [col for col in cols_to_show_5m_interesting if col in interesting_rows_5m.columns]
    print(interesting_rows_5m[cols_to_show_5m_interesting].head(30)) # Increased head to 30
else:
    print("No 5-minute candles were flagged for any significant events or signals.")

print("\n--- df_red_news (Filtered High-Impact News) ---")
print(df_red_news.head())
if df_red_news.empty:
    print("No high-impact news events found for EUR/USD after filtering.")

print(f"\nScript finished. Final df_5m shape: {df_5m.shape}")

# --- 13. Run Backtest ---
trades_summary_df = run_backtest(df_5m)

print("\n--- Full Backtest Summary (First 20 Trades if available) ---")
if not trades_summary_df.empty:
    print(trades_summary_df.head(20))
else:
    print("No trades to display in summary.") 