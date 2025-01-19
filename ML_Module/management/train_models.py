import numpy as np
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

tqdm.pandas()
from keras.api import Sequential
from keras.api.layers import Dense, Dropout, LSTM, Input
from keras.api.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ML_Module.utils.WassabiClient import initialize_wasabi_client


wasabi = initialize_wasabi_client()


def check_missing_data(df):
    for column in df.columns:
        missing_ratio = df[column].isna().sum() / len(df)
        if missing_ratio > 0.4:
            return False

    return True


def prepare_stock_data_analysis(df):
    try:
        numeric_columns = [
            'Last trade price', 'Max', 'Min', 'Avg. Price',
            '%chg.', 'Volume', 'Turnover in BEST in denars', 'Total turnover in denars'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)

        return df
    except Exception as e:
        return None


def handle_missing_values(df):

    df['Last trade price'] = df['Last trade price'].bfill()
    df['Turnover in BEST in denars'] = df['Turnover in BEST in denars'].fillna(0)
    df['Total turnover in denars'] = df['Total turnover in denars'].fillna(0)
    df['Volume'] = df['Volume'].fillna(0)
    df['%chg.'] = df['%chg.'].fillna(0)

    for index, row in df.iterrows():
        if pd.isna(row['Max']) and pd.isna(row['Min']):
            df.at[index, 'Max'] = row['Last trade price']
            df.at[index, 'Min'] = row['Last trade price']
        elif pd.isna(row['Max']):
            df.at[index, 'Max'] = row['Last trade price']
        elif pd.isna(row['Min']):
            df.at[index, 'Min'] = row['Last trade price']

        if pd.isna(row['Avg. Price']):
            df.at[index, 'Avg. Price'] = (row['Max'] + row['Min']) / 2

    return df


def train_model(stock_code):

    df = wasabi.fetch_data(stock_code)

    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')

    df.set_index('Date', inplace=True)

    # df.head()

    df = handle_missing_values(df)
    if check_missing_data(df) is None:
        return None

    df = prepare_stock_data_analysis(df)

    df = df[~df.index.duplicated(keep='first')]

    tech_indicators = pd.DataFrame()

    tech_indicators['garman_klass_vol'] = (
            (np.log(df['Max']) - np.log(df['Min'])) ** 2 / 2 -
            (2 * np.log(2) - 1) * (np.log(df['Last trade price']) - np.log(df['Avg. Price'])) ** 2
    )

    tech_indicators['atr'] = ta.atr(high=df['Max'], low=df['Min'], close=df['Last trade price'], length=15)

    tech_indicators['RSI'] = ta.rsi(df['Last trade price'], length=15)
    tech_indicators['EMAF'] = ta.ema(df['Last trade price'], length=15)

    df = df[['Last trade price']]

    final = pd.merge(df, tech_indicators, left_index=True, right_index=True)

    final.dropna(axis=0, inplace=True)

    columns_initial = ['RSI', 'EMAF', 'atr', 'garman_klass_vol']

    features = len(columns_initial) + 1

    periods = range(-1, -31, -1)

    lags = final.shift(periods=periods)

    lag_num = len(periods)

    lags.dropna(axis=0, inplace=True)

    final = pd.merge(final, lags, left_index=True, right_index=True)

    final.drop(columns=columns_initial, axis=1, inplace=True)


    train_x, test_x, train_y, test_y = train_test_split(final.drop(columns=['Last trade price'], axis=1),
                                                        final['Last trade price'], test_size=0.3, shuffle=False)

    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    train_x = train_x.reshape(train_x.shape[0], lag_num, features)
    test_x = test_x.reshape(test_x.shape[0], lag_num, features)

    def evaluate_prediction(orig_y, pred_y):
        print(f"MSE: ", mean_squared_error(orig_y, pred_y))
        print(f"MAE: ", mean_absolute_error(orig_y, pred_y))
        print(f"R2 SCORE: ", r2_score(orig_y, pred_y))

    model = Sequential([
        Input(shape=(lag_num, features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.01, clipnorm=1.0), loss='mse', metrics=['r2_score'])

    model.fit(train_x, train_y, epochs=100, batch_size=32, validation_split=0.15)

    # pred_y = model.predict(test_x)
    #
    # evaluate_prediction(test_y, pred_y)

    wasabi.save_model_to_cloud(stock_code, model)
    print(f"Model saved to cloud: {stock_code}.keras")
    return




for stock in ['ADIN', 'ALK', 'ALKB', 'AMBR', 'AMEH', 'APTK', 'ATPP', 'AUMK', 'BANA', 'BGOR', 'BIKF', 'BIM', 'BLTU', 'CBNG', 'CDHV', 'CEVI', 'CKB', 'CKBKO', 'DEBA', 'DIMI', 'EDST', 'ELMA', 'ELNC', 'ENER', 'ENSA', 'EUHA', 'EUMK', 'EVRO', 'FAKM', 'FERS', 'FKTL', 'FROT', 'FUBT', 'GALE', 'GDKM', 'GECK', 'GECT', 'GIMS', 'GRDN', 'GRNT', 'GRSN', 'GRZD', 'GTC', 'GTRG', 'IJUG', 'INB', 'INDI', 'INEK', 'INHO', 'INOV', 'INPR', 'INTP', 'JAKO', 'JULI', 'JUSK', 'KARO', 'KDFO', 'KJUBI', 'KKFI', 'KKST', 'KLST', 'KMB', 'KMPR', 'KOMU', 'KONF', 'KONZ', 'KORZ', 'KPSS', 'KULT', 'KVAS', 'LAJO', 'LHND', 'LOTO', 'LOZP', 'MAGP', 'MAKP', 'MAKS', 'MB', 'MERM', 'MKSD', 'MLKR', 'MODA', 'MPOL', 'MPT', 'MPTE', 'MTUR', 'MZHE', 'MZPU', 'NEME', 'NOSK', 'OBPP', 'OILK', 'OKTA', 'OMOS', 'OPFO', 'OPTK', 'ORAN', 'OSPO', 'OTEK', 'PELK', 'PGGV', 'PKB', 'POPK', 'PPIV', 'PROD', 'PROT', 'PTRS', 'RADE', 'REPL', 'RIMI', 'RINS', 'RZEK', 'RZIT', 'RZIZ', 'RZLE', 'RZLV', 'RZTK', 'RZUG', 'RZUS', 'SBT', 'SDOM', 'SIL', 'SKON', 'SKP', 'SLAV', 'SNBT', 'SNBTO', 'SOLN', 'SPAZ', 'SPAZP', 'SPOL', 'SSPR', 'STB', 'STBP', 'STIL', 'STOK', 'TAJM', 'TASK', 'TBKO', 'TEAL', 'TEHN', 'TEL', 'TETE', 'TIGA', 'TIKV', 'TKPR', 'TKVS', 'TNB', 'TRDB', 'TRPS', 'TRUB', 'TSMP', 'TSZS', 'TTK', 'UNI', 'USJE', 'VARG', 'VFPM', 'VITA', 'VROS', 'VSC', 'VTKS', 'ZAS', 'ZILU', 'ZILUP', 'ZIMS', 'ZKAR', 'ZPKO', 'ZPOG', 'ZSIL', 'ZUAS']:
    train_model(stock)

