import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataHandler:

    @staticmethod
    def prepare_stock_data_analysis(df):
        """"
        Casts all the columns to type float
        """
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
            print(f"Error preparing stock data: {e}")
            return None

    @staticmethod
    def handle_missing_values(data):
        """"
        handles the missing values in the last trade price column
        """
        data['Last trade price'] = data['Last trade price'].bfill()

        return data

    @staticmethod
    def check_missing_data(df):
        """"
        Checks for amount of missing data in the last trade pricce and if all the prices in this column are 0,
        Since its the only column we use during model training

        Returns true if there is too much missing data (more than 70%) or last trade price is filled with 0's
        """

        missing = df['Last trade price'].isna().sum() / len(df)

        all_zeros = (df['Last trade price'] == 0).all()

        return missing > 0.7 or all_zeros


    @staticmethod
    def get_fitted_scaler(df, train_date='2025-01-18'):
        """"
        Returns a scaler fitted with the data up to the date of the model training

        """
        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Date of training the models
        cutoff_date = pd.to_datetime(train_date)

        # Filter rows where 'Date' is before the cutoff date
        df = df[df['Date'] < cutoff_date]
        df = df[['Last trade price']]
        scaler = MinMaxScaler()
        scaler.fit(df['Last trade price'].values.reshape(-1, 1))
        return scaler
