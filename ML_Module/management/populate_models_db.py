import datetime
import os
import django
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Crystal.settings')
django.setup()
from ML_Module.models import StockPredictionModel


def populate_models_db():
    """Populates the StockPredictionModel database with entries for each stock."""
    today = datetime.date.today()

    for stock in ['ADIN', 'ALK', 'ALKB', 'AMBR', 'AMEH', 'APTK', 'ATPP', 'AUMK', 'BANA', 'BGOR', 'BIKF', 'BIM', 'BLTU', 'CBNG', 'CDHV', 'CEVI', 'CKB', 'CKBKO', 'DEBA', 'DIMI', 'EDST', 'ELMA', 'ELNC', 'ENER', 'ENSA', 'EUHA', 'EUMK', 'EVRO', 'FAKM', 'FERS', 'FKTL', 'FROT', 'FUBT', 'GALE', 'GDKM', 'GECK', 'GECT', 'GIMS', 'GRDN', 'GRNT', 'GRSN', 'GRZD', 'GTC', 'GTRG', 'IJUG', 'INB', 'INDI', 'INEK', 'INHO', 'INOV', 'INPR', 'INTP', 'JAKO', 'JULI', 'JUSK', 'KARO', 'KDFO', 'KJUBI', 'KKFI', 'KKST', 'KLST', 'KMB', 'KMPR', 'KOMU', 'KONF', 'KONZ', 'KORZ', 'KPSS', 'KULT', 'KVAS', 'LAJO', 'LHND', 'LOTO', 'LOZP', 'MAGP', 'MAKP', 'MAKS', 'MB', 'MERM', 'MKSD', 'MLKR', 'MODA', 'MPOL', 'MPT', 'MPTE', 'MTUR', 'MZHE', 'MZPU', 'NEME', 'NOSK', 'OBPP', 'OILK', 'OKTA', 'OMOS', 'OPFO', 'OPTK', 'ORAN', 'OSPO', 'OTEK', 'PELK', 'PGGV', 'PKB', 'POPK', 'PPIV', 'PROD', 'PROT', 'PTRS', 'RADE', 'REPL', 'RIMI', 'RINS', 'RZEK', 'RZIT', 'RZIZ', 'RZLE', 'RZLV', 'RZTK', 'RZUG', 'RZUS', 'SBT', 'SDOM', 'SIL', 'SKON', 'SKP', 'SLAV', 'SNBT', 'SNBTO', 'SOLN', 'SPAZ', 'SPAZP', 'SPOL', 'SSPR', 'STB', 'STBP', 'STIL', 'STOK', 'TAJM', 'TASK', 'TBKO', 'TEAL', 'TEHN', 'TEL', 'TETE', 'TIGA', 'TIKV', 'TKPR', 'TKVS', 'TNB', 'TRDB', 'TRPS', 'TRUB', 'TSMP', 'TSZS', 'TTK', 'UNI', 'USJE', 'VARG', 'VFPM', 'VITA', 'VROS', 'VSC', 'VTKS', 'ZAS', 'ZILU', 'ZILUP', 'ZIMS', 'ZKAR', 'ZPKO', 'ZPOG', 'ZSIL', 'ZUAS']:
        # Check if a prediction model already exists for the stock
        if not StockPredictionModel.objects.filter(stock=stock).exists():
            model = StockPredictionModel(
                stock=stock,
                last_modified=today,
                created_at=datetime.datetime.now(),
                cloud_key=f"Models/{stock.code}.keras",
                last_used_for_prediction=None  # Set to None if not available
            )
            model.save()
            print(f"Created model for stock: {stock.code}")
        else:
            print(f"Model already exists for stock: {stock.code}")

if __name__ == "__main__":
    populate_models_db()
