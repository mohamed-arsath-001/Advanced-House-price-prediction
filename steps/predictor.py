import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    service.start(timeout=10)
    data = json.loads(input_data)

    data.pop("columns", None) 
    data.pop("index", None)  

    expected_columns = [
        "Order",
        "PID",
        "MS SubClass",
        "Lot Frontage",
        "Lot Area",
        "Overall Qual",
        "Overall Cond",
        "Year Built",
        "Year Remod/Add",
        "Mas Vnr Area",
        "BsmtFin SF 1",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "1st Flr SF",
        "2nd Flr SF",
        "Low Qual Fin SF",
        "Gr Liv Area",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
        "Full Bath",
        "Half Bath",
        "Bedroom AbvGr",
        "Kitchen AbvGr",
        "TotRms AbvGrd",
        "Fireplaces",
        "Garage Yr Blt",
        "Garage Cars",
        "Garage Area",
        "Wood Deck SF",
        "Open Porch SF",
        "Enclosed Porch",
        "3Ssn Porch",
        "Screen Porch",
        "Pool Area",
        "Misc Val",
        "Mo Sold",
        "Yr Sold",
    ]
    df = pd.DataFrame(data["data"], columns=expected_columns)

    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    prediction = service.predict(data_array)

    return prediction
