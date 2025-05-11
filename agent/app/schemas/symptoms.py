from pydantic import BaseModel, conint
from typing import Literal

class HealthData(BaseModel):
    gender: Literal["male", "female"]
    age: conint(ge=0)
    smoking: bool
    yellowFingers: bool
    anxiety: bool
    peerPressure: bool
    chronicDisease: bool
    fatigue: bool
    allergy: bool
    wheezing: bool
    alcohol: bool
    coughing: bool
    shortnessOfBreath: bool
    swallowingDifficulty: bool
    chestPain: bool