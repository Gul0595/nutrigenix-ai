from dataclasses import dataclass
from typing import List


@dataclass
class Supplement:
    name: str
    dose: str
    reason: str
    evidence: list


class SupplementAgent:

    def generate_protocol(self, deficiencies: List[str]):

        protocol = []

        for d in deficiencies:

            if d == "iron_deficient":

                protocol.append(
                    Supplement(
                        name="Iron Bisglycinate",
                        dose="30 mg daily",
                        reason="Improves ferritin and hemoglobin levels",
                        evidence=[]
                    )
                )

                protocol.append(
                    Supplement(
                        name="Vitamin C",
                        dose="250 mg daily",
                        reason="Enhances iron absorption",
                        evidence=[]
                    )
                )

            if d == "vitamin_d_deficient":

                protocol.append(
                    Supplement(
                        name="Vitamin D3",
                        dose="2000 IU daily",
                        reason="Restores optimal vitamin D levels",
                        evidence=[]
                    )
                )

        return protocol