import pandas as pd
import numpy as np

from typing import List, Dict

# Constants:
MAX_LTV = 0.8  # Loan-to-value ratio
LOAN_DURATION_YEARS = 30  # Assuming fixed 30 years loans
MEAN_INCOME = 50000  # Mean salary in EUR in Belgium
MINIMUM_INCOME = 10000  # Minimum salary in EUR in Belgium

# EPC score distribution (assumed, replace with real data if available)
EPC_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
EPC_DISTRIBUTION = [0.05, 0.15, 0.3, 0.2, 0.15, 0.1, 0.05]
EPC_DISTRIBUTION_ADJ = [0.3, 0.3, 0.2, 0.1, 0.07, 0.02, 0.01]
EPC_SCORE_MAP = {
    'A': [0, 85], 'B': [85, 170], 'C': [170, 255], 'D': [255, 340],
    'E': [340, 425], 'F': [425, 510], 'G': [510, 1000]
}
# Define arbitrary energy costs per EPC label and per sqm.
# These can be adjusted to reflect actual costs.
EPC_ENERGY_COST_MULTIPLIERS = {'A': 1, 'B': 1.5, 'C': 2.0, 'D': 2.5, 'E': 3., 'F': 3.6, 'G': 4.5}
ENERGY_COST_PER_SQM = 10  # Arbitrary cost per square meter
PRICE_PER_SQM = 2000  # Price per sqm in EUR in Belgium

MEAN_FLOOR_AREA = 100  # Mean floor area in sqm in Belgium
MINIMUM_FLOOR_AREA = 20
MINIMUM_PROPERTY_PRICE = 40000

RISK_FREE_RATE = 0.03
PD_BETA = 0.05  # Multiplies the prob of default when determining credit_spread
DURATION_BETA = 0.0002


def calculate_income(
    income_mean: float = MEAN_INCOME,
    minimum_income: float = MINIMUM_INCOME
) -> float:
    """
    Calculate income using a lognormal distribution.

    Parameters:
        income_mean (float): Mean yearly salary.
        minimum_income (float): Minimum yearly salary.

    Returns:
        float: the simulated income
    """
    return minimum_income + np.random.lognormal(
        mean=np.log(income_mean - minimum_income),
        sigma=1
    )


def calculate_floor_area(
    income: float,
    mean_income: float = MEAN_INCOME,
    mean_floor_area: int = MEAN_FLOOR_AREA,
    minimum_floor_area: int = MINIMUM_FLOOR_AREA
) -> int:
    """
    Simulates floor area as a function of income.

    Parameters:
        income (float): The income value.
        mean_income (float): mean income in EUR
        mean_floor_area (int): mean floor area in square meters
        minimum_floor_area (int): minimum floor area in square meters

    Returns:
        int: The generated floor area.
    """
    # calculate floor area as a linear function of income
    random_factor = 0.9 + (0.1 * np.random.lognormal(0, 0.25))  # random element
    floor_area = minimum_floor_area + random_factor * (
            (np.sqrt(income) / np.sqrt(mean_income)) * (mean_floor_area - minimum_floor_area)
    )
    return int(floor_area)


def get_epc_distribution(
    income: float,
    base_distribution: List[float] = EPC_DISTRIBUTION,
    distribution_adj: List[float] = EPC_DISTRIBUTION_ADJ,
) -> List[float]:
    """
    Adjusts the EPC label distribution based on income.
    Higher income leads to a higher chance of a better EPC label.

    Parameters:
        income (float): The income for which to adjust the EPC distribution.
        base_distribution (List[float]): EPC distribution of houses in the population.
        distribution_adj (List[float]): EPC distribution adjustment factors.
            Move the bulk of the distribution left, increasing the chance of better EPC.
            Note the same adjustment is applied for time as for income.
    Returns:
        List[float]: The adjusted EPC label distribution.
    """

    if len(base_distribution) != len(EPC_SCORE_MAP):
        raise ValueError(
            f'Length of base_distribution: {len(base_distribution)} '
            f'should match the length of EPC_LABELS: {len(EPC_LABELS)}'
        )

    # Use income and time to modify the distribution.
    # Here we assume an arbitrary relationship where higher incomes
    # lead to better EPC scores.
    # This can be adjusted to reflect actual statistical relationships.
    adj_factor = (income / 10000)  # Arbitrary scaling factor

    # Initialize adjusted distribution with base distribution
    adjusted_distribution = np.array(base_distribution)

    # Perform distribution adjustment
    adjusted_distribution += (adj_factor * np.array(distribution_adj))

    # Make sure the distribution sums to 1
    adjusted_distribution /= np.sum(adjusted_distribution)

    return adjusted_distribution.tolist()


def calculate_energy_expenses(
    epc_label: str, floor_plan: float,
    energy_cost_per_sqm: float = ENERGY_COST_PER_SQM,
    energy_cost_multiplier: Dict[str, float] = EPC_ENERGY_COST_MULTIPLIERS,
) -> float:
    """
    Calculate energy expenses based on the EPC label and floor plan of a property.

    Parameters:
        epc_label (str): The EPC label of the property
        floor_plan (float): The floor plan of the property
        energy_cost_per_sqm: The base energy rate per sqm.
        energy_cost_multiplier: Specifies multipliers for the base rate (per EPC label)

    Returns:
        float: The calculated energy expenses
    """
    return energy_cost_multiplier[epc_label] * energy_cost_per_sqm * floor_plan


def calculate_property_value(
    epc_label: str,
    floor_plan: float,
    price_per_sqm: float = PRICE_PER_SQM,
    minimum_price: float = MINIMUM_PROPERTY_PRICE,
    minimum_floor_plan: float = MINIMUM_FLOOR_AREA,
) -> float:
    """
    Calculate the property value based on the EPC score and floor plan.
    Assumes a random factor and price per m^2 based on EPC score.

    Parameters:
        epc_label (str): EPC label of the property.
        floor_plan (float): Floor plan area in m^2.
        price_per_sqm (float): Mean price per square meter.
        minimum_price (float): Minimum property price
        minimum_floor_plan (float): minimum property size in square meters.

    Returns:
        float: The estimated property value.
    """
    epc_factor = {'A': 1.0, 'B': 0.98, 'C': 0.955, 'D': 0.925, 'E': 0.89, 'F': 0.84, 'G': 0.8}[epc_label]
    random_factor = 0.9 + (0.1 * np.random.lognormal(floor_plan/1e3, 0.5))  # random element
    return random_factor * (minimum_price + price_per_sqm * (floor_plan - minimum_floor_plan) * epc_factor)


def calculate_loan_value(property_value: float, ltv: float = MAX_LTV) -> float:
    """
    Calculate the loan value based on the property value.
    Assumes a maximum loan-to-value (LTV) ratio.

    Parameters:
        property_value (float): The property value.
        ltv: initial loan to value ratio: [0-3].

    Returns:
        float: The loan value.
    """
    return property_value * ltv


def calculate_principal_repayment_charge(
        loan_value: float, loan_duration: int = LOAN_DURATION_YEARS
) -> float:
    """
    Calculate the annual principal repayment charge for a loan.
    Assumes a fixed loan duration.

    Parameters:
        loan_value (float): The loan value.
        loan_duration: loan duration (from start)

    Returns:
        float: The annual principal repayment charge.
    """
    return loan_value / loan_duration


def calculate_probability_of_default(
    total_debt_to_income: float,
    energy_expenses_to_income: float,
) -> float:
    """
    Calculate the probability of default based on the total-debt-to-income ratio,
    and annual energy expenses. Uses a logistic regression model with random noise.

    Parameters:
        total_debt_to_income (float): The debt-to-income ratio.
        energy_expenses_to_income (float): annual energy expenses as fraction of yearly income.

    Returns:
        float: The estimated probability of default.
    """

    # Calculate default factor with reduced weights
    default_factor = -5 + 0.05 * total_debt_to_income + 0.2 * energy_expenses_to_income

    # Apply sigmoid function to map to a value between 0 and 1
    probability_of_default = 1 / (1 + np.exp(-default_factor))

    return probability_of_default


def calculate_interest_charge(
        loan_value: float, prob_default: float, loan_duration: int,
        risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Calculate the annual nominal interest charge for a loan.
    Assumes a credit spread proportional to the probability of default and the loan duration.

    Parameters:
        loan_value (float): The loan value.
        prob_default (float): The estimated probability of default.
        loan_duration (int): The duration of the loan in years.
        risk_free_rate (float): risk free rate [0-1].

    Returns:
        float: The annual nominal interest charge.
    """
    # We introduce DURATION_BETA that scales the influence of loan duration on the interest charge
    # We modify it to a logarithmic scale to represent the diminishing effect of duration over time
    credit_risk_spread = PD_BETA * prob_default
    duration_spread = DURATION_BETA * np.log(loan_duration + 1)

    return loan_value * (risk_free_rate + credit_risk_spread + duration_spread)


def generate_mortgage(
    mean_income: float = MEAN_INCOME,
    minimum_income: float = MINIMUM_INCOME,
    mean_floor_area: int = MEAN_FLOOR_AREA,
    minimum_floor_area: int = MINIMUM_FLOOR_AREA,
    base_distribution: List[float] = EPC_DISTRIBUTION,
    distribution_adj: List[float] = EPC_DISTRIBUTION_ADJ,
    energy_cost_per_sqm: float = ENERGY_COST_PER_SQM,
    energy_cost_multiplier: Dict[str, float] = EPC_ENERGY_COST_MULTIPLIERS,
    price_per_sqm: float = PRICE_PER_SQM,
    minimum_price: float = MINIMUM_PROPERTY_PRICE,
    minimum_floor_plan: float = MINIMUM_FLOOR_AREA,
    risk_free_rate: float = RISK_FREE_RATE,
    ltv: float = MAX_LTV,
    loan_duration: int = LOAN_DURATION_YEARS,
) -> dict:
    """
    Generate a single mortgage.

    Parameters:
        mean_income (float): mean income in EUR
        minimum_income (float): Minimum yearly salary.
        mean_floor_area (int): mean floor area in square meters
        minimum_floor_area (int): minimum floor area in square meters
        energy_cost_per_sqm: The base energy rate per sqm.
        energy_cost_multiplier: Specifies multipliers for the base rate (per EPC label)
        base_distribution (List[float]): EPC distribution of houses in the population.
        distribution_adj (List[float]): EPC distribution adjustment factors.
            Move the bulk of the distribution left, increasing the chance of better EPC.
            Note the same adjustment is applied for time as for income.
        price_per_sqm (float): Mean price per square meter.
        minimum_price (float): Minimum property price
        minimum_floor_plan (float): minimum property size in square meters.
        ltv: initial loan to value ratio: [0-3].
        loan_duration: loan duration (from start) in years.
        risk_free_rate (float): risk free rate [0-1].

    Returns:
        dict: A dictionary representing a mortgage.
    """
    income = calculate_income(
        income_mean=mean_income, minimum_income=minimum_income
    )

    floor_plan = calculate_floor_area(
        income, mean_income=mean_income, mean_floor_area=mean_floor_area,
        minimum_floor_area=minimum_floor_area
    )
    epc_label = np.random.choice(
        EPC_LABELS,
        p=get_epc_distribution(
            income,
            base_distribution=base_distribution,
            distribution_adj=distribution_adj,
        )
    )
    epc_score = np.random.randint(EPC_SCORE_MAP[epc_label][0], EPC_SCORE_MAP[epc_label][1] + 1)
    property_value = calculate_property_value(
        epc_label, floor_plan,
        price_per_sqm=price_per_sqm,
        minimum_price=minimum_price,
        minimum_floor_plan=minimum_floor_plan,
    )
    energy_expenses = calculate_energy_expenses(
        epc_label, floor_plan,
        energy_cost_per_sqm=energy_cost_per_sqm,
        energy_cost_multiplier=energy_cost_multiplier,
    )
    energy_expense_to_income = energy_expenses / income

    loan_value = calculate_loan_value(property_value, ltv=ltv)
    debt_to_income = loan_value / income
    probability_of_default = calculate_probability_of_default(debt_to_income, energy_expense_to_income)
    yearly_nominal_interest_charge = calculate_interest_charge(
        loan_value, probability_of_default, loan_duration,
        risk_free_rate=risk_free_rate,
    )
    yearly_principal_repayment = calculate_principal_repayment_charge(
        loan_value, loan_duration=loan_duration
    )
    yearly_total_repayment = yearly_nominal_interest_charge + yearly_principal_repayment
    debt_expense_to_income = yearly_total_repayment / income
    ltv = loan_value / property_value

    return {
        'Income': income,
        'Floor_plan': floor_plan,
        'EPC_Label': epc_label,
        'EPC_Score': epc_score,
        'Property_Value': property_value,
        'Energy_Expenses': energy_expenses,
        'Years_Left': loan_duration,
        'Loan_Value': loan_value,
        'Probability_of_Default': probability_of_default,
        'Yearly_Nominal_Interest_Charge': yearly_nominal_interest_charge,
        'Yearly_Principal_Repayment': yearly_principal_repayment,
        'Yearly_Total_Repayment': yearly_total_repayment,
        'Energy_Charge_to_Income': energy_expense_to_income,
        'Debt_Charge_to_Income': debt_expense_to_income,
        'Debt_to_Income': debt_to_income,
        'LTV': ltv,
    }


def generate_portfolio(
    n_mortgages: int = 100000,
    *args, **kwargs,
) -> pd.DataFrame:
    """
    Generate a portfolio of mortgages.

    Parameters:
        n_mortgages (int): The number of mortgages to generate.
        *args, **kwargs: see generate_mortgage documentation

    Returns:
        pd.DataFrame: A dataframe containing the generated mortgage portfolio.
    """
    return pd.DataFrame([generate_mortgage(*args, **kwargs) for _ in range(n_mortgages)])


if __name__ == "__main__":
    portfolio = generate_portfolio()
    for col in portfolio.columns:
        print(col, portfolio[col].min(), portfolio[col].max())
