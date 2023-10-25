"""
The script below focuses on initialising new mortgage
portfolio. Helper functions are created for determining
probability of default, the floor plan an individual
is likely to want based on their income, etc.
Note that all income amounts refer to net income amounts.
Please also note that the code can and should be extended
to include other helper functions (a house's region for
instance should be simulated) and better models.
"""

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
EPC_DISTRIBUTION_ADJ = [0.3, 0.3, 0.2, 0.1, 0.07, 0.02, 0.01]  # impact of income on epc distribution
EPC_SCORE_MAP = {
    'A': [0, 85], 'B': [85, 170], 'C': [170, 255], 'D': [255, 340],
    'E': [340, 425], 'F': [425, 510], 'G': [510, 1000]
}
# Define link between epc label and house value:
EPC_HOUSE_PRICE_MULTIPLIERS = {
    'A': 1.0, 'B': 0.98, 'C': 0.955, 'D': 0.925, 'E': 0.89, 'F': 0.84, 'G': 0.8
}
# Define arbitrary energy costs per EPC label and per sqm.
# These can be adjusted to reflect actual costs.
EPC_ENERGY_COST_MULTIPLIERS = {'A': 1, 'B': 1.5, 'C': 2.0, 'D': 2.5, 'E': 3., 'F': 3.6, 'G': 4.5}
ENERGY_COST_PER_SQM = 10  # Arbitrary cost per square meter
PRICE_PER_SQM = 2000  # Price per sqm in EUR in Belgium

# DEFINE PROPERTY ATTRIBUTES
MEAN_FLOOR_AREA = 100  # Mean floor area in sqm in Belgium
MINIMUM_FLOOR_AREA = 20
MINIMUM_PROPERTY_PRICE = 40000

# DEFINE RISK FACTORS
RISK_FREE_RATE = 0.03
PD_BETA = 0.05  # Multiplies the prob of default when determining credit_spread
DURATION_BETA = 0.0002

# DEFINE GREEN MORTGAGE PARAMS:
EPC_RENOVATION_MULTIPLIER = {
    'A': 0, 'B': 1.0, 'C': 0.86, 'D': 0.73, 'E': 0.59, 'F': 0.46, 'G': 0.35
}  # multiplies the b2a_renovation cost, meaning that for instance moving from G->F = 35% of b2a cost
B2A_RENOVATION_COST = 20000
GREEN_LOAN_DURATION = 10

# KEYS:
INCOME_KEY = 'Income'
FLOOR_PLAN_KEY = 'Floor_plan'
EPC_LABEL_KEY = 'EPC_Label'
EPC_SCORE_KEY = 'EPC_Score'
PROPERTY_VALUE_KEY = 'Property_Value'
ORIGINAL_PROP_VALUE_KEY = 'Original_Property_Value'
PROPERTY_MULTIPLIER_KEY = 'Property_Value_Multiplier'
ENERGY_EXPENSES_KEY = 'Energy_Expenses'
ORIGINAL_ENERGY_EXP_KEY = 'Original_Energy_Expenses'
YEARS_LEFT_KEY = 'Years_Left'
LOAN_VALUE_KEY = 'Loan_Value'
PROB_OF_DEFAULT_KEY = 'Probability_of_Default'
YEARLY_INTEREST_CHARGE_KEY = 'Yearly_Interest_Charge'
YEARLY_PRINCIPAL_REPAYMENT_KEY = 'Yearly_Principal_Repayment'
ENERGY_EXPENSE_TO_INCOME_KEY = 'Energy_Charge_to_Income'
DEBT_TO_INCOME_KEY = 'Debt_to_Income'
LTV_KEY = 'LTV'
RISK_FREE_RATE_KEY = 'RiskFree_Rate'
GREEN_VALUE_KEY = 'Green_Loan_Value'
GREEN_YEARS_LEFT_KEY = 'Green_Years_Left'
GREEN_YEARLY_REPAYMENT_KEY = 'Green_Yearly_Principal_Repayment'
GREEN_INTEREST_CHARGE_KEY = 'Green_Yearly_Interest_Charge'


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
        distribution_adj (List[float]): EPC distribution adjustment factors linked to income.
            Move the bulk of the distribution left, increasing the chance of better EPC
            for afluent borrowers.
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
    energy_cost_epc_multiplier: Dict[str, float] = EPC_ENERGY_COST_MULTIPLIERS,
) -> float:
    """
    Calculate energy expenses based on the EPC label and floor plan of a property.

    Parameters:
        epc_label (str): The EPC label of the property
        floor_plan (float): The floor plan of the property
        energy_cost_per_sqm (float): The base energy rate per sqm.
        energy_cost_epc_multiplier (Dict): Specifies multipliers for the base rate (per EPC label)

    Returns:
        float: The calculated energy expenses
    """
    return energy_cost_epc_multiplier[epc_label] * energy_cost_per_sqm * floor_plan


def get_property_val_multiplier(
    floor_plan: float,
) -> float:
    """
    Determine a random property value multiplier which
    scales with the floor plan. Idea is that larger homes
    can grow in value more than linearly.
    :return: property value multiplier
    """
    value_multiplier = 0.9 + (
            0.1 * np.random.lognormal(floor_plan / 1e3, 0.5)  # random element
    )
    return value_multiplier


def calculate_property_value(
    epc_label: str,
    floor_plan: float,
    epc_house_multiplier: Dict[str, float] = EPC_HOUSE_PRICE_MULTIPLIERS,
    price_per_sqm: float = PRICE_PER_SQM,
    minimum_price: float = MINIMUM_PROPERTY_PRICE,
    minimum_floor_plan: float = MINIMUM_FLOOR_AREA,
    value_multiplier: float = None
) -> float:
    """
    Calculate the property value based on the EPC score and floor plan.
    Assumes a random factor and price per m^2 based on EPC score.

    Parameters:
        epc_label (str): EPC label of the property.
        floor_plan (float): Floor plan area in m^2.
        epc_house_multiplier (dict): Specifies the relationship (multiplier) between the EPC label and house price.
        price_per_sqm (float): Mean price per square meter.
        minimum_price (float): Minimum property price
        minimum_floor_plan (float): minimum property size in square meters.
        value_multiplier (float): random factor used to simulate differences in area costs.

    Returns:
        float: The estimated property value
    """
    epc_factor = epc_house_multiplier[epc_label]
    property_value = value_multiplier * (
            minimum_price + price_per_sqm * (floor_plan - minimum_floor_plan) * epc_factor
    )
    return property_value


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
    loan_to_value_ratio: float,
    risk_free_rate: float,
    energy_expense_beta: float = 0.4,
    ltv_beta: float = 0.1,
    debt_to_income_beta: float = 0.05,
) -> float:
    """
    Calculate the probability of default based on the total-debt-to-income ratio,
    annual energy expenses, loan-to-value ratio, and annual risk-free rates.
    Uses a logistic regression model with random noise.

    Parameters:
        total_debt_to_income (float): The debt-to-income ratio.
        energy_expenses_to_income (float): annual energy expenses as fraction of yearly income.
        loan_to_value_ratio (float): The loan-to-value ratio.
        risk_free_rate (float): The risk-free rate between 0-1.
        energy_expense_beta: multiplies the energy_expense_to_income.
            Note that a positive value means that a higher
            energy-expense-to-income leads to a higher probability of default.
        ltv_beta: multiplies the ltv.
            Note that a positive value means that a higher ltv
             leads to a higher probability of default.
        debt_to_income_beta: multiplies the debt-to-income.
            Note that a positive value means that a higher
            debt-to-income leads to a higher probability of default.
    Returns:
        float: The estimated probability of default.
    """

    # Calculate default factor with reduced weights
    default_factor = -5.08 + \
        debt_to_income_beta * total_debt_to_income + \
        energy_expense_beta * energy_expenses_to_income + \
        ltv_beta * loan_to_value_ratio + \
        (total_debt_to_income * risk_free_rate * 5)

    # Apply sigmoid function to map to a value between 0 and 1
    probability_of_default = 1 / (1 + np.exp(-default_factor))

    return probability_of_default


def calculate_interest_charge(
    prob_default: float, loan_duration: int,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Calculate the annual nominal interest charge for a loan.
    Assumes a credit spread proportional to the probability of default and the loan duration.

    Parameters:
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

    return risk_free_rate + credit_risk_spread + duration_spread


def get_cost_of_green_mortgage(
        floor_plan, epc_label,
        b2a_renovation_cost: float = B2A_RENOVATION_COST,
        renovation_multipliers: Dict[str, float] = EPC_RENOVATION_MULTIPLIER
):
    """
    Determine cost of moving down one epc label (B -> A for instance).
    :param floor_plan: The floor plan/area of the property
    :param epc_label: EPC label of the property.
    :param b2a_renovation_cost: renovation cost for B->A for a 100 m2 property
    :param renovation_multipliers: Multiplies the b2a renovation cost to get the
        costs of one label movements for other EPCs.
    :return:
    """
    renovation_multiplier_ = renovation_multipliers[epc_label]
    green_loan_value = (renovation_multiplier_ * b2a_renovation_cost) * (floor_plan/100)
    return green_loan_value


def issue_green_mortgage(
    row: pd.Series,
    b2a_renovation_cost: float = B2A_RENOVATION_COST,
    price_per_sqm: float = PRICE_PER_SQM,
    energy_cost_per_sqm: float = ENERGY_COST_PER_SQM,
    loan_duration: int = GREEN_LOAN_DURATION,
    energy_expense_beta: float = 0.4,
    ltv_beta: float = 0.1,
    debt_to_income_beta: float = 0.05,
    is_deny: bool = False,
) -> pd.Series:
    """
    Issue a green mortgage and update loan details

    Args:
    row: pandas Series with mortgage details. Must have:
        floor_plan, epc_label, risk_free_rate, property value multiplies and income
    b2a_renovation_cost (float): renovation cost for B->A for a 100 m2 property
    price_per_sqm (float): Mean property price per square meter.
    energy_cost_per_sqm (float): The base energy rate per sqm.
    loan_duration (int): duration of green loan.
    energy_expense_beta: multiplies the energy_expense_to_income.
        Note that a positive value means that a higher
        energy-expense-to-income leads to a higher probability of default.
    ltv_beta: multiplies the ltv.
        Note that a positive value means that a higher ltv
         leads to a higher probability of default.
    debt_to_income_beta: multiplies the debt-to-income.
        Note that a positive value means that a higher
        debt-to-income leads to a higher probability of default.
    is_deny: if True, only issues green mortgage when issuing the mortgage
        decreases the persons probability of default

    Returns:
    row: pandas Series with updated details of the mortgage
    """
    if row[GREEN_VALUE_KEY] > 0:
        print(
            "Cannot issue a green bond before previous loan has been repayed"
        )
        return row

    # TODO: Refactor avoiding lazy solution for resetting values
    original_row = row.copy()

    # Assume EPC score improves by one level and property value consequently increases
    green_loan_value = get_cost_of_green_mortgage(
        row[FLOOR_PLAN_KEY], row[EPC_LABEL_KEY],
        b2a_renovation_cost=b2a_renovation_cost,
    )
    row[GREEN_YEARS_LEFT_KEY] = loan_duration
    row[GREEN_VALUE_KEY] += green_loan_value
    row[GREEN_YEARLY_REPAYMENT_KEY] = green_loan_value / loan_duration

    # Adjust tracked variables
    row[EPC_LABEL_KEY] = chr(
        ord(row[EPC_LABEL_KEY]) - 1
    ) if row[EPC_LABEL_KEY] not in ['A', 'B'] else 'A'
    row[PROPERTY_VALUE_KEY] = calculate_property_value(
        row[EPC_LABEL_KEY], row[FLOOR_PLAN_KEY],
        price_per_sqm=price_per_sqm,
        value_multiplier=row[PROPERTY_MULTIPLIER_KEY]
    )
    row[ENERGY_EXPENSES_KEY] = calculate_energy_expenses(
        epc_label=row[EPC_LABEL_KEY], floor_plan=row[FLOOR_PLAN_KEY],
        energy_cost_per_sqm=energy_cost_per_sqm,
    )
    row[ENERGY_EXPENSE_TO_INCOME_KEY] = row[ENERGY_EXPENSES_KEY] / row[INCOME_KEY]
    total_debt = row[LOAN_VALUE_KEY] + row[GREEN_VALUE_KEY]
    row[LTV_KEY] = total_debt / row[PROPERTY_VALUE_KEY]
    row[DEBT_TO_INCOME_KEY] = total_debt / row[INCOME_KEY]

    row[PROB_OF_DEFAULT_KEY] = calculate_probability_of_default(
        total_debt_to_income=row[DEBT_TO_INCOME_KEY],
        energy_expenses_to_income=row[ENERGY_EXPENSE_TO_INCOME_KEY],
        loan_to_value_ratio=row[LTV_KEY],
        risk_free_rate=row[RISK_FREE_RATE_KEY],
        energy_expense_beta=energy_expense_beta,
        ltv_beta=ltv_beta,
        debt_to_income_beta=debt_to_income_beta,
    )

    # RESET VALUES IF GREEN LOAN INCREASES THE PROB OF DEFAULT
    if is_deny:
        # DO not issue loan if it increases prob of default
        if original_row[PROB_OF_DEFAULT_KEY] < row[PROB_OF_DEFAULT_KEY]:
            return original_row

    row[GREEN_INTEREST_CHARGE_KEY] = calculate_interest_charge(
        prob_default=row[PROB_OF_DEFAULT_KEY],
        loan_duration=row[GREEN_LOAN_DURATION],
        risk_free_rate=row[RISK_FREE_RATE_KEY],
    )
    return row


def generate_mortgage(
    mean_income: float = MEAN_INCOME,
    minimum_income: float = MINIMUM_INCOME,
    mean_floor_area: int = MEAN_FLOOR_AREA,
    minimum_floor_area: int = MINIMUM_FLOOR_AREA,
    base_distribution: List[float] = EPC_DISTRIBUTION,
    energy_cost_per_sqm: float = ENERGY_COST_PER_SQM,
    energy_cost_epc_multiplier: Dict[str, float] = EPC_ENERGY_COST_MULTIPLIERS,  # TODO: rename
    price_per_sqm: float = PRICE_PER_SQM,
    minimum_price: float = MINIMUM_PROPERTY_PRICE,
    minimum_floor_plan: float = MINIMUM_FLOOR_AREA,
    risk_free_rate: float = RISK_FREE_RATE,
    init_ltv: float = MAX_LTV,
    loan_duration: int = LOAN_DURATION_YEARS,
    energy_expense_beta: float = 0.4,
    ltv_beta: float = 0.1,
    debt_to_income_beta: float = 0.05,
) -> dict or None:
    """
    Generate a single mortgage.

    Parameters:
        mean_income (float): mean income in EUR
        minimum_income (float): Minimum yearly salary.
        mean_floor_area (int): mean floor area in square meters
        minimum_floor_area (int): minimum floor area in square meters
        energy_cost_per_sqm: The base energy rate per sqm.
        energy_cost_epc_multiplier: Specifies multipliers for the base energy rate (per EPC label)
        base_distribution (List[float]): EPC distribution of houses in the population.
        price_per_sqm (float): Mean property price per square meter.
        minimum_price (float): Minimum property price
        minimum_floor_plan (float): minimum property size in square meters.
        init_ltv: initial loan to value ratio: [0-3].
        loan_duration: loan duration (from start) in years.
        risk_free_rate (float): risk free rate [0-1].
        energy_expense_beta: multiplies the energy_expense_to_income.
            Note that a positive value means that a higher
            energy-expense-to-income leads to a higher probability of default.
        ltv_beta: multiplies the ltv.
            Note that a positive value means that a higher ltv
             leads to a higher probability of default.
        debt_to_income_beta: multiplies the debt-to-income.
            Note that a positive value means that a higher
            debt-to-income leads to a higher probability of default.

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
        )
    )
    epc_score = np.random.randint(
        EPC_SCORE_MAP[epc_label][0], EPC_SCORE_MAP[epc_label][1] + 1
    )
    prop_value_multiplier = get_property_val_multiplier(floor_plan)
    property_value = calculate_property_value(
        epc_label, floor_plan,
        value_multiplier=prop_value_multiplier,
        price_per_sqm=price_per_sqm,
        minimum_price=minimum_price,
        minimum_floor_plan=minimum_floor_plan,
    )
    energy_expenses = calculate_energy_expenses(
        epc_label, floor_plan,
        energy_cost_per_sqm=energy_cost_per_sqm,
        energy_cost_epc_multiplier=energy_cost_epc_multiplier,
    )
    energy_expense_to_income = energy_expenses / income

    loan_value = calculate_loan_value(property_value, ltv=init_ltv)
    debt_to_income = loan_value / income
    probability_of_default = calculate_probability_of_default(
        debt_to_income, energy_expense_to_income,
        loan_to_value_ratio=init_ltv,
        risk_free_rate=risk_free_rate,
        energy_expense_beta=energy_expense_beta,
        ltv_beta=ltv_beta,
        debt_to_income_beta=debt_to_income_beta,
    )

    yearly_interest_charge = calculate_interest_charge(
        probability_of_default, loan_duration,
        risk_free_rate=risk_free_rate,
    )
    yearly_principal_repayment = calculate_principal_repayment_charge(
        loan_value, loan_duration=loan_duration
    )
    yearly_total_repayment = (yearly_interest_charge * loan_value) + yearly_principal_repayment

    if (yearly_total_repayment + energy_expenses) < 0.8 * income:
        return {
            INCOME_KEY: income,
            FLOOR_PLAN_KEY: floor_plan,
            EPC_LABEL_KEY: epc_label,
            EPC_SCORE_KEY: epc_score,
            PROPERTY_VALUE_KEY: property_value,
            ORIGINAL_PROP_VALUE_KEY: property_value,
            PROPERTY_MULTIPLIER_KEY: prop_value_multiplier,
            ENERGY_EXPENSES_KEY: energy_expenses,
            ORIGINAL_ENERGY_EXP_KEY: energy_expenses,
            YEARS_LEFT_KEY: loan_duration,
            LOAN_VALUE_KEY: loan_value,
            PROB_OF_DEFAULT_KEY: probability_of_default,
            YEARLY_INTEREST_CHARGE_KEY: yearly_interest_charge,
            YEARLY_PRINCIPAL_REPAYMENT_KEY: yearly_principal_repayment,
            ENERGY_EXPENSE_TO_INCOME_KEY: energy_expense_to_income,
            DEBT_TO_INCOME_KEY: debt_to_income,
            LTV_KEY: init_ltv,
            RISK_FREE_RATE_KEY: risk_free_rate,
            GREEN_VALUE_KEY: 0,
            GREEN_YEARS_LEFT_KEY: 0,
            GREEN_YEARLY_REPAYMENT_KEY: 0,
        }
    else:
        return None


def generate_n_mortgages(
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

    df_portfolio = pd.DataFrame()
    index = list(range(n_mortgages))
    while len(df_portfolio) < n_mortgages:
        new_row = generate_mortgage(*args, **kwargs)
        if new_row is not None:
            df_portfolio = pd.concat(
                [df_portfolio, pd.DataFrame(new_row, index=[index.pop()])],
                ignore_index=True,
            )

    return df_portfolio


def generate_mortgage_portfolio(
    portfolio_value: float,
    *args, **kwargs,
) -> pd.DataFrame:
    """
    Generate a portfolio of mortgages with a value of almost portfolio_value.

    Parameters:
        portfolio_value (float): Value (cum sum) of mortgage portfolio to create.
        *args, **kwargs: see generate_mortgage documentation

    Returns:
        pd.DataFrame: A dataframe containing the generated mortgage portfolio.
    """

    df_portfolio = pd.DataFrame()
    index = 0
    while df_portfolio[LOAN_VALUE_KEY].sum() < portfolio_value:
        new_row = generate_mortgage(*args, **kwargs)
        if new_row is not None:
            df_portfolio = pd.concat(
                [df_portfolio, pd.DataFrame(new_row, index=[index])],
                ignore_index=True,
            )
            index += 1

    return df_portfolio.iloc[:-1]  # don't return last row as this probably exceeds portfolio_value


# if __name__ == "__main__":
portfolio = generate_n_mortgages()
for col in portfolio.columns:
    print(col, portfolio[col].min(), portfolio[col].max())
