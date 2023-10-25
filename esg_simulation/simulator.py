import pandas as pd
import numpy as np

from typing import List, Dict, Any, Tuple

from .initialisor import (
    calculate_probability_of_default,
    calculate_energy_expenses,
    calculate_property_value,
    generate_mortgage_portfolio,
    issue_green_mortgage,
    LOAN_VALUE_KEY, YEARS_LEFT_KEY,
    PROB_OF_DEFAULT_KEY, PROPERTY_VALUE_KEY,
    INCOME_KEY, ENERGY_EXPENSES_KEY,
    YEARLY_PRINCIPAL_REPAYMENT_KEY,
    YEARLY_INTEREST_CHARGE_KEY,
    PRICE_PER_SQM,
    ENERGY_COST_PER_SQM,
    EPC_LABEL_KEY,
    EPC_DISTRIBUTION,
    FLOOR_PLAN_KEY,
    PROPERTY_MULTIPLIER_KEY,
    LTV_KEY,
    DEBT_TO_INCOME_KEY,
    ENERGY_EXPENSE_TO_INCOME_KEY,
    RISK_FREE_RATE_KEY,
    GREEN_VALUE_KEY,
    GREEN_YEARS_LEFT_KEY,
    GREEN_INTEREST_CHARGE_KEY,
    GREEN_YEARLY_REPAYMENT_KEY,
    MEAN_INCOME,
    B2A_RENOVATION_COST,
)


# Constants:
MAX_SALE_LOSS = 0.4  # max fraction of value lost during sale.
MIN_SALE_LOSS = 0.1  # min fraction of value lost during sale.
PREPAYMENT_PROBABILITY = 0.03  # rate at which prepayments happen [0-1]
INCOME_GROWTH = 0.02  # rate at which income grows [0-1]
GREEN_FUNDS_DECAY = 0.5
MORTGAGE_FUNDS_DECAY = 0.7
EPCS_NOT_FOR_GREEN = ['A']


def get_total_debt(portfolio: pd.DataFrame) -> pd.Series:
    total_debt = portfolio[LOAN_VALUE_KEY] + portfolio[GREEN_VALUE_KEY]
    return total_debt


def get_income(
    income: float,
    prob_default: float,
) -> float:
    """
    Get income. Assume default manifests as loss of all income

    Parameters:
        income (float): Current income.
        prob_default (float): Probability of default

    Returns:
        float: Updated income.
    """

    if np.random.random() < prob_default:  # assume default manifests as loss of all income
        income = 0

    return income


def get_funds(
        total_funds: float,
        fraction: float,
        risk_free_rate: float,
        c: float = 1.0
) -> float:
    """
    Calculate the amount of funds to lend out as green loans in a stochastic manner.

    Parameters:
    - total_funds (float): Total amount of funds available.
    - fraction (float): The fraction of total funds to be lent out as specific loans.
    - risk_free_rate (float): The risk-free rate of return.
    - c (float): Constant to adjust the decay based on risk-free rate.

    Returns:
    - float: Stochastically determined green loan funds.

    Note:
    The distribution is designed such that the maximum possible value (given by total_funds * green_fraction)
    has the highest probability, and smaller values can also be selected with decreasing probability.
    """
    mean_green_funds = total_funds * fraction

    alpha = np.exp(-risk_free_rate)
    beta = np.exp(-risk_free_rate) + c

    stochastic_factor = np.random.beta(alpha, beta)
    return mean_green_funds * stochastic_factor


def update_income(
    income: float,
    income_growth_rate: float,
) -> float:
    """
    Update the income value at the end of a year.

    Parameters:
        income (float): Current income.
        income_growth_rate (float): average rate at which income grows [0-1].

    Returns:
        float: Updated income.
    """

    random_factor = np.random.lognormal(income_growth_rate, 0.1)
    income *= 1 + random_factor + np.random.uniform(-0.05, 0.05)
    return income


def simulate_default(
    income: float,
    expenses: float,
    interest_charge: float,
    principal_repayment_charge: float,
    loan_value: float,
    property_value: float,
    min_sale_loss: float = MIN_SALE_LOSS,
    max_sale_loss: float = MAX_SALE_LOSS
) -> float:
    """
    Simulate the default event

    Args:
    loan_value: current value of the loan
    property_value: current value of the property

    Returns:
        Default repayment. If 0 no default was observed.
    """
    if income < expenses + interest_charge + principal_repayment_charge:  # if unable to service obligations
        sale_loss = np.random.uniform(min_sale_loss, max_sale_loss)
        value_kept = 1 - sale_loss
        return min(value_kept * property_value, loan_value)  # cannot get more than loan value
    else:  # else no default
        return 0


def prepay_loan(
        loan_value: float,
        risk_free_rate: float,
        prepayment_prob: float = PREPAYMENT_PROBABILITY,
) -> float:
    """
    Calculate new loan value after repayment

    Args:
    loan_value: current value of the loan
    risk_free_rate: risk-free rate [0-1]
    prepayment_prob: prepayment probability [0-1]

    Returns:
        Prepayment amount. If 0 no prepayment is observed.
    """

    prepayment_prob -= 0.1 * risk_free_rate  # adjust prepayment prob (higher rf lower prepayments).

    prepayment = np.random.choice(
        [0, 1], p=[1-prepayment_prob, prepayment_prob]
    )
    prepayment_value = loan_value if prepayment else 0
    return prepayment_value


def update_ratios(
    portfolio: pd.DataFrame
) -> None:
    """
        Recalculate LTV, Debt_to_Income, and Energy_Exp_to_Income
        for all loans (not just loans that received green loans).

        Parameters:
            portfolio (pd.DataFrame): The initial mortgage portfolio.
    """
    total_debt = get_total_debt(portfolio)
    portfolio[DEBT_TO_INCOME_KEY] = total_debt / portfolio[INCOME_KEY]
    portfolio[LTV_KEY] = total_debt / portfolio[PROPERTY_VALUE_KEY]
    portfolio[ENERGY_EXPENSE_TO_INCOME_KEY] = portfolio[ENERGY_EXPENSES_KEY] / portfolio[INCOME_KEY]
    return


def update_indicators(
    portfolio: pd.DataFrame, risk_free_rate: float,
    energy_cost_per_sqm: float, price_per_sqm: float,
    income_growth_rate: float
) -> None:
    """
        Updates various 'raw' attributes of the loan portfolio for next year.

        Parameters:
            portfolio (pd.DataFrame): The initial mortgage portfolio.
            risk_free_rate (float) : risk-free rate [0-1] for next year.
            energy_cost_per_sqm (float): The cost of energy per square meter.
            price_per_sqm (float): The average property price per square meter.
            income_growth_rate (float): The growth rate for income.
    """

    # Update Income, Energy Expenses, Property Value, and rates
    portfolio[INCOME_KEY] = portfolio.apply(
        lambda row: update_income(
            income=row[INCOME_KEY],
            income_growth_rate=income_growth_rate,
        ), axis=1
    )
    portfolio[ENERGY_EXPENSES_KEY] = portfolio.apply(
        lambda row: calculate_energy_expenses(
            epc_label=row[EPC_LABEL_KEY],
            floor_plan=row[FLOOR_PLAN_KEY],
            energy_cost_per_sqm=energy_cost_per_sqm
        ), axis=1
    )
    portfolio[PROPERTY_VALUE_KEY] = portfolio.apply(
        lambda row: calculate_property_value(
            epc_label=row[EPC_LABEL_KEY],
            floor_plan=row[FLOOR_PLAN_KEY],
            price_per_sqm=price_per_sqm,
            value_multiplier=row[PROPERTY_MULTIPLIER_KEY]
        ), axis=1
    )
    # Update rates.
    for interest_key in [YEARLY_INTEREST_CHARGE_KEY, GREEN_INTEREST_CHARGE_KEY]:
        portfolio[interest_key] = portfolio.apply(
            lambda row:  max(
                row[interest_key] - (row[RISK_FREE_RATE_KEY] - risk_free_rate),  # higher new rate increases int.charge
                0
            ), axis=1
        )
    portfolio[RISK_FREE_RATE_KEY] = risk_free_rate
    return


def handle_defaults_and_prepayments(
        portfolio: pd.DataFrame
) -> Tuple[float, float]:
    """
        Processes defaults and prepayments for the provided mortgage portfolio.

        This function simulates defaults for each loan in the portfolio based on factors such as energy expenses,
        income, interest charges, principal repayment charges, loan values, and property values. For loans that
        default, the loan value is set to zero.

        The function also simulates prepayments for each loan based on its value and the risk-free rate. For
        loans that prepay, the loan value is set to zero.

        After processing defaults and prepayments, loans with zero value are removed from the portfolio.

        Parameters:
            portfolio (pd.DataFrame): The current mortgage portfolio.

        Returns:
            float: The total sum of all prepayments.
            float: The total sum of all defaults.
    """

    # Handle defaults (assume default on green bond as well when defaulting on mortgage)
    defaults = portfolio.apply(
        lambda row: simulate_default(
            expenses=row[ENERGY_EXPENSES_KEY],
            income=get_income(row[INCOME_KEY], row[PROB_OF_DEFAULT_KEY]),
            interest_charge=(row[YEARLY_INTEREST_CHARGE_KEY] * row[LOAN_VALUE_KEY]),
            principal_repayment_charge=row[YEARLY_PRINCIPAL_REPAYMENT_KEY],
            loan_value=row[LOAN_VALUE_KEY],
            property_value=row[PROPERTY_VALUE_KEY]
        ), axis=1
    )

    # Adjust loan value and green loan value to factor in defaults:
    for loan_key, repayment_key in [
        (LOAN_VALUE_KEY, YEARLY_PRINCIPAL_REPAYMENT_KEY),
        (GREEN_VALUE_KEY, GREEN_YEARLY_REPAYMENT_KEY),
    ]:
        portfolio.loc[defaults != 0, loan_key] = 0
        portfolio.loc[defaults != 0, repayment_key] = 0
    defaults_sum = defaults.sum()

    # Handle prepayments:
    prepayments_sum = 0
    for loan_key in [LOAN_VALUE_KEY, GREEN_VALUE_KEY]:
        prepayments = portfolio.apply(lambda row: prepay_loan(
            loan_value=row[loan_key], risk_free_rate=row[RISK_FREE_RATE_KEY]
        ), axis=1)
        portfolio.loc[prepayments != 0, loan_key] = 0
        prepayments_sum += prepayments.sum()

    return prepayments_sum, defaults_sum


def calculate_total_inflow(
    portfolio: pd.DataFrame, cash_inflow: float,
    prepayments_sum: float, defaults_sum: float
) -> float:
    """
        Computes the total cash inflow to the bank based on various income streams.

        The total cash inflow is calculated as the sum of:
        1. Yearly principal repayments from the portfolio.
        2. Yearly interest charges based on the outstanding loan values.
        3. Yearly repayments for green loans.
        4. Yearly interest charges for green loans based on the outstanding loan values.
        5. Cash inflows from external sources or previous periods.
        6. Amounts from loan prepayments.
        7. Amounts recovered from defaults.

        Parameters:
            portfolio (pd.DataFrame): The current mortgage portfolio.
            cash_inflow (float): Cash inflow from external sources or previous periods.
            prepayments_sum (float): Total sum of all prepayments.
            defaults_sum (float): Total sum recovered from all defaults.

        Returns:
            float: The total computed cash inflow to the bank.
    """
    total_inflow = portfolio[YEARLY_PRINCIPAL_REPAYMENT_KEY].sum() + \
        (portfolio[YEARLY_INTEREST_CHARGE_KEY] * portfolio[LOAN_VALUE_KEY]).sum() + \
        portfolio[GREEN_YEARLY_REPAYMENT_KEY].sum() + \
        (portfolio[GREEN_INTEREST_CHARGE_KEY] * portfolio[GREEN_VALUE_KEY]).sum() + \
        cash_inflow + prepayments_sum + defaults_sum

    # adjust loan values and years left to reflect payments made:
    for loan_key, repayment_key, year_key in [
        (LOAN_VALUE_KEY, YEARLY_PRINCIPAL_REPAYMENT_KEY, YEARS_LEFT_KEY),
        (GREEN_VALUE_KEY, GREEN_YEARLY_REPAYMENT_KEY, GREEN_YEARS_LEFT_KEY),
    ]:
        portfolio.loc[portfolio[loan_key] != 0, loan_key] -= portfolio[repayment_key]
        portfolio.loc[portfolio[year_key] != 0, year_key] -= 1
    return total_inflow


def issue_green_mortgages(
        portfolio: pd.DataFrame,
        green_loan_funds: float,
        b2a_renovation_cost: float,
        price_per_sqm: float,
        energy_cost_per_sqm: float,
        not_eligible_epcs: List[str] = ['A'],
        is_deny: bool = False,
) -> pd.DataFrame:
    """
    Issues green mortgages to eligible loans.

    Parameters:
    - portfolio (pd.DataFrame): DataFrame containing loan details.
    - green_loan_funds (float): Maximum funds available to issue as green loans.
    - b2a_renovation_cost (float): Renovation cost for transitioning from B to A for a 100 sqm property.
    - price_per_sqm (float): Mean property price per square meter.
    - energy_cost_per_sqm (float): Base energy rate per sqm.
    - not_eligible_epcs (List[str]): List of EPC labels that are not eligible for green mortgages.
    - is_deny (bool): If True, only issues a green loan when issuing the loan decreases the person's probability of default.

    Returns:
    - pd.DataFrame: Updated portfolio with issued green mortgages.
    """

    eligible_loans = portfolio[
        ~portfolio[EPC_LABEL_KEY].isin(not_eligible_epcs) & (portfolio[GREEN_VALUE_KEY] == 0)
        ].copy()

    initial_green_value_sum = portfolio[GREEN_VALUE_KEY].sum()

    for idx, row in eligible_loans.iterrows():
        loaned_out_green_funds = portfolio[GREEN_VALUE_KEY].sum() - initial_green_value_sum
        if loaned_out_green_funds >= green_loan_funds:
            break

        updated_row = issue_green_mortgage(
            row,
            b2a_renovation_cost=b2a_renovation_cost,
            price_per_sqm=price_per_sqm,
            energy_cost_per_sqm=energy_cost_per_sqm,
            is_deny=is_deny,
        )
        potential_new_loaned_out_funds = loaned_out_green_funds + updated_row[GREEN_VALUE_KEY]

        if potential_new_loaned_out_funds <= green_loan_funds:
            portfolio.loc[idx] = updated_row

    return portfolio


# TODO: Add car loans
#  and add functionality for missing data in portfolio.
#  Also add stochasticity to loan issuance.
def simulate_year(
    portfolio: pd.DataFrame,
    new_loan_fraction: float = 0.75,
    green_loan_fraction: float = 0.25,
    risk_free_rate: float = 0.03,
    price_per_sqm: float = PRICE_PER_SQM,
    energy_cost_per_sqm: float = ENERGY_COST_PER_SQM,
    income_growth_rate: float = INCOME_GROWTH,
    epc_distribution: List[float] = EPC_DISTRIBUTION,
    cash_inflow: float = 0,
    b2a_renovation_cost: float = B2A_RENOVATION_COST,
    is_deny: bool = False,
    not_eligible_epcs: List[str] = EPCS_NOT_FOR_GREEN,
) -> Dict[str, Any]:
    """
    Simulates a year in the mortgage portfolio. Repay loans, issue new loans and green mortgages,
    and updates the relevant financial and property details.

    Parameters:
        portfolio (pd.DataFrame): The initial mortgage portfolio.
        new_loan_fraction (float, optional): The fraction of available funds to be
         used for new loans. Default is 0.5.
        green_loan_fraction (float, optional): The fraction of available funds to be
         used for green mortgages. Default is 0.5.
        risk_free_rate: risk-free rate [0-1] for next year.
        energy_cost_per_sqm: The base energy rate per sqm for next year.
        price_per_sqm (float): Mean property price per square meter for next year.
        income_growth_rate (float): Mean income growth rate.
        epc_distribution (List[float]): EPC distribution of houses in the population.
        cash_inflow (float): 'extra' cash available for distribution. Either from the
            previous period or from net customer inflows. Note can be negative.
        b2a_renovation_cost (float): Renovation cost for transitioning from B to A for a 100 sqm property.
        not_eligible_epcs (List[str]): List of EPC labels that are not eligible for green mortgages.
        is_deny (bool): If True, only issues a green loan when issuing the loan decreases the person's probability of default.

    Returns:
        Dict:
            'portfolio' : pd.DataFrame - The updated mortgage portfolio after a year
            'cash': float - cash that could not be dispersed.
    """

    prepayments_sum, defaults_sum = handle_defaults_and_prepayments(
        portfolio=portfolio,
    )

    # Determine available funds: note interest charge is a prc and not nominal value
    total_inflow = calculate_total_inflow(
        portfolio=portfolio, cash_inflow=cash_inflow,
        prepayments_sum=prepayments_sum, defaults_sum=defaults_sum,
    )
    new_loan_funds = get_funds(
        total_funds=total_inflow,
        fraction=new_loan_fraction,
        risk_free_rate=risk_free_rate,
        c=MORTGAGE_FUNDS_DECAY
    )
    green_loan_funds = get_funds(
        total_funds=total_inflow,
        fraction=green_loan_fraction,
        risk_free_rate=risk_free_rate,
        c=GREEN_FUNDS_DECAY,
    )

    # UPDATING VALUES IN PORTFOLIO BASED ON NEW ENERGY AND PROPERTY VALUES FOR NEXT YEAR:
    # Note this deos not update LTV, DEBT2INCOME, ENERGY2INCOME, ETC.
    update_indicators(
        portfolio=portfolio,
        energy_cost_per_sqm=energy_cost_per_sqm,
        price_per_sqm=price_per_sqm,
        income_growth_rate=income_growth_rate
    )

    # Issue green mortgages
    initial_green_value_sum = portfolio[GREEN_VALUE_KEY].sum()
    portfolio = issue_green_mortgages(
        portfolio=portfolio,
        green_loan_funds=green_loan_funds,
        b2a_renovation_cost=b2a_renovation_cost,
        price_per_sqm=price_per_sqm,
        energy_cost_per_sqm=energy_cost_per_sqm,
        not_eligible_epcs=not_eligible_epcs,
        is_deny=is_deny,
    )
    loaned_out_green_funds = portfolio[GREEN_VALUE_KEY].sum() - initial_green_value_sum

    # CONTINUE UPDATING INDICATORS FOLLOWING ISSUANCE OF NEW GREEN LOANS:
    update_ratios(portfolio)

    portfolio[PROB_OF_DEFAULT_KEY] = portfolio.apply(
        lambda row: calculate_probability_of_default(
            total_debt_to_income=row[DEBT_TO_INCOME_KEY],
            energy_expenses_to_income=row[ENERGY_EXPENSE_TO_INCOME_KEY],
            loan_to_value_ratio=row[LTV_KEY],
            risk_free_rate=risk_free_rate
        ), axis=1
    )

    # Issue new loans based on the available funds and MAX LTV ratio
    new_loans = generate_mortgage_portfolio(
        new_loan_funds,
        mean_income=MEAN_INCOME * (1 + income_growth_rate),
        base_distribution=epc_distribution,
        energy_cost_per_sqm=energy_cost_per_sqm,
        price_per_sqm=price_per_sqm,
        risk_free_rate=risk_free_rate,
    )
    loaned_out_funds = new_loans[LOAN_VALUE_KEY].sum()
    portfolio = pd.concat([portfolio, new_loans], ignore_index=True)

    return {
        'portfolio': portfolio,
        'cash': total_inflow - loaned_out_funds - loaned_out_green_funds
    }
