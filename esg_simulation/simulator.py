import pandas as pd
import numpy as np

from typing import List, Dict, Any

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
)


# Constants:
MAX_SALE_LOSS = 0.4  # max fraction of value lost during sale.
MIN_SALE_LOSS = 0.1  # min fraction of value lost during sale.
PREPAYMENT_PROBABILITY = 0.03  # rate at which prepayments happen [0-1]
INCOME_GROWTH = 0.02  # rate at which income grows [0-1]


def update_income(
    income: float,
    income_growth_rate: float,
    prob_default: float,
) -> float:
    """
    Update the income, expenses, and property value at the end of a year.

    Parameters:
        income (float): Current income.
        prob_default (float): Probability of default
        income_growth_rate (float): average rate at which income grows [0-1].

    Returns:
        float: Updated income.
    """

    if np.random.random() < prob_default:  # assume default manifests as loss of all income
        income = 0
    else:
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
    prob_default: probability of default
    loan_value: current value of the loan
    property_value: current value of the property

    Returns:
        Default repayment. If 0 no default was observed.
    """
    if income < expenses + interest_charge + principal_repayment_charge:  # if unable to service obligations
        sale_loss = np.random.uniform(min_sale_loss, max_sale_loss)
        value_kept = 1 - sale_loss
        return min(value_kept * property_value, loan_value)  # cannot get more than loan value
    else:  # else not default
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


def simulate_year(
        portfolio: pd.DataFrame,
        new_loan_fraction: float = 0.5,
        green_loan_fraction: float = 0.5,
        risk_free_rate: float = 0.03,
        price_per_sqm: float = PRICE_PER_SQM,
        energy_cost_per_sqm: float = ENERGY_COST_PER_SQM,
        income_growth_rate: float = INCOME_GROWTH,
        epc_distribution: List[float] = EPC_DISTRIBUTION,
        cash_inflow: float = 0
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
        risk_free_rate: risk-free rate [0-1].
        energy_cost_per_sqm: The base energy rate per sqm.
        price_per_sqm (float): Mean property price per square meter.
        income_growth_rate (float): Mean income growth rate.
        epc_distribution (List[float]): EPC distribution of houses in the population.
        cash_inflow (float): 'extra' cash available for distribution. Either from the
            previous period or from net customer inflows. Note can be negative.

    Returns:
        Dict:
            'portfolio' : pd.DataFrame - The updated mortgage portfolio after a year
            'cash': float - cash that could not be dispersed.
    """

    # UPDATE VALUES IN EXISTING PORTFOLIO:
    portfolio[INCOME_KEY] = portfolio.apply(
        lambda row: update_income(
            row[INCOME_KEY],
            income_growth_rate=income_growth_rate,
            prob_default=row[PROB_OF_DEFAULT_KEY]
        ), axis=1,
    )
    portfolio[ENERGY_EXPENSES_KEY] = portfolio.apply(
        lambda row: calculate_energy_expenses(
            epc_label=row[EPC_LABEL_KEY],
            floor_plan=row[FLOOR_PLAN_KEY],
            energy_cost_per_sqm=energy_cost_per_sqm,
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
    portfolio[YEARS_LEFT_KEY] = portfolio.apply(
        lambda row: max(row[YEARS_LEFT_KEY] - 1, 0), axis=1
    )
    portfolio[GREEN_YEARS_LEFT_KEY] = portfolio.apply(
        lambda row: max(row[GREEN_YEARS_LEFT_KEY] - 1, 0), axis=1
    )

    # Managing defaults and prepayments
    defaults = portfolio.apply(
        lambda row: simulate_default(
            expenses=row[ENERGY_EXPENSES_KEY],
            income=row[INCOME_KEY],
            interest_charge=(row[YEARLY_INTEREST_CHARGE_KEY] * row[LOAN_VALUE_KEY]),
            principal_repayment_charge=row[YEARLY_PRINCIPAL_REPAYMENT_KEY],
            loan_value=row[LOAN_VALUE_KEY],
            property_value=row[PROPERTY_VALUE_KEY]
        ), axis=1
    )
    # where default happens cancel the loan
    portfolio.loc[defaults != 0, LOAN_VALUE_KEY] = 0

    prepayments = portfolio.apply(
        lambda row: prepay_loan(
            row[LOAN_VALUE_KEY], risk_free_rate
        ), axis=1
    )
    # where prepayment happens cancel the loan
    portfolio.loc[prepayments != 0, LOAN_VALUE_KEY] = 0

    # Remove rows with Loan_value equal to 0 from 'portfolio'
    portfolio = portfolio[portfolio[LOAN_VALUE_KEY] != 0]

    # Determine available funds: note interest charge is a prc and not nominal value
    total_inflow = portfolio[YEARLY_PRINCIPAL_REPAYMENT_KEY].sum() \
        + (portfolio[YEARLY_INTEREST_CHARGE_KEY] * portfolio[LOAN_VALUE_KEY]).sum() \
        + portfolio[GREEN_YEARLY_REPAYMENT_KEY] \
        + (portfolio[GREEN_INTEREST_CHARGE_KEY] * portfolio[GREEN_VALUE_KEY]).sum() \
        + prepayments.sum() + defaults.sum() \
        + cash_inflow

    # TODO: add stochasticity.
    new_loan_funds = total_inflow * new_loan_fraction
    green_loan_funds = total_inflow * green_loan_fraction

    # CONTINUE UPDATING VALUES IN EXISTING PORTFOLIO:
    portfolio[YEARLY_INTEREST_CHARGE_KEY] -= (portfolio[RISK_FREE_RATE_KEY] - risk_free_rate)  # if rf up rates go up.
    portfolio[LOAN_VALUE_KEY] -= portfolio[YEARLY_PRINCIPAL_REPAYMENT_KEY]
    portfolio[GREEN_INTEREST_CHARGE_KEY] = portfolio.apply(
        lambda row:  max(
            row[GREEN_INTEREST_CHARGE_KEY] - (row[RISK_FREE_RATE_KEY] - risk_free_rate),
            0
        ), axis=1
    )
    portfolio[GREEN_VALUE_KEY] = portfolio.apply(
        lambda row:  max(
            row[GREEN_VALUE_KEY] - row[GREEN_YEARLY_REPAYMENT_KEY],
            0
        ), axis=1
    )
    portfolio[RISK_FREE_RATE_KEY] = risk_free_rate

    # Issue green mortgages
    # Step 1: Filter Rows
    eligible_loans = portfolio[(portfolio[EPC_LABEL_KEY] not in ['A']) & (portfolio[GREEN_VALUE_KEY] == 0)].copy()

    # Step 2: Iterate and Apply Function
    # Store the initial sum of GREEN_VALUE_KEY to compute the difference later.
    initial_green_value_sum = portfolio[GREEN_VALUE_KEY].sum()

    for idx, row in eligible_loans.iterrows():
        # Check whether we have enough funds to issue another green loan
        # Compute the difference in GREEN_VALUE_KEY sum to determine how much has been loaned out so far.
        loaned_out_green_funds = portfolio[GREEN_VALUE_KEY].sum() - initial_green_value_sum

        if loaned_out_green_funds < green_loan_funds:
            # Temporary update the row using issue_green_mortgage
            updated_row = issue_green_mortgage(row)  # Add necessary parameters if needed

            # Calculate the potential new loaned out funds after this iteration
            potential_new_loaned_out_funds = loaned_out_green_funds + updated_row[GREEN_VALUE_KEY]

            # Check that by updating with the new value, we're not exceeding green_loan_funds
            if potential_new_loaned_out_funds <= green_loan_funds:
                # Update the actual DataFrame
                portfolio.loc[idx] = updated_row
            else:
                # If the updated_value cannot be accommodated, you may break the loop
                # or continue to the next iteration based on your use case.
                pass
        else:
            # If we have issued all available funds, exit the loop.
            break
    loaned_out_green_funds = portfolio[GREEN_VALUE_KEY].sum() - initial_green_value_sum

    # Recalculate LTV, Debt_to_Income, Probability_of_Default, and Yearly_Nominal_Interest_Charge
    # for all loans (not just loans that received green loans).
    total_debt = portfolio[LOAN_VALUE_KEY] + portfolio[GREEN_VALUE_KEY]
    portfolio[DEBT_TO_INCOME_KEY] = total_debt / portfolio[INCOME_KEY]
    portfolio[LTV_KEY] = total_debt / portfolio[PROPERTY_VALUE_KEY]
    portfolio[ENERGY_EXPENSE_TO_INCOME_KEY] = portfolio[ENERGY_EXPENSES_KEY] / portfolio[INCOME_KEY]
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
        mean_income=MEAN_INCOME * (1 + INCOME_GROWTH),
        base_distribution=epc_distribution,
        energy_cost_per_sqm=energy_cost_per_sqm,
        price_per_sqm=price_per_sqm,
        risk_free_rate=risk_free_rate,
    )
    loaned_out_funds = new_loans[LOAN_VALUE_KEY].sum()
    portfolio = pd.concat([portfolio, new_loans], ignore_index=True)

    return {
        'portfolio': portfolio,
        'cash': green_loan_funds + new_loan_funds - loaned_out_funds - loaned_out_green_funds
    }
