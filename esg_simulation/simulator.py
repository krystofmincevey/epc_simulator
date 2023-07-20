import pandas as pd
import numpy as np

from typing import Tuple

from .initialisor import (
    calculate_probability_of_default,
    calculate_interest_charge,
    generate_portfolio
)

print("Hello World.")

# Constants:
SALE_LOSS = 0.4  # fraction of value lost during sale.
INFLATION = 0.02  # Rate at which income, expenses should be increased. Default is 0.02 (2%).


def update_yearly_income_and_expenses(
    income: float, expenses: float,
    inflation: float = INFLATION
) -> Tuple[float, float]:
    """
    Update the income and expenses at the end of a year.

    Parameters:
        income (float): Current income.
        expenses (float): Current expenses.
        inflation (float): yearly inflation rate.

    Returns:
        Tuple[float, float]: Updated income and expenses.
    """
    income *= 1 + np.log(inflation)
    expenses *= 1 + inflation
    return income, expenses


def simulate_default(
        prob_default: float, loan_value: float, property_value: float,
        sale_loss: float = SALE_LOSS,
) -> float:
    """
    Simulate the default event

    Args:
    prob_default: probability of default
    loan_value: current value of the loan
    property_value: current value of the property

    Returns:
    new_loan_value: updated value of the loan after simulating default
    """
    if np.random.random() < prob_default:
        value_kept = 1 - sale_loss
        return min(value_kept * property_value, loan_value)
    else:
        return loan_value


def repay_loan(loan_value: float, years_left: int) -> float:
    """
    Calculate new loan value after repayment

    Args:
    loan_value: current value of the loan
    years_left: remaining years on the loan

    Returns:
    new_loan_value: updated value of the loan after repayment
    """
    # This is a simplification: in reality, loan repayment would also depend on the interest rate
    # Adding in the possibility of prepayment
    prepayment = np.random.choice([0, 1], p=[0.95, 0.05])
    prepayment_value = loan_value * np.random.uniform(0.05, 0.2) if prepayment else 0

    return loan_value - min(loan_value / years_left + prepayment_value, loan_value)


def issue_green_mortgage(row: pd.Series, green_loan_value: float) -> pd.Series:
    """
    Issue a green mortgage and update loan details

    Args:
    row: pandas Series with details of a mortgage
    green_loan_value: value of the green loan

    Returns:
    row: pandas Series with updated details of the mortgage
    """
    # Assume EPC score improves by one level and property value increases
    improved_epc_score = chr(ord(row['EPC_Label']) - 1) if row['EPC_Label'] != 'A' else 'A'
    improved_property_value = row['Property_value'] * np.random.uniform(
        1.05, 1.15
    )  # Factoring in random and deterministic increase

    # Recalculate loan value, LTV, probability of default and interest charge
    updated_loan_value = row['Loan_value'] + green_loan_value
    updated_ltv = updated_loan_value / improved_property_value
    updated_probability_of_default = calculate_probability_of_default(updated_ltv, row['Debt_to_Income'])
    updated_interest_charge = calculate_interest_charge(updated_loan_value, updated_probability_of_default, updated_ltv)

    # Update the row values
    row['EPC_Label'] = improved_epc_score
    row['Property_value'] = improved_property_value
    row['Loan_value'] = updated_loan_value
    row['LTV'] = updated_ltv
    row['Probability_of_Default'] = updated_probability_of_default
    row['Yearly_Nominal_Interest_Charge'] = updated_interest_charge

    return row


def simulate_year(portfolio: pd.DataFrame, new_loan_fraction=0.5, green_loan_fraction=0.5) -> pd.DataFrame:
    """
    Simulates a year in the mortgage portfolio. Repays loans, issues new loans and green mortgages,
    and updates the relevant financial and property details.

    Parameters:
        portfolio (pd.DataFrame): The initial mortgage portfolio.
        new_loan_fraction (float, optional): The fraction of available funds to be used for new loans. Default is 0.5.
        green_loan_fraction (float, optional): The fraction of available funds to be used for green mortgages. Default is 0.5.

    Returns:
        pd.DataFrame: The updated mortgage portfolio after a year.
    """

    # Repay loans and possibly default
    portfolio['Years_Left'] = portfolio['Years_Left'] - 1
    portfolio['Loan_value'] = portfolio.apply(lambda row: repay_loan(row['Loan_value'], row['Years_Left']), axis=1)
    portfolio['Loan_value'] = portfolio.apply(
        lambda row: simulate_default(
            row['Probability_of_Default'], row['Loan_value'], row['Property_value']
        ), axis=1
    )

    # Existing loan repayments and interest
    total_inflow = portfolio['Loan_value'].sum() / LOAN_DURATION_YEARS + portfolio[
        'Yearly_Nominal_Interest_Charge'].sum()
    available_for_new_loans = total_inflow * new_loan_fraction
    available_for_green_loans = total_inflow * green_loan_fraction

    # Issue new loans based on the available funds and MAX LTV ratio
    max_new_loan_value = total_inflow * new_loan_fraction
    new_loans = generate_portfolio(int(max_new_loan_value / portfolio['Loan_value'].mean()))
    new_loans_cumsum = new_loans['Loan_value'].cumsum()
    new_loans = new_loans.loc[new_loans_cumsum <= max_new_loan_value]
    portfolio = pd.concat([portfolio, new_loans], ignore_index=True)

    # Issue green mortgages
    green_loan_value = portfolio['Loan_value'].mean()
    green_mortgages_candidates = portfolio[portfolio['EPC_Label'] > 'B'].sample(frac=green_loan_fraction)
    green_mortgages_candidates = green_mortgages_candidates.apply(
        lambda row: issue_green_mortgage(row, green_loan_value), axis=1
    )
    portfolio.update(green_mortgages_candidates)

    # Update income, expenses, and property value
    portfolio[['Income', 'Expenses']] = portfolio.apply(
        lambda row: update_yearly_income_and_expenses(row['Income'], row['Expenses']), axis=1, result_type='expand'
    )
    portfolio['Property_value'] *= 1 + np.random.uniform(0, 0.05)  # Property value increases by 0% to +5%

    # Recalculate LTV, Debt_to_Income, Probability_of_Default, and Yearly_Nominal_Interest_Charge
    portfolio['LTV'] = portfolio['Loan_value'] / portfolio['Property_value']
    portfolio['Debt_to_Income'] = portfolio['Loan_value'] / (portfolio['Income'] - portfolio['Expenses'])
    portfolio['Probability_of_Default'] = portfolio.apply(
        lambda row: calculate_probability_of_default(row['LTV'], row['Debt_to_Income']), axis=1
    )
    portfolio['Yearly_Nominal_Interest_Charge'] = portfolio.apply(
        lambda row: calculate_interest_charge(row['Loan_value'], row['Probability_of_Default'], row['LTV']), axis=1
    )

    return portfolio