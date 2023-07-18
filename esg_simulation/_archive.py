import pandas as pd
import numpy as np

# Constants
N_MORTGAGES = 10000
MAX_LTV = 0.8  # Loan-to-value ratio
LOAN_DURATION_YEARS = 30  # assuming fixed 30 years loans -> important in repayment plan

# EPC score distribution (assumed, replace with real data if available)
EPC_SCORES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
EPC_DISTRIBUTION = [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]
EPC_SCORE_MAP = {
    'A': [0, 85], 'B': [85, 170], 'C': [170, 255], 'D': [255, 340],
    'E': [340, 425], 'F': [425, 510], 'G': [510, 1000]
}

DEBT_TO_INCOME_DISTRIBUTION = np.random.normal(loc=1.36, scale=0.2, size=10000)
RISK_FREE_RATE = 0.03
PD_BETA = 0.05  # multiplies the prob of default when determining credit_spread

PREPAYMENT_PROBABILITY = 0.005
PROPERTY_SALE_LOSS = 0.4  # [0-1]: indicates fraction of property value lost during sale.

# Function to calculate property value (assumed model, replace with a real model if available)
def calculate_property_value(epc_score, floor_plan):
    epc_factor = {'A': 1.0, 'B': 0.9, 'C': 0.8, 'D': 0.7, 'E': 0.6, 'F': 0.5, 'G': 0.4}[epc_score]
    random_factor = np.random.uniform(0.9, 1.1)  # random element
    return floor_plan * epc_factor * random_factor * 1000  # assumed price per m2


# Function to calculate loan value
def calculate_loan_value(property_value):
    return property_value * MAX_LTV


def calculate_debt_to_income():
    return np.random.choice(DEBT_TO_INCOME_DISTRIBUTION)


# Function to calculate probability of default (assumed model, replace with a real model if available)
def calculate_probability_of_default(ltv, debt_to_income):
    return 1 / (1 + np.exp(-(0.5 * ltv + 0.5 * debt_to_income + np.random.normal(0, 0.1))))


# Function to calculate nominal interest charge (assumed model, replace with a real model if available)
def calculate_interest_charge(loan_value, prob_default):
    return loan_value * (RISK_FREE_RATE + PD_BETA * prob_default)


def calculate_principal_repayment_charge(loan_value):
    return loan_value / LOAN_DURATION_YEARS


# Generate portfolio
def generate_portfolio(n_mortgages):
    portfolio = pd.DataFrame({
        'EPC_Label': np.random.choice(list(EPC_SCORE_MAP.keys()), size=n_mortgages, p=EPC_DISTRIBUTION),
        'Floor_plan': np.random.normal(loc=100, scale=20, size=n_mortgages)  # assumed average size and standard deviation
    })

    portfolio['EPC_Score'] = portfolio['EPC_Label'].apply(
        lambda x: np.random.randint(EPC_SCORE_MAP[x][0], EPC_SCORE_MAP[x][1] + 1)
    )
    portfolio['Property_value'] = portfolio.apply(
        lambda row: calculate_property_value(row['EPC_Label'], row['Floor_plan']), axis=1
    )
    portfolio['Loan_value'] = portfolio['Property_value'].apply(calculate_loan_value)
    portfolio['Years_Left'] = LOAN_DURATION_YEARS
    portfolio['LTV'] = portfolio['Loan_value'] / portfolio['Property_value']
    portfolio['Debt_to_Income'] = portfolio.apply(lambda _: calculate_debt_to_income(), axis=1)
    portfolio['Probability_of_Default'] = portfolio.apply(
        lambda row: calculate_probability_of_default(row['LTV'], row['Debt_to_Income']), axis=1
    )
    portfolio['Yearly_Nominal_Interest_Charge'] = portfolio.apply(
        lambda row: calculate_interest_charge(row['Loan_value'], row['Probability_of_Default']), axis=1
    )
    portfolio['Yearly_Principal_Repayment'] = portfolio.apply(
        lambda row: calculate_principal_repayment_charge(row['Loan_value']), axis=1
    )
    return portfolio


def get_principal_repayment(loan_value, principal_repayment):
    # Allow for the possibility of prepayment.


def repay_loan(loan_value, principal_repayment):
    # this is a simplification: in reality, loan repayment would also depend on the interest rate
    return loan_value - principal_repayment


def simulate_default(prob_default, loan_value, property_value):
    if np.random.random() < prob_default:
        return min(PROPERTY_SALE_LOSS * property_value, loan_value)
    else:
        return loan_value


def issue_green_mortgage(row, green_loan_value):
    """Issue a green mortgage and update loan details"""
    # Assume EPC score improves by one level and property value increases
    improved_epc_score = chr(ord(row['EPC_Label']) - 1) if row['EPC_Label'] != 'A' else 'A'
    improved_property_value = row['Property_value'] * np.random.uniform(1, 1.15)

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


def simulate_year(portfolio, new_loan_fraction=0.5, green_loan_fraction=0.5):
    # Repay loans and possibly default
    portfolio['Years_Left'] = portfolio['Years_Left'] - 1
    portfolio['Loan_value'] = portfolio.apply(lambda row: repay_loan(row['Loan_value'], row['Years_Left']), axis=1)
    portfolio['Loan_value'] = portfolio.apply(
        lambda row: simulate_default(row['Probability_of_Default'], row['Loan_value']), axis=1
    )

    # Existing loan repayments and interest
    total_inflow = portfolio['Loan_value'].sum() / LOAN_DURATION_YEARS + portfolio['Yearly_Nominal_Interest_Charge'].sum()
    available_for_new_loans = total_inflow * new_loan_fraction
    available_for_green_loans = total_inflow * green_loan_fraction

    # Issue new loans - Incorrect: this should be based on the MAX LTV and Property Value.
    num_new_loans = int(available_for_new_loans / portfolio['Loan_value'].mean())
    new_loans = generate_portfolio(num_new_loans)
    portfolio = pd.concat([portfolio, new_loans], ignore_index=True)

    # Issue green mortgages
    green_loan_value = portfolio['Loan_value'].mean()
    green_mortgages_candidates = portfolio[portfolio['EPC_Label'] > 'B'].sample(frac=green_loan_fraction)
    green_mortgages_candidates = green_mortgages_candidates.apply(
        lambda row: issue_green_mortgage(row, green_loan_value), axis=1
    )
    portfolio.update(green_mortgages_candidates)

    # Adjust remaining features:
    portfolio['LTV'] = portfolio['Loan_value'] / portfolio['Property_value']
    portfolio['Probability_of_Default'] = portfolio.apply(lambda row: calculate_probability_of_default(row['LTV'], row['Debt_to_Income']), axis=1)
    portfolio['Yearly_Nominal_Interest_Charge'] = portfolio.apply(
        lambda row: calculate_interest_charge(row['Loan_value'], row['Probability_of_Default'], row['LTV']
    ), axis=1)
    return portfolio
