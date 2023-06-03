from .simulator import generate_portfolio


def main(n_mortgages):
    portfolio_evolution = []
    initial_portfolio = generate_portfolio(n_mortgages)


if __name__ == "__main__":
    main()