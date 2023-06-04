import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from scipy.stats import lognorm
from sklearn.linear_model import LogisticRegression

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Distribution Plots"),

    html.Div([
        html.H3("Log-Normal Distribution"),
        html.Label("Mean"),
        dcc.Slider(
            id="lognormal-mean-slider",
            min=0,
            max=50,
            step=0.1,
            value=1,
            marks={i: str(i) for i in range(0, 51, 5)}
        ),
        html.Label("Standard Deviation"),
        dcc.Slider(
            id="lognormal-std-dev-slider",
            min=0,
            max=10,
            step=0.1,
            value=0.5,
            marks={i: str(i) for i in range(11)}
        ),
        dcc.Graph(id="lognormal-plot")
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.H3("Logistic Regression"),
        html.Label("Feature 1 Beta"),
        dcc.Slider(
            id="logistic-beta1-slider",
            min=-1,
            max=1,
            step=0.1,
            value=0.5,
            marks={i / 10: str(i / 10) for i in range(-10, 11)}
        ),
        html.Label("Feature 2 Beta"),
        dcc.Slider(
            id="logistic-beta2-slider",
            min=-1,
            max=1,
            step=0.1,
            value=-0.5,
            marks={i / 10: str(i / 10) for i in range(-10, 11)}
        ),
        html.Label("Constant"),
        dcc.Slider(
            id="logistic-constant-slider",
            min=-5,
            max=1,
            step=1,
            value=0,
            marks={i: str(i) for i in range(-5, 2)}
        ),
        dcc.Graph(id="logistic-plot")
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
])


# Callback for updating the log-normal distribution plot
@app.callback(
    Output("lognormal-plot", "figure"),
    Input("lognormal-mean-slider", "value"),
    Input("lognormal-std-dev-slider", "value")
)
def update_lognormal_plot(mean, std_dev):
    x = np.linspace(0.01, 100, 10000)
    y = lognorm.pdf(x, std_dev, scale=np.exp(mean))

    fig = go.Figure(data=go.Scatter(x=x, y=y))
    fig.update_layout(
        title='Log-Normal Distribution',
        xaxis_title='X',
        yaxis_title='Probability Density',
        width=500,
        height=400
    )

    return fig


# Callback for updating the logistic regression plot
@app.callback(
    Output("logistic-plot", "figure"),
    Input("logistic-beta1-slider", "value"),
    Input("logistic-beta2-slider", "value"),
    Input("logistic-constant-slider", "value")
)
def update_logistic_plot(beta1, beta2, constant):
    # Restrict beta parameters between -1 and 1
    beta1 = max(min(beta1, 1), -1)
    beta2 = max(min(beta2, 1), -1)

    x1 = np.random.uniform(0, 1, 100)
    x2 = np.random.uniform(0, 1, 100)
    y = np.random.binomial(1, logistic_func(constant + beta1 * x1 + beta2 * x2))
    model = LogisticRegression()
    model.fit(np.column_stack((x1, x2)), y)
    # Generate grid of points for visualization
    xx1, xx2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    zz = model.predict_proba(np.column_stack((xx1.ravel(), xx2.ravel())))[:, 1].reshape(xx1.shape)

    # Create 3D surface plot
    fig = go.Figure(data=go.Surface(x=xx1, y=xx2, z=zz))
    fig.update_layout(
        title='Logistic Regression',
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Probability',
        ),
        width=700,
        height=500,
    )

    return fig


def logistic_func(x):
    return 1 / (1 + np.exp(-x))


# Function to start the Dash application
def start_dash_app():
    app.run_server(debug=True)


# Example usage
if __name__ == "__main__":
    start_dash_app()
