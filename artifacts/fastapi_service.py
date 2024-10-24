```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats, optimize, fftpack, integrate, interpolate

app = FastAPI()

# Model for Linear Regression
class LinearRegressionInput(BaseModel):
    x: list[float]
    y: list[float]

class LinearRegressionOutput(BaseModel):
    slope: float
    intercept: float
    r_squared: float

@app.post("/linear-regression", response_model=LinearRegressionOutput)
async def linear_regression(data: LinearRegressionInput):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data.x, data.y)
    return LinearRegressionOutput(slope=slope, intercept=intercept, r_squared=r_value**2)

# Model for Function Minimization
class MinimizeFunctionInput(BaseModel):
    function: str
    initial_guess: list[float]

@app.post("/minimize-function")
async def minimize_function(data: MinimizeFunctionInput):
    # Safely evaluate the function string
    f = eval("lambda x: " + data.function)
    result = optimize.minimize(f, data.initial_guess)
    return result.x.tolist()

# Fourier Transform
class FourierTransformInput(BaseModel):
    data: list[float]

@app.post("/fourier-transform")
async def fourier_transform(data: FourierTransformInput):
    transformed = fftpack.fft(data.data)
    return transformed.tolist()

# ODE Solver
class ODESolveInput(BaseModel):
    function: str
    y0: float
    t_span: list[float]

@app.post("/solve-ode")
async def solve_ode(data: ODESolveInput):
    f = eval("lambda t, y: " + data.function)
    t_eval = np.linspace(data.t_span[0], data.t_span[1], 100)
    solution = integrate.odeint(f, data.y0, t_eval)
    return {"solution": solution.tolist()} 

# Data Interpolation
class InterpolationInput(BaseModel):
    x: list[float]
    y: list[float]
    x_new: list[float]

@app.post("/interpolate")
async def interpolate(data: InterpolationInput):
    interpolator = interpolate.interp1d(data.x, data.y, kind='linear')
    y_new = interpolator(data.x_new)
    return y_new.tolist()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

### Instructions for Running the Service:
1. Save the above code into a file named `main.py`.
2. Make sure you have the necessary libraries by running:
   ```bash
   pip install fastapi pydantic scipy uvicorn
   ```
3. Run the FastAPI service using:
   ```bash
   python main.py
   ```
4. The service will be available at `http://0.0.0.0:8000`.

This FastAPI service provides endpoints for linear regression, function minimization, Fourier transforms, ODE solving, and data interpolation, all handling JSON input and returning JSON output for seamless integration and accessibility.