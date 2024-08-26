from sklearn.linear_model import LinearRegression

# Example model creation (you would typically load a saved model)
reg = LinearRegression()
reg.fit([[1], [2], [3]], [2, 4, 6])  # Dummy data
