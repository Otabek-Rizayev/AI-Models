# Dastlabki ma'lumotlar
x_soat = [1.0, 2.0, 3.0]
y_baho = [3.0, 5.0, 7.0]  # Bu safar ideal model: y = 2x + 1

# Dastlabki og‘irlik va bias
w = 0.0
b = 0.0

# Model funksiyasi
def forward(x):
    return x * w + b

# Xatolik funksiyasi (MSE)
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# Gradientlar: dL/dw va dL/db
def gradient(x, y):
    y_pred = forward(x)
    grad_w = 2 * x * (y_pred - y)  # dL/dw
    grad_b = 2 * (y_pred - y)      # dL/db
    return grad_w, grad_b

# O‘rganish tezligi
lr = 0.01

# Trening sikli
for epoch in range(10):
    total_loss = 0
    for x_val, y_val in zip(x_soat, y_baho):
        grad_w, grad_b = gradient(x_val, y_val)
        w -= lr * grad_w
        b -= lr * grad_b
        total_loss += loss(x_val, y_val)
    print(f"Epoch {epoch+1}: w={round(w, 3)}, b={round(b, 3)}, loss={round(total_loss, 3)}")

# Bashorat
print("Bashorat (training dan keyin): 4 soat = ", round(forward(4), 3))
